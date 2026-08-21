"""Single-device FP32/AMP training, validation, accumulation, and resume."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import make_data
from .models import make_model
from .reporting import environment_record, write_json


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int
    duration_seconds: float
    samples_per_second: float
    optimizer_steps: int = 0


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finish_metrics(
    total_loss: float,
    total_correct: int,
    total_samples: int,
    duration: float,
    optimizer_steps: int = 0,
) -> EpochMetrics:
    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
        duration_seconds=duration,
        samples_per_second=total_samples / duration,
        optimizer_steps=optimizer_steps,
    )


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
) -> EpochMetrics:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer_steps = 0
    use_amp = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)
    _sync(device)
    started = time.perf_counter()
    batch_iterator = iter(batches)
    while group := list(islice(batch_iterator, accumulation_steps)):
        group_samples = sum(targets.shape[0] for _, targets in group)
        for features, targets in group:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, targets)
                backward_loss = loss * (targets.shape[0] / group_samples)
            if use_amp:
                assert scaler is not None
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            batch_size = targets.shape[0]
            total_loss += loss.detach().float().item() * batch_size
            total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        if use_amp:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    _sync(device)
    duration = time.perf_counter() - started
    return _finish_metrics(total_loss, total_correct, total_samples, duration, optimizer_steps)


@torch.inference_mode()
def validate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    _sync(device)
    started = time.perf_counter()
    for features, targets in batches:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        total_loss += loss.float().item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
    _sync(device)
    duration = time.perf_counter() - started
    return _finish_metrics(total_loss, total_correct, total_samples, duration)


def run_training(config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; install the cu128 extra and check driver"
        )
    seed_everything(config.seed)
    device = torch.device(config.device)
    data = make_data(config)
    model = make_model(config).to(device)
    if config.compile_model:
        model.compile(mode="reduce-overhead")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
    )
    use_amp = config.precision == "amp"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if config.resume:
        restored = load_checkpoint(
            config.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_generator=data.generator,
            map_location=device,
        )
        start_epoch = int(restored["epoch"])
        global_step = int(restored["global_step"])

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "environment.json", environment_record())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wall_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            accumulation_steps=config.accumulation_steps,
        )
        global_step += train_metrics.optimizer_steps
        valid_metrics = validate(
            model, data.valid_loader, criterion, device, use_amp=use_amp
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train": asdict(train_metrics),
            "valid": asdict(valid_metrics),
        }
        history.append(row)
        print(
            f"epoch={epoch + 1} train_loss={train_metrics.loss:.4f} "
            f"valid_loss={valid_metrics.loss:.4f} valid_accuracy={valid_metrics.accuracy:.3f} "
            f"train_samples_per_second={train_metrics.samples_per_second:.1f}"
        )
        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            data_generator=data.generator,
            config=config.to_dict(),
            metrics=row,
        )
        save_checkpoint(output / "last.pt", checkpoint)
        write_json(output / "metrics.json", history)

    summary = {
        "experiment_name": config.experiment_name,
        "config": config.to_dict(),
        "environment": environment_record(),
        "history": history,
        "total_wall_seconds": time.perf_counter() - wall_started,
        "global_step": global_step,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    write_json(output / "summary.json", summary)
    return summary
