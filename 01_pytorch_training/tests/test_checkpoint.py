import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from trainscale_training.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from trainscale_training.models import make_mlp


def test_checkpoint_restores_complete_training_and_rng_state(tmp_path: Path) -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    data_generator = torch.Generator().manual_seed(9)
    model = make_mlp()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cpu")
    state = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        global_step=17,
        data_generator=data_generator,
        config={"seed": 3},
        metrics={"loss": 0.5},
    )
    path = tmp_path / "state.pt"
    save_checkpoint(path, state)
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    expected_data = torch.rand(1, generator=data_generator)

    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        data_generator=data_generator,
    )
    assert restored["epoch"] == 2
    assert restored["global_step"] == 17
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    torch.testing.assert_close(torch.rand(1), expected_torch)
    torch.testing.assert_close(torch.rand(1, generator=data_generator), expected_data)


def test_resume_matches_continuous_next_step(tmp_path: Path) -> None:
    torch.manual_seed(11)
    features = torch.randn(8, 16)
    targets = torch.randint(0, 4, (8,))
    criterion = nn.CrossEntropyLoss()
    continuous_model = make_mlp()
    continuous_optimizer = torch.optim.SGD(continuous_model.parameters(), lr=0.05, momentum=0.9)
    first_loss = criterion(continuous_model(features), targets)
    first_loss.backward()
    continuous_optimizer.step()
    continuous_optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        build_checkpoint(
            model=continuous_model,
            optimizer=continuous_optimizer,
            epoch=1,
            global_step=1,
        ),
    )

    continuous_loss = criterion(continuous_model(features), targets)
    continuous_loss.backward()
    continuous_optimizer.step()

    resumed_model = make_mlp()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.05, momentum=0.9)
    load_checkpoint(path, model=resumed_model, optimizer=resumed_optimizer)
    resumed_loss = criterion(resumed_model(features), targets)
    resumed_loss.backward()
    resumed_optimizer.step()

    torch.testing.assert_close(resumed_loss, continuous_loss)
    for resumed, continuous in zip(
        resumed_model.parameters(), continuous_model.parameters(), strict=True
    ):
        torch.testing.assert_close(resumed, continuous)
