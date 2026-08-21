from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from trainscale_training.data import make_classification_dataset
from trainscale_training.engine import seed_everything, train_one_epoch, validate
from trainscale_training.models import SmallCifarCNN, make_mlp


def test_single_batch_overfits() -> None:
    torch.manual_seed(0)
    loader = DataLoader(make_classification_dataset(16, 8, 3, seed=1), batch_size=16)
    model = make_mlp(8, 32, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    criterion = nn.CrossEntropyLoss()
    initial = validate(model, loader, criterion, torch.device("cpu")).loss
    for _ in range(100):
        final = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert final.loss < initial * 0.1
    assert final.accuracy == 1.0


def test_train_and_validation_modes_and_sample_counts() -> None:
    loader = DataLoader(make_classification_dataset(20), batch_size=6)
    model = make_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train_metrics = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert model.training
    valid_metrics = validate(model, loader, criterion, torch.device("cpu"))
    assert not model.training
    assert train_metrics.samples == valid_metrics.samples == 20


def test_accumulation_matches_effective_batch_update() -> None:
    torch.manual_seed(5)
    features = torch.randn(8, 16)
    targets = torch.randint(0, 4, (8,))
    dataset = TensorDataset(features, targets)
    reference = make_mlp()
    accumulated = deepcopy(reference)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(
        reference,
        DataLoader(dataset, batch_size=8, shuffle=False),
        reference_optimizer,
        criterion,
        torch.device("cpu"),
    )
    metrics = train_one_epoch(
        accumulated,
        DataLoader(dataset, batch_size=2, shuffle=False),
        accumulated_optimizer,
        criterion,
        torch.device("cpu"),
        accumulation_steps=4,
    )
    assert metrics.optimizer_steps == 1
    for actual, expected in zip(accumulated.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_seed_reproducibility_boundary() -> None:
    first_data = make_classification_dataset(seed=13).tensors
    second_data = make_classification_dataset(seed=13).tensors
    different_data = make_classification_dataset(seed=14).tensors
    torch.testing.assert_close(first_data[0], second_data[0])
    torch.testing.assert_close(first_data[1], second_data[1])
    assert not torch.equal(first_data[0], different_data[0])

    seed_everything(21)
    first_model = make_mlp()
    seed_everything(21)
    second_model = make_mlp()
    for first, second in zip(first_model.parameters(), second_model.parameters(), strict=True):
        torch.testing.assert_close(first, second)


def test_cifar_model_output_shape_and_mode_sensitive_buffers() -> None:
    model = SmallCifarCNN()
    inputs = torch.randn(2, 3, 32, 32)
    assert model(inputs).shape == (2, 10)
    model.eval()
    with torch.inference_mode():
        assert model(inputs).shape == (2, 10)
