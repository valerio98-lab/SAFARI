import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from uncertaintyNetwork import DenoisingAutoencoder
from uncertaintyNetwork import UncertaintyTrainer


@pytest.fixture
def setup_trainer():
    torch.manual_seed(42)
    state_dim = 8
    hidden = [16, 4]
    noise_std = 0.05

    data_clean = torch.randn(100, state_dim)
    # Pre‐compute una coppia clean/noisy fissa
    data_noisy = data_clean + noise_std * torch.randn_like(data_clean)
    dataset = TensorDataset(data_clean, data_noisy)
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    dae = DenoisingAutoencoder(
        input_dim=state_dim, hidden_dim=hidden, noise_std=noise_std
    )
    optimizer = torch.optim.SGD(dae.parameters(), lr=1e-2)
    trainer = UncertaintyTrainer(
        model=dae,
        optimizer=optimizer,
        dataloader=loader,
        epochs=3,
        device="cpu",
    )
    return dae, trainer, loader


def test_output_shape_and_type(setup_trainer):
    dae, _, loader = setup_trainer
    clean, noisy = next(iter(loader))
    print(noisy.type(), clean.type())
    recon = dae.forward(noisy)
    assert isinstance(recon, torch.Tensor)
    assert recon.shape == clean.shape


def test_uncertainty_decreases_after_training(setup_trainer):
    dae, trainer, loader = setup_trainer
    unc_before = trainer.evaluate_denoising_uncertainty(dae, loader)
    trainer.train()
    unc_after = trainer.evaluate_denoising_uncertainty(dae, loader)
    tol = 1e-4
    assert unc_after <= unc_before - tol, (
        f"Expected uncertainty to decrease by at least {tol:.5f}, "
        f"ma è passata da {unc_before:.5f} a {unc_after:.5f}"
    )
