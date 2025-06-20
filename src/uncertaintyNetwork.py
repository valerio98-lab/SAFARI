import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


"""
Simple MLP Denoising Autoencoder. Is the uncertainty network used to estimate the uncertainty of the agent in a 
given state. In a nutshell how far the agent is from the "well-known" distribution of states. 
Args:
    input_dim (int): Dimensionality of input state vector.
    hidden_dims (list of int): Sizes of hidden layers.
    noise_std (float): Standard deviation of Gaussian noise added to inputs.
"""


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, hidden_layers: int, noise_std=0.1
    ):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_std = noise_std
        self.encoder = Encoder(input_dim, hidden_dim, hidden_layers)
        self.decoder = Decoder(input_dim, hidden_dim, hidden_layers)

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        noisy_x = x + (torch.randn_like(x) * self.noise_std) if add_noise else x
        input_encoded = self.encoder(noisy_x)
        return self.decoder(input_encoded)

    def get_noise_std(self) -> float:
        return self.noise_std

    def reconstruction_error(self, x: Tensor) -> Tensor:
        recon = self.forward(x, add_noise=False)
        return ((x - recon) ** 2).sum(dim=1)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int):
        super(Encoder, self).__init__()
        self.hiddens = [hidden_dim] * hidden_layers
        self.layers = [input_dim] + self.hiddens
        self.build()

    def build(self):
        encoder = []
        for i in range(len(self.layers) - 1):
            encoder.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int):
        super(Decoder, self).__init__()
        self.hiddens = [hidden_dim] * hidden_layers
        self.layers = [self.hiddens[-1]] + self.hiddens[-2::-1] + [input_dim]
        self.build()

    def build(self):
        decoder = []
        for i in range(len(self.layers) - 1):
            decoder.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:
                continue
            decoder.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class UncertaintyTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        epochs: int = 10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        self.device = device
        self.model.to(self.device)

    def train(self):
        self.model.train()
        for epoch in tqdm(
            range(self.epochs), desc="Training Uncertainty Network", unit="epoch"
        ):
            total_loss = 0.0
            for noisy, clean in self.dataloader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(noisy)
                loss = self.loss_fn(outputs, clean)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * noisy.size(0)
            total_loss /= len(self.dataloader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

    """ 
    Computes the uncertainty of the denoising autoencoder by evaluating the reconstruction error
    """

    def evaluate_denoising_uncertainty(
        self, model: DenoisingAutoencoder, dataloader: DataLoader
    ) -> Tensor:
        model.eval()
        errors = []
        with torch.no_grad():
            for noisy, clean in dataloader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                err = model.reconstruction_error(clean)
                errors.append(err)
        return torch.cat(errors).mean().item()


if __name__ == "__main__":
    input_dim = 8
    hidden_dim = 128
    hidden_layers = 4
    noise_std = 0.05

    data_clean = torch.randn(10000, input_dim)
    data_noisy = data_clean + noise_std * torch.randn_like(data_clean)
    dataset = TensorDataset(data_noisy, data_clean)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    dae = DenoisingAutoencoder(input_dim, hidden_dim, hidden_layers, noise_std)
    optimizer = torch.optim.Adam(dae.parameters(), lr=1e-3)

    trainer = UncertaintyTrainer(dae, optimizer, dataloader, epochs=5)
    trainer.train()
