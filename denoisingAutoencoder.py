import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from typing import List

"""
Simple MLP Denoising Autoencoder.
Args:
    input_dim (int): Dimensionality of input state vector.
    hidden_dims (list of int): Sizes of hidden layers.
    noise_std (float): Standard deviation of Gaussian noise added to inputs.
"""


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List[int], noise_std=0.1):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_std = noise_std
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim)

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
    def __init__(self, input_dim: int, hidden_dim: List[int]):
        super(Encoder, self).__init__()
        self.layers = [input_dim] + hidden_dim
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
    def __init__(self, input_dim: int, hidden_dim: List[int]):
        super(Decoder, self).__init__()
        self.layers = [hidden_dim[-1]] + hidden_dim[-2::-1] + [input_dim]
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
