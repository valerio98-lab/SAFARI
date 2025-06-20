import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor

from denoisingAutoencoder import DenoisingAutoencoder


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        epochs: int = 10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in self.dataloader:
                noisy, clean = batch
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
            for batch in dataloader:
                noisy, clean = batch
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                err = model.reconstruction_error(clean)
                errors.append(err)
        return torch.cat(errors).mean().item()
