import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PolicyNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int
    ):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hiddens = [hidden_dim] * hidden_layers
        self.dims = [state_dim] + self.hiddens + [action_dim]
        self.build()

    def build(self):
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i < len(self.dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PolicyDataset(Dataset):
    """
    Dataset of (state) -> action.
    Input:
      states:   (N, state_dim)
      actions:  (N, action_dim)
    Output:
      actions: (N, action_dim)
    """

    def __init__(self, states: torch.Tensor, actions: torch.Tensor):
        assert (
            states.shape[0] == actions.shape[0]
        ), "States and actions must have belong to the same rollout ma boy"
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class PolicyTrainer:
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        pass
