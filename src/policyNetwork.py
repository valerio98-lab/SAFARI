import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

"""

"""


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
        state = state.float()
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
        self.optimizer: Optimizer = optimizer
        self.dataloader: DataLoader = dataloader
        self.epochs = epochs
        self.device = device
        self.criterion = nn.MSELoss()

    def train(self):
        total_loss = 0
        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Training Policy"):
            for state, action in self.dataloader:
                state = state.to(self.device).float()
                action = action.to(self.device).float()

                self.optimizer.zero_grad()
                action_pred = self.model(state)
                loss = self.criterion(action_pred, action)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * state.size(0)

            avg = total_loss / len(self.dataloader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg :.4f}")


if __name__ == "__main__":

    states, actions = (torch.randn(10000, 50), torch.randn(10000, 50))

    dataset = PolicyDataset(states, actions)
    model = PolicyNetwork(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        hidden_dim=128,
        hidden_layers=4,
    )
    trainer = PolicyTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        dataloader=DataLoader(dataset, batch_size=32, shuffle=True),
        epochs=5,
    )
    trainer.train()
