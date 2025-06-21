import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DynamicsNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, hidden_layers: int
    ):
        super(DynamicsNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hiddens = [hidden_dim] * hidden_layers
        self.dims = [state_dim + action_dim] + self.hiddens + [state_dim]
        self.build()

    def build(self):
        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if i < len(self.dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat((action, state), dim=-1)
        x = x.float()
        return self.network(x)


class DynamicsDataset(Dataset):
    """
    Dataset of (state, action) -> delta to the next state.
    Input:
      states:   (N, state_dim)
      actions:  (N, action_dim)
      next_states:   (N, state_dim)
    Output:
      deltas = next_states - states: (N, state_dim)
    """

    def __init__(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ):
        assert (
            states.shape[0] == actions.shape[0] == next_states.shape[0]
        ), "All tensors must have belong to the same rollout moron"
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.deltas = next_states - states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.deltas[idx]


class DynamicsTrainer:
    def __init__(
        self,
        model: DynamicsNetwork,
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
        for epoch in tqdm(range(self.epochs), desc="Training Dynamics"):
            total_loss = 0.0
            for states, actions, deltas in self.dataloader:
                states, actions, deltas = (
                    states.to(self.device).float(),
                    actions.to(self.device).float(),
                    deltas.to(self.device).float(),
                )
                self.optimizer.zero_grad()
                predictions = self.model(actions, states)
                loss = self.loss_fn(predictions, deltas)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * states.size(0)

            total_loss /= len(self.dataloader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {total_loss :.4f}")


if __name__ == "__main__":

    states, actions, next_states = (
        torch.randn(1000, 50),
        torch.randn(1000, 50),
        torch.randn(1000, 50),
    )

    dataset = DynamicsDataset(states, actions, next_states)
    model = DynamicsNetwork(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        hidden_dim=128,
        hidden_layers=4,
    )
    trainer = DynamicsTrainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        dataloader=DataLoader(dataset, batch_size=32, shuffle=True),
        epochs=5,
    )
    trainer.train()
