import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from policyNetwork import PolicyNetwork, PolicyDataset, PolicyTrainer
from dynamicsNetwork import DynamicsNetwork, DynamicsDataset, DynamicsTrainer
from uncertaintyNetwork import DenoisingAutoencoder
import gymnasium as gym

from typing import Union, Dict, Tuple
import numpy as np


"""
Performs active learning by monitoring the uncertainty u(s) and querying the expert when uncertainty exceeds a given threshold.

Args:
    env: Environment object supporting reset(), step(action), and expert_action(state) methods.
    policy_model: Trained policy network.
    dynamics_model: Trained dynamics network.
    autoencoder: Trained autoencoder module.
    states: Initial tensor of states.
    actions: Initial tensor of actions.
    next_states: Initial tensor of next states.
    threshold: Uncertainty threshold for querying the expert.
    max_iterations: Maximum number of episodes to run.
    T: Horizon length for uncertainty computation.
    retrain_epochs: Number of epochs to retrain models after querying.
"""


class ActiveLearning:
    def __init__(
        self,
        env,
        policy_model: PolicyNetwork,
        dynamics_model: DynamicsNetwork,
        uncertainty_model: DenoisingAutoencoder,
        states: torch.Tensor,
        actions: torch.Tensor,
        actions_dim: int,
        state_dim: int,
        next_states: torch.Tensor,
        threshold: float = 0.1,
        safe_threshold: float = 0.05,
        max_iterations: int = 1000,
        retrain_epochs: int = 10,
        cem_iterations: int = 10,
        cem_horizon: int = 5,
        n_samples: int = 100,
        beta=0.5,
        elite_samples: int = 10,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.action_dim = actions_dim
        self.state_dim = state_dim

        self.threshold = threshold
        self.safe_threshold = safe_threshold
        self.max_iterations = max_iterations
        self.horizon = cem_horizon
        self.retrain_epochs = retrain_epochs
        self.cem_iterations = cem_iterations
        self.cem_samples = n_samples
        self.cem_elite = elite_samples
        self.beta = beta

        self.device = device
        self.policy_model = policy_model.to(self.device)
        self.dynamics_model = dynamics_model.to(self.device)
        self.uncertainty_model = uncertainty_model.to(self.device)

    def run(self):
        state, _ = self.env.reset()
        state = self.flatten(state)
        done = False
        step = 0
        while not done and step < self.max_iterations:
            step += 1
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            unc_error = self.compute_uncertainty(s, iterations=10)
            action = self.policy_model(s)

            if unc_error > self.threshold:
                optimal_action = self.cem_planner(s).unsqueeze(0)
                s_optimal = s + self.dynamics_model(s, optimal_action)
                unc_optimal_err = self.compute_uncertainty(s_optimal, iterations=10)
            else:
                optimal_action = action
                unc_optimal_err = unc_error

            if unc_optimal_err > self.safe_threshold:
                expert_action = self.expert_action(state)
                query = True
            else:
                expert_action = (
                    (action + self.beta * optimal_action).cpu().detach().numpy()[0]
                )
                query = False

            next_state, _, terminated, truncated, _ = self.env.step(expert_action)
            done = terminated or truncated
            next_state = self.flatten(next_state)
            if query:
                self._add_sample(s, expert_action, next_state)
                self._retrain()
            s = next_state

    def cem_planner(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs Cross-Entropy Method (CEM) planning to find the best action sequence.
        """
        mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        sigma = torch.ones(self.horizon, self.action_dim, device=self.device) * 0.5

        for _ in range(self.cem_iterations):
            A = torch.normal(
                mu.expand(self.cem_samples, -1, -1),
                sigma.expand(self.cem_samples, -1, -1),
            )
            s = state.repeat(self.cem_samples, 1)
            with torch.no_grad():
                for timestep in range(self.horizon):
                    action = A[:, timestep, :]
                    s = s + self.dynamics_model(action, s)
                recon = self.uncertainty_model(s, add_noise=False)
                cost = self.uncertainty_model.reconstruction_error(s, recon)
            elite_indices = torch.argsort(cost)[: self.cem_elite]
            mu = A[elite_indices].mean(dim=0)
            sigma = A[elite_indices].std(dim=0) + 1e-6  # Avoid division by zero

        return mu[0]

    def compute_uncertainty(self, init_state, iterations=10, device="cpu"):
        """Rolls out policy for T steps, returns avg recon‑MSE."""
        self.policy_model.eval()
        self.dynamics_model.eval()
        self.uncertainty_model.eval()

        s = init_state.to(self.device)
        errs = []
        with torch.no_grad():
            for _ in range(iterations):
                a = self.policy_model(s)
                s = s + self.dynamics_model(s, a)
                recon = self.uncertainty_model(s)
                err = self.uncertainty_model.reconstruction_error(s, recon).item()
                errs.append(err)
        return sum(errs) / len(errs)

    def _add_sample(self, s, a, s_next):
        new_s = torch.tensor(s).to(self.device)  # (1, state_dim)
        new_a = (
            torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device)
        )  # (1, action_dim)
        new_sn = torch.tensor(s_next).unsqueeze(0).to(self.device)  # (1, state_dim)

        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.next_states = self.next_states.to(self.device)

        if len(new_sn.shape) < 2:
            self.next_states = self.next_states.unsqueeze(0)

        self.states = torch.cat([self.states, new_s], dim=0)
        self.actions = torch.cat([self.actions, new_a], dim=0)
        self.next_states = torch.cat([self.next_states, new_sn], dim=0)

    def _retrain(self):
        dyn_dataset = DynamicsDataset(self.states, self.actions, self.next_states)
        dyn_dataloader = torch.utils.data.DataLoader(
            dyn_dataset, batch_size=64, shuffle=True, num_workers=0
        )
        dyn_trainer = DynamicsTrainer(
            model=self.dynamics_model,
            optimizer=torch.optim.Adam(self.dynamics_model.parameters(), lr=1e-3),
            dataloader=dyn_dataloader,
            epochs=self.retrain_epochs,
            device=self.device,
        )
        dyn_trainer.train()

        pol_dataset = PolicyDataset(self.states, self.actions)
        pol_dataloader = torch.utils.data.DataLoader(
            pol_dataset, batch_size=64, shuffle=True, num_workers=0
        )
        pol_trainer = PolicyTrainer(
            model=self.policy_model,
            optimizer=torch.optim.Adam(self.policy_model.parameters(), lr=1e-3),
            dataloader=pol_dataloader,
            epochs=self.retrain_epochs,
            device=self.device,
        )
        pol_trainer.train()

    def expert_action(self, state):
        """
        TODO: Implement expert action logic.
        For now it will just return a random action.
        """
        return self.env.action_space.sample()

    def flatten(self, state: Union[np.ndarray, Dict[str, np.ndarray], Tuple]):
        if isinstance(state, np.ndarray):
            return state.ravel()

        if isinstance(state, dict):
            return np.concatenate(
                [np.asarray(v).ravel() for k, v in sorted(state.items())]
            )

        if isinstance(state, tuple):
            parts = []
            for item in state:
                if isinstance(item, np.ndarray):
                    parts.append(item.ravel())
                elif isinstance(item, dict):
                    parts.extend(
                        [np.asarray(v).ravel() for k, v in sorted(item.items())]
                    )
                else:
                    parts.append(np.asarray(item).ravel())
            return np.concatenate(parts)

        return np.asarray(state).ravel()


if __name__ == "__main__":

    policy_model = PolicyNetwork(
        state_dim=23, action_dim=7, hidden_dim=256, hidden_layers=4
    )
    dynamics_model = DynamicsNetwork(
        state_dim=23, action_dim=7, hidden_dim=256, hidden_layers=4
    )
    autoencoder = DenoisingAutoencoder(
        input_dim=23, hidden_dim=256, hidden_layers=4, noise_std=0.1
    )

    env = gym.make("Pusher-v5", render_mode="human")
    al = ActiveLearning(
        env=env,
        policy_model=policy_model,
        dynamics_model=dynamics_model,
        uncertainty_model=autoencoder,
        actions=torch.randn(1, 2).to("cuda"),
        states=torch.randn(1, 23).to("cuda"),
        next_states=torch.randn(1, 23).to("cuda"),
        state_dim=23,
        actions_dim=7,
        retrain_epochs=1000,
        threshold=2,
        safe_threshold=1,
        max_iterations=5000,
        cem_iterations=1000,
        cem_horizon=100,
        n_samples=1000,
        beta=0.5,
        elite_samples=100,
        device="cuda",
    )
    al.run()
    env.close()
