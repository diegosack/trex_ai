import os

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(self._initialize_weights)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, observations):
        action_distribution = Categorical(logits=self.actor(observations))
        state_value = self.critic(observations).squeeze(-1)
        return action_distribution, state_value


class RolloutBuffer:
    def __init__(self, capacity, observation_dim, device):
        self.capacity = capacity
        self.device = device
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probabilities = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def store(self, observation, action, log_probability, value, reward, done):
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.log_probabilities[index] = log_probability
        self.values[index] = value
        self.rewards[index] = reward
        self.dones[index] = float(done)
        self.position += 1

    def clear(self):
        self.position = 0

    def is_full(self):
        return self.position >= self.capacity

    def compute_advantages(self, last_value, gamma, gae_lambda):
        size = self.position
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0
        for index in reversed(range(size)):
            next_value = last_value if index == size - 1 else self.values[index + 1]
            non_terminal = 1.0 - self.dones[index]
            delta = self.rewards[index] + gamma * next_value * non_terminal - self.values[index]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[index] = last_gae
        returns = advantages + self.values[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return (
            torch.as_tensor(self.observations[:size], device=self.device),
            torch.as_tensor(self.actions[:size], device=self.device),
            torch.as_tensor(self.log_probabilities[:size], device=self.device),
            torch.as_tensor(advantages, device=self.device),
            torch.as_tensor(returns, device=self.device),
        )


class PPOAgent:
    def __init__(
        self,
        observation_dim,
        action_dim,
        learning_rate=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.02,
        max_grad_norm=0.5,
        update_epochs=6,
        minibatch_size=512,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.initial_learning_rate = learning_rate
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, observation, deterministic=False):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            distribution, value = self.model(observation_tensor)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(value.item())

    def update(self, buffer, last_value, progress=0.0):
        observations, actions, old_log_probs, advantages, returns = buffer.compute_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        learning_rate = self.initial_learning_rate * (1.0 - progress)
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        update_count = 0
        for _ in range(self.update_epochs):
            permutation = torch.randperm(observations.shape[0], device=self.device)
            for start in range(0, observations.shape[0], self.minibatch_size):
                indices = permutation[start : start + self.minibatch_size]
                distribution, values = self.model(observations[indices])
                log_probabilities = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()
                ratio = (log_probabilities - old_log_probs[indices]).exp()
                surrogate = torch.min(
                    ratio * advantages[indices],
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[indices],
                )
                policy_loss = -surrogate.mean()
                value_loss = ((values - returns[indices]) ** 2).mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                update_count += 1
        return {
            "policy_loss": policy_loss_total / update_count,
            "value_loss": value_loss_total / update_count,
            "entropy": entropy_total / update_count,
        }

    def save(self, path, extra=None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "extra": extra or {},
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device=None, **kwargs):
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        observation_dim = kwargs.pop("observation_dim", checkpoint["observation_dim"])
        action_dim = kwargs.pop("action_dim", checkpoint["action_dim"])
        agent = cls(observation_dim, action_dim, device=device, **kwargs)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        return agent, checkpoint.get("extra", {})
