import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super(DQNNetwork, self).__init__()

        hidden_layers = config["models"]["dqn"]["hidden_layers"]
        activation = config["models"]["dqn"]["activation"]
        dropout = config["models"]["dqn"]["dropout"]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PPONetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super(PPONetwork, self).__init__()

        hidden_layers = config["models"]["ppo"]["hidden_layers"]
        activation = config["models"]["ppo"]["activation"]
        dropout = config["models"]["ppo"]["dropout"]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim), nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return policy, value

    def get_action(self, state):
        policy, value = self.forward(state)
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, value


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        import random

        batch = random.sample(self.buffer, batch_size)

        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch])
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.stack([item[3] for item in batch])
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int = 64):
        super(FeatureExtractor, self).__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.feature_net(x)
