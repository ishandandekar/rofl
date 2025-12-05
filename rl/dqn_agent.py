import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from .networks import DQNNetwork, ReplayBuffer
from .env import ClaimReserveEnv


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQNNetwork(state_dim, action_dim, config).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config["training"]["learning_rate"]
        )

        self.replay_buffer = ReplayBuffer(config["training"]["buffer_size"])
        self.gamma = config["training"]["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.update_target_every = 100
        self.update_count = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        reward_tensor = torch.tensor([reward]).to(self.device)
        done_tensor = torch.tensor([done]).to(self.device)

        self.replay_buffer.push(
            state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor
        )

    def replay(self, batch_size: int = 64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        actions_long = actions.long()
        if actions_long.dim() == 1:
            actions_long = actions_long.unsqueeze(1)
        current_q_values = self.q_network(states).gather(1, actions_long)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Ensure both tensors have the same shape
        current_q_values = current_q_values.squeeze()

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_count += 1
        if self.update_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train_episode(
        self, env: ClaimReserveEnv, max_steps: int = 60
    ) -> Dict[str, float]:
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = self.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        self.replay()

        return {"total_reward": total_reward, "steps": steps, "epsilon": self.epsilon}

    def evaluate(
        self, env: ClaimReserveEnv, num_episodes: int = 10
    ) -> Dict[str, float]:
        total_rewards = []
        total_steps = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0

            while True:
                action = self.act(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                state = next_state
                episode_reward += reward
                steps += 1

                if done or steps >= 60:
                    break

            total_rewards.append(episode_reward)
            total_steps.append(steps)

        return {
            "avg_reward": float(np.mean(total_rewards)),
            "avg_steps": float(np.mean(total_steps)),
            "std_reward": float(np.std(total_rewards)),
        }

    def save(self, path: str):
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", 0.01)
