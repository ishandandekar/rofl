import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from .networks import PPONetwork
from .env import ClaimReserveEnv


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = PPONetwork(state_dim, action_dim, config).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config["training"]["learning_rate"]
        )

        self.gamma = config["training"]["gamma"]
        self.clip_ratio = config["models"]["ppo"]["clip_ratio"]
        self.entropy_coef = config["models"]["ppo"]["entropy_coef"]

        self.memory = []

    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor)

        return action.item(), log_prob, value

    def remember(self, state, action, log_prob, value, reward, done):
        self.memory.append(
            {
                "state": torch.FloatTensor(state).to(self.device),
                "action": torch.tensor(action).to(self.device),
                "log_prob": log_prob,
                "value": value,
                "reward": torch.tensor(reward).to(self.device),
                "done": torch.tensor(done).to(self.device),
            }
        )

    def compute_advantages(
        self, rewards: List[float], values: List[torch.Tensor], dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        returns = []

        G = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1].item()

            G = rewards[i] + self.gamma * next_value * (1 - dones[i])
            returns.insert(0, G)

            advantage = G - values[i].item()
            advantages.insert(0, advantage)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, epochs: int = 4, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return

        states = torch.stack([m["state"] for m in self.memory])
        actions = torch.stack([m["action"] for m in self.memory])
        old_log_probs = torch.stack([m["log_prob"].squeeze() for m in self.memory])
        values = torch.stack([m["value"].squeeze() for m in self.memory])
        rewards = [m["reward"].item() for m in self.memory]
        dones = [m["done"].item() for m in self.memory]

        returns, advantages = self.compute_advantages(rewards, values, dones)

        for epoch in range(epochs):
            for _ in range(0, len(states), batch_size):
                batch_indices = np.random.choice(len(states), batch_size)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                policy, value = self.network(batch_states)
                action_dist = torch.distributions.Categorical(policy)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    * batch_advantages
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                entropy_loss = -self.entropy_coef * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()

    def train_episode(
        self, env: ClaimReserveEnv, max_steps: int = 60
    ) -> Dict[str, float]:
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action, log_prob, value = self.act(state)
            next_state, reward, done, info = env.step(action)

            self.remember(state, action, log_prob, value, reward, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        self.update()

        return {"total_reward": total_reward, "steps": steps}

    def evaluate(
        self, env: ClaimReserveEnv, num_episodes: int = 10
    ) -> Dict[str, float]:
        total_rewards = []
        total_steps = []

        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0

            while True:
                with torch.no_grad():
                    action, _, _ = self.act(state)

                next_state, reward, done, info = env.step(action)

                state = next_state
                episode_reward += reward
                steps += 1

                if done or steps >= 60:
                    break

            total_rewards.append(episode_reward)
            total_steps.append(steps)

        return {
            "avg_reward": np.mean(total_rewards),
            "avg_steps": np.mean(total_steps),
            "std_reward": np.std(total_rewards),
        }

    def save(self, path: str):
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
