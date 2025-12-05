import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging

from utils.db import ROFLDatabase
from utils.transforms import FeatureEngineer
from rl.env import ClaimReserveEnv
from rl.dqn_agent import DQNAgent
from rl.ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROFLTrainer:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.db = ROFLDatabase(config_path)
        self.feature_engineer = FeatureEngineer()
        self.env = ClaimReserveEnv(self.config)

        self.state_dim = self.config["environment"]["state_dim"]
        self.action_dim = len(self.config["environment"]["actions"])

        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim, self.config)
        self.ppo_agent = PPOAgent(self.state_dim, self.action_dim, self.config)

        self.models_dir = Path("outputs/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = Path("outputs/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def prepare_data(self, csv_path: str = None) -> List[pd.DataFrame]:
        if csv_path:
            logger.info(f"Loading data from {csv_path}")
            self.db.load_claims_data(csv_path)

        logger.info("Extracting training data from database")
        df = self.db.get_training_data()

        if len(df) == 0:
            logger.info("No data found, generating synthetic episodes")
            episodes = []
            for i in range(100):
                episode = self.env._generate_synthetic_episode()
                episodes.append(episode)
            return episodes

        logger.info(f"Processing {len(df)} claim snapshots")
        self.feature_engineer.fit(df)

        episodes = self.feature_engineer.create_episodes(df)
        logger.info(f"Created {len(episodes)} training episodes")

        return episodes

    def train_dqn(
        self, episodes: List[pd.DataFrame], num_episodes: int = 1000
    ) -> Dict[str, Any]:
        logger.info("Starting DQN training")

        training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilon_values": [],
        }

        for episode in range(num_episodes):
            episode_data = np.random.choice(episodes)
            metrics = self.dqn_agent.train_episode(self.env)

            training_metrics["episode_rewards"].append(metrics["total_reward"])
            training_metrics["episode_lengths"].append(metrics["steps"])
            training_metrics["epsilon_values"].append(metrics["epsilon"])

            if episode % 100 == 0:
                avg_reward = np.mean(training_metrics["episode_rewards"][-100:])
                logger.info(
                    f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}"
                )

                eval_metrics = self.dqn_agent.evaluate(self.env, num_episodes=5)
                logger.info(
                    f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}"
                )

        model_path = self.models_dir / "dqn_model.pt"
        self.dqn_agent.save(str(model_path))
        logger.info(f"DQN model saved to {model_path}")

        return training_metrics

    def train_ppo(
        self, episodes: List[pd.DataFrame], num_episodes: int = 1000
    ) -> Dict[str, Any]:
        logger.info("Starting PPO training")

        training_metrics = {"episode_rewards": [], "episode_lengths": []}

        for episode in range(num_episodes):
            episode_data = np.random.choice(episodes)
            metrics = self.ppo_agent.train_episode(self.env)

            training_metrics["episode_rewards"].append(metrics["total_reward"])
            training_metrics["episode_lengths"].append(metrics["steps"])

            if episode % 100 == 0:
                avg_reward = np.mean(training_metrics["episode_rewards"][-100:])
                logger.info(
                    f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}"
                )

                eval_metrics = self.ppo_agent.evaluate(self.env, num_episodes=5)
                logger.info(
                    f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}"
                )

        model_path = self.models_dir / "ppo_model.pt"
        self.ppo_agent.save(str(model_path))
        logger.info(f"PPO model saved to {model_path}")

        return training_metrics

    def train_both(self, csv_path: str = None) -> Dict[str, Any]:
        episodes = self.prepare_data(csv_path)

        logger.info("Training DQN agent")
        dqn_metrics = self.train_dqn(episodes)

        logger.info("Training PPO agent")
        ppo_metrics = self.train_ppo(episodes)

        feature_path = self.models_dir / "feature_engineer.pkl"
        self.feature_engineer.save(str(feature_path))
        logger.info(f"Feature engineer saved to {feature_path}")

        return {
            "dqn_metrics": dqn_metrics,
            "ppo_metrics": ppo_metrics,
            "num_episodes": len(episodes),
        }

    def compare_models(self, num_eval_episodes: int = 20) -> Dict[str, Any]:
        logger.info("Comparing DQN vs PPO models")

        dqn_results = self.dqn_agent.evaluate(self.env, num_eval_episodes)
        ppo_results = self.ppo_agent.evaluate(self.env, num_eval_episodes)

        comparison = {
            "dqn": dqn_results,
            "ppo": ppo_results,
            "winner": "dqn"
            if dqn_results["avg_reward"] > ppo_results["avg_reward"]
            else "ppo",
        }

        logger.info(f"Comparison complete. Winner: {comparison['winner']}")
        logger.info(f"DQN - Avg Reward: {dqn_results['avg_reward']:.2f}")
        logger.info(f"PPO - Avg Reward: {ppo_results['avg_reward']:.2f}")

        return comparison


if __name__ == "__main__":
    trainer = ROFLTrainer()

    try:
        metrics = trainer.train_both()
        logger.info("Training completed successfully")

        comparison = trainer.compare_models()
        logger.info("Model comparison completed")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
