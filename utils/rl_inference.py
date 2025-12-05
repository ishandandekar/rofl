import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pickle
from datetime import datetime

from utils.db import ROFLDatabase
from utils.transforms import FeatureEngineer
from rl.dqn_agent import DQNAgent
from rl.ppo_agent import PPOAgent
from rl.env import ClaimReserveEnv


class ROFLInference:
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
        self.current_model = None
        self.current_model_type = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_models(self, model_type: str = "dqn"):
        feature_path = self.models_dir / "feature_engineer.pkl"
        if feature_path.exists():
            self.feature_engineer.load(str(feature_path))

        if model_type == "dqn":
            model_path = self.models_dir / "dqn_model.pt"
            if model_path.exists():
                self.dqn_agent.load(str(model_path))
                self.current_model = self.dqn_agent
                self.current_model_type = "dqn"
        elif model_type == "ppo":
            model_path = self.models_dir / "ppo_model.pt"
            if model_path.exists():
                self.ppo_agent.load(str(model_path))
                self.current_model = self.ppo_agent
                self.current_model_type = "ppo"

        if self.current_model is None:
            raise ValueError(f"No trained model found for type: {model_type}")

    def predict_single(self, claim_snapshot: pd.Series) -> Dict[str, Any]:
        if not self.feature_engineer.is_fitted:
            raise ValueError("Feature engineer not fitted. Load models first.")

        state = self.feature_engineer.get_state_vector(claim_snapshot)

        if self.current_model_type == "dqn":
            action_idx = self.current_model.act(state, training=False)
            confidence = self._calculate_dqn_confidence(state, action_idx)
        else:
            with torch.no_grad():
                action_idx, _, _ = self.current_model.act(state)
            confidence = self._calculate_ppo_confidence(state, action_idx)

        action_pct = self.config["environment"]["actions"][action_idx]
        current_reserve = claim_snapshot["outstanding"]
        recommended_reserve = current_reserve * (1 + action_pct)

        result = {
            "claim_id": claim_snapshot["claim_id"],
            "as_at_date": claim_snapshot["as_at_date"],
            "recommended_action_idx": action_idx,
            "recommended_pct": action_pct,
            "recommended_new_reserve": recommended_reserve,
            "policy_confidence": confidence,
            "model_type": self.current_model_type,
            "model_path": str(self.models_dir / f"{self.current_model_type}_model.pt"),
            "inference_ts": datetime.now(),
        }

        return result

    def predict_batch(self, claim_snapshots: pd.DataFrame) -> pd.DataFrame:
        results = []

        for _, row in claim_snapshots.iterrows():
            result = self.predict_single(row)
            results.append(result)

        results_df = pd.DataFrame(results)
        self.db.save_inference(results_df)

        return results_df

    def predict_portfolio(self, as_at_date: str) -> pd.DataFrame:
        snapshot_df = self.db.get_claims_snapshot(as_at_date)

        if len(snapshot_df) == 0:
            raise ValueError(f"No claim snapshots found for date: {as_at_date}")

        return self.predict_batch(snapshot_df)

    def _calculate_dqn_confidence(self, state: np.ndarray, action_idx: int) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.current_model.q_network(state_tensor)

        q_values_np = q_values.cpu().numpy().flatten()
        max_q = np.max(q_values_np)
        second_max_q = np.partition(q_values_np, -2)[-2] if len(q_values_np) > 1 else 0

        confidence = (max_q - second_max_q) / (max_q + 1e-6)
        return min(max(confidence, 0.0), 1.0)

    def _calculate_ppo_confidence(self, state: np.ndarray, action_idx: int) -> float:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy, _ = self.current_model.network(state_tensor)

        policy_probs = policy.cpu().numpy().flatten()
        action_prob = policy_probs[action_idx]

        entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-6))
        confidence = action_prob * (1 - entropy / np.log(len(policy_probs)))

        return min(max(confidence, 0.0), 1.0)

    def generate_explanations(
        self, claim_snapshot: pd.Series, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.feature_engineer.is_fitted:
            raise ValueError("Feature engineer not fitted. Load models first.")

        state = self.feature_engineer.get_state_vector(claim_snapshot)

        feature_names = [
            "normalized_paid",
            "normalized_outstanding",
            "normalized_total_incurred",
            "t",
            "industry_encoded",
            "loss_category_encoded",
            "reserve_gap",
            "paid_velocity",
        ]

        explanations = []

        for i, feature_name in enumerate(feature_names):
            perturbed_state = state.copy()
            perturbed_state[i] += 0.1

            if self.current_model_type == "dqn":
                original_action = self.current_model.act(state, training=False)
                perturbed_action = self.current_model.act(
                    perturbed_state, training=False
                )
            else:
                with torch.no_grad():
                    original_action, _, _ = self.current_model.act(state)
                    perturbed_action, _, _ = self.current_model.act(perturbed_state)

            action_change = abs(original_action - perturbed_action)

            explanations.append(
                {
                    "claim_id": claim_snapshot["claim_id"],
                    "as_at_date": claim_snapshot["as_at_date"],
                    "feature_name": feature_name,
                    "feature_value": state[i],
                    "shap_value": action_change,
                    "explanation_rank": 0,
                }
            )

        explanations.sort(key=lambda x: x["shap_value"], reverse=True)

        for i, exp in enumerate(explanations[:top_k]):
            exp["explanation_rank"] = i + 1

        return explanations[:top_k]

    def get_inference_history(
        self, claim_id: str = None, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM rofl_inference WHERE 1=1"

        if claim_id:
            query += f" AND claim_id = '{claim_id}'"
        if start_date:
            query += f" AND as_at_date >= '{start_date}'"
        if end_date:
            query += f" AND as_at_date <= '{end_date}'"

        query += " ORDER BY inference_ts DESC"

        return self.db.conn.execute(query).df()

    def get_model_performance_metrics(self) -> Dict[str, Any]:
        history_df = self.get_inference_history()

        if len(history_df) == 0:
            return {"error": "No inference history found"}

        metrics = {
            "total_inferences": len(history_df),
            "avg_confidence": history_df["policy_confidence"].mean(),
            "confidence_std": history_df["policy_confidence"].std(),
            "model_type": history_df["model_type"].value_counts().to_dict(),
            "action_distribution": history_df["recommended_action_idx"]
            .value_counts()
            .to_dict(),
            "date_range": {
                "start": history_df["as_at_date"].min(),
                "end": history_df["as_at_date"].max(),
            },
        }

        return metrics
