import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from gymnasium import spaces
import random


class ClaimReserveEnv:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.actions = config["environment"]["actions"]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config["environment"]["state_dim"],),
            dtype=np.float32,
        )

        self.reward_weights = config["environment"]["reward_weights"]
        self.current_episode = None
        self.current_step = 0
        self.max_steps = 60

        self.development_curves = self._initialize_development_curves()

    def reset(
        self,
        episode_data: Optional[pd.DataFrame] = None,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if episode_data is not None:
            self.current_episode = episode_data.reset_index(drop=True)
        else:
            self.current_episode = self._generate_synthetic_episode()

        self.current_step = 0
        return self._get_state(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_episode is None:
            raise ValueError("Environment not reset. Call reset() first.")

        if self.current_step >= len(self.current_episode) - 1:
            return self._get_state(), 0.0, True, False, {}

        reward = self._calculate_reward(action)
        info = self._get_info(action)

        self.current_step += 1
        next_state = self._get_state()

        terminated = self.current_step >= len(self.current_episode) - 1
        truncated = self.current_step >= self.max_steps

        return next_state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        if self.current_episode is None or self.current_step >= len(
            self.current_episode
        ):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        current_row = self.current_episode.iloc[self.current_step]
        state = np.array(
            [
                current_row.get("normalized_paid", 0.0),
                current_row.get("normalized_outstanding", 0.0),
                current_row.get("normalized_total_incurred", 0.0),
                current_row.get("t", 0) / 60.0,
                current_row.get("industry_encoded", 0.0),
                current_row.get("loss_category_encoded", 0.0),
                current_row.get("reserve_gap", 0.0),
                current_row.get("paid_velocity", 0.0),
                current_row.get("outstanding_drift", 0.0),
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_reward(self, action: int) -> float:
        if self.current_step >= len(self.current_episode) - 1:
            return 0.0

        current_row = self.current_episode.iloc[self.current_step]
        next_row = self.current_episode.iloc[self.current_step + 1]

        action_pct = self.actions[action]
        current_reserve = current_row["outstanding"]
        recommended_reserve = current_reserve * (1 + action_pct)

        actual_next_reserve = next_row["outstanding"]
        reserve_error = abs(recommended_reserve - actual_next_reserve) / (
            actual_next_reserve + 1e-6
        )

        under_reserve_penalty = 0.0
        if recommended_reserve < actual_next_reserve * 0.9:
            under_reserve_penalty = (
                self.reward_weights["under_reserve_penalty"] * reserve_error
            )

        volatility_penalty = 0.0
        if self.current_step > 0:
            prev_row = self.current_episode.iloc[self.current_step - 1]
            prev_action_pct = prev_row.get("last_action_pct", 0.0)
            action_change = abs(action_pct - prev_action_pct)
            volatility_penalty = (
                self.reward_weights["volatility_penalty"] * action_change
            )

        accuracy_bonus = 0.0
        if reserve_error < 0.1:
            accuracy_bonus = self.reward_weights["accuracy_bonus"] * (
                0.1 - reserve_error
            )

        stability_bonus = 0.0
        if abs(action_pct) < 0.05:
            stability_bonus = self.reward_weights["stability_bonus"]

        reward = (
            -under_reserve_penalty
            - volatility_penalty
            + accuracy_bonus
            + stability_bonus
        )

        self.current_episode.at[self.current_step, "last_action_pct"] = action_pct

        return reward

    def _get_info(self, action: int) -> Dict[str, Any]:
        current_row = self.current_episode.iloc[self.current_step]
        action_pct = self.actions[action]

        return {
            "claim_id": current_row.get("claim_id", "unknown"),
            "as_at_date": current_row.get("as_at_date", "unknown"),
            "action": action,
            "action_pct": action_pct,
            "current_reserve": current_row["outstanding"],
            "recommended_reserve": current_row["outstanding"] * (1 + action_pct),
            "reserve_gap": current_row.get("reserve_gap", 0.0),
            "step": self.current_step,
        }

    def _initialize_development_curves(self) -> Dict[str, np.ndarray]:
        curves = {}

        industries = ["Motor", "Property", "Liability", "WorkersComp"]
        categories = ["BodilyInjury", "PropertyDamage", "MedicalOnly", "Comprehensive"]

        for industry in industries:
            for category in categories:
                curve = self._generate_development_curve(industry, category)
                curves[f"{industry}_{category}"] = curve

        return curves

    def _generate_development_curve(self, industry: str, category: str) -> np.ndarray:
        base_curve = np.array(
            [
                1.0,
                0.85,
                0.75,
                0.65,
                0.58,
                0.52,
                0.47,
                0.43,
                0.40,
                0.37,
                0.35,
                0.33,
                0.31,
                0.30,
                0.29,
                0.28,
                0.27,
                0.26,
                0.25,
                0.24,
                0.23,
                0.22,
                0.21,
                0.20,
                0.19,
                0.18,
                0.17,
                0.16,
                0.15,
                0.14,
                0.13,
                0.12,
                0.11,
                0.10,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
                0.005,
                0.002,
                0.001,
                0.0005,
                0.0002,
                0.0001,
                0.0,
            ]
        )

        if industry == "Motor":
            base_curve *= 1.2
        elif industry == "Property":
            base_curve *= 0.8
        elif industry == "Liability":
            base_curve *= 1.5

        if category == "BodilyInjury":
            base_curve *= 1.3
        elif category == "PropertyDamage":
            base_curve *= 0.9
        elif category == "MedicalOnly":
            base_curve *= 0.7

        noise = np.random.normal(0, 0.05, len(base_curve))
        curve = base_curve + noise
        curve = np.clip(curve, 0, 1)

        return curve

    def _generate_synthetic_episode(self) -> pd.DataFrame:
        industries = ["Motor", "Property", "Liability", "WorkersComp"]
        categories = ["BodilyInjury", "PropertyDamage", "MedicalOnly", "Comprehensive"]

        industry = random.choice(industries)
        category = random.choice(categories)
        curve_key = f"{industry}_{category}"
        development_curve = self.development_curves[curve_key]

        ultimate_loss = np.random.uniform(1000, 50000)
        episode_length = np.random.randint(12, 48)

        data = []
        cumulative_paid = 0.0

        for t in range(episode_length):
            if t < len(development_curve):
                paid_ratio = 1 - development_curve[t]
                incremental_paid = ultimate_loss * (
                    paid_ratio - cumulative_paid / ultimate_loss
                )
                cumulative_paid += incremental_paid

                outstanding = ultimate_loss * development_curve[t]
                total_incurred = cumulative_paid + outstanding

                paid_velocity = (
                    incremental_paid / (outstanding + 1e-6) if t > 0 else 0.0
                )
                reserve_gap = total_incurred - cumulative_paid - outstanding

                data.append(
                    {
                        "claim_id": f"synthetic_{random.randint(1000, 9999)}",
                        "as_at_date": f"2023-{t + 1:02d}-01",
                        "t": t,
                        "industry": industry,
                        "loss_category": category,
                        "paid": cumulative_paid,
                        "outstanding": outstanding,
                        "total_incurred": total_incurred,
                        "paid_velocity": paid_velocity,
                        "reserve_gap": reserve_gap,
                        "industry_encoded": industries.index(industry),
                        "loss_category_encoded": categories.index(category),
                        "normalized_paid": cumulative_paid / ultimate_loss,
                        "normalized_outstanding": outstanding / ultimate_loss,
                        "normalized_total_incurred": total_incurred / ultimate_loss,
                        "outstanding_drift": 0.0,
                    }
                )

        return pd.DataFrame(data)
