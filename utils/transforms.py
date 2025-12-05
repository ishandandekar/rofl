import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.industry_encoder = LabelEncoder()
        self.loss_category_encoder = LabelEncoder()
        self.feature_columns = [
            "normalized_paid",
            "normalized_outstanding",
            "normalized_total_incurred",
            "t",
            "industry_encoded",
            "loss_category_encoded",
            "reserve_gap",
            "paid_velocity",
        ]
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        df = df.copy()

        df["paid_velocity"] = self._calculate_paid_velocity(df)
        df["reserve_gap"] = self._calculate_reserve_gap(df)

        self.industry_encoder.fit(df["industry"].fillna("Unknown"))
        self.loss_category_encoder.fit(df["loss_category"].fillna("Unknown"))

        feature_data = self._prepare_features(df)
        self.scaler.fit(feature_data)
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        df = df.copy()
        df["paid_velocity"] = self._calculate_paid_velocity(df)
        df["reserve_gap"] = self._calculate_reserve_gap(df)

        feature_data = self._prepare_features(df)
        return self.scaler.transform(feature_data)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def _calculate_paid_velocity(self, df: pd.DataFrame) -> pd.Series:
        df = df.sort_values(["claim_id", "as_at_date"])
        df["paid_prev"] = df.groupby("claim_id")["paid"].shift(1)
        df["outstanding_prev"] = df.groupby("claim_id")["outstanding"].shift(1)

        velocity = np.where(
            df["outstanding_prev"] > 0,
            (df["paid"] - df["paid_prev"]) / df["outstanding_prev"],
            0.0,
        )
        return pd.Series(velocity, index=df.index)

    def _calculate_reserve_gap(self, df: pd.DataFrame) -> pd.Series:
        return df["total_incurred"] - df["paid"] - df["outstanding"]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["industry_encoded"] = self.industry_encoder.transform(
            df["industry"].fillna("Unknown")
        )
        df["loss_category_encoded"] = self.loss_category_encoder.transform(
            df["loss_category"].fillna("Unknown")
        )

        features = pd.DataFrame()
        features["normalized_paid"] = df["paid"]
        features["normalized_outstanding"] = df["outstanding"]
        features["normalized_total_incurred"] = df["total_incurred"]
        features["t"] = df["t"]
        features["industry_encoded"] = df["industry_encoded"]
        features["loss_category_encoded"] = df["loss_category_encoded"]
        features["reserve_gap"] = df["reserve_gap"]
        features["paid_velocity"] = df["paid_velocity"].fillna(0)

        return features[self.feature_columns]

    def create_episodes(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        episodes = []
        for claim_id in df["claim_id"].unique():
            claim_data = df[df["claim_id"] == claim_id].sort_values("as_at_date")
            if len(claim_data) > 1:
                episodes.append(claim_data)
        return episodes

    def get_state_vector(self, claim_snapshot: pd.Series) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first")

        df = pd.DataFrame([claim_snapshot])
        features = self.transform(df)
        return features[0]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "industry_encoder": self.industry_encoder,
                    "loss_category_encoder": self.loss_category_encoder,
                    "feature_columns": self.feature_columns,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.industry_encoder = data["industry_encoder"]
            self.loss_category_encoder = data["loss_category_encoder"]
            self.feature_columns = data["feature_columns"]
            self.is_fitted = data["is_fitted"]
