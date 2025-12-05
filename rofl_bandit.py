
"""rofl_bandit.py

Updated ROFL contextual bandit utilities.

Adds:
- NeuralBootstrapBandit: simple neural contextual bandit using an ensemble of MLPRegressors for Thompson-like sampling.
- ips_weighted_regression: fit a regression model using IPS sample weights to directly learn policy
- utilities to save/load models (joblib)

Dependencies: numpy, sklearn, joblib

Note: Replace synthetic data / DuckDB placeholders with your production queries.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.special import expit
import joblib
import os

# ----------------------------
# LinUCB (disjoint) - unchanged
# ----------------------------
@dataclass
class LinUCB:
    n_arms: int
    n_features: int
    alpha: float = 1.0  # exploration parameter
    A: Optional[np.ndarray] = None  # (n_arms, d, d)
    b: Optional[np.ndarray] = None  # (n_arms, d)

    def __post_init__(self):
        self.A = np.array([np.eye(self.n_features) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features))

    def select_arm(self, x: np.ndarray) -> int:
        p_vals = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            p = theta.dot(x) + self.alpha * np.sqrt(x.dot(A_inv).dot(x))
            p_vals.append(p)
        return int(np.argmax(p_vals))

    def update(self, chosen_arm: int, x: np.ndarray, reward: float):
        x = x.reshape(-1)
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x

# ----------------------------
# Linear Thompson Sampling (Gaussian linear bandit) - unchanged
# ----------------------------
@dataclass
class LinearThompson:
    n_arms: int
    n_features: int
    v2: float = 1.0  # prior variance scaling
    lambda_reg: float = 1.0

    def __post_init__(self):
        self.V = np.array([self.lambda_reg * np.eye(self.n_features) for _ in range(self.n_arms)])  # precision matrix
        self.Vinv = np.array([np.linalg.inv(self.V[a]) for a in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, self.n_features))

    def select_arm(self, x: np.ndarray) -> int:
        samples = []
        for a in range(self.n_arms):
            Sigma = self.Vinv[a] * self.v2
            mu = Sigma.dot(self.b[a])
            try:
                draw = np.random.multivariate_normal(mu, Sigma)
            except np.linalg.LinAlgError:
                draw = mu
            samples.append(draw.dot(x))
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, x: np.ndarray, reward: float):
        x = x.reshape(-1)
        self.V[chosen_arm] += np.outer(x, x)
        self.Vinv[chosen_arm] = np.linalg.inv(self.V[chosen_arm])
        self.b[chosen_arm] += reward * x

# ----------------------------
# Neural bootstrap ensemble bandit (simple)
# ----------------------------
class NeuralBootstrapBandit:
    """
    Ensemble of MLPRegressors trained on bootstrapped subsets. For selection, sample one model per arm
    (or sample predictions with per-model noise) to obtain Thompson-like exploration.
    """
    def __init__(self, n_arms: int, n_features: int, n_models: int = 8, hidden_layer_sizes=(64,32), random_seed: int = 42):
        self.n_arms = n_arms
        self.n_features = n_features
        self.n_models = n_models
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_seed = random_seed
        # ensemble: list of lists: ensemble[model_idx][arm] = regressor
        self.ensemble = [[MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=300, random_state=random_seed + m + 100*a)
                          for a in range(n_arms)]
                         for m in range(n_models)]
        self.is_fitted = False

    def fit(self, X: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_seed)
        for m in range(self.n_models):
            idxs = rng.choice(n, size=n, replace=True)
            Xm = X[idxs]
            rm = rewards[idxs]
            am = actions[idxs]
            for a in range(self.n_arms):
                # train regressor for arm a using only examples where action==a in the bootstrap sample
                mask = (am == a)
                if mask.sum() < 10:
                    # fallback: train on full bootstrap sample with target=rm where action==a else small-sample smoothing
                    y = np.where(am == a, rm, rm.mean())
                    try:
                        self.ensemble[m][a].fit(Xm, y)
                    except Exception:
                        self.ensemble[m][a] = None
                else:
                    try:
                        self.ensemble[m][a].fit(Xm[mask], rm[mask])
                    except Exception:
                        self.ensemble[m][a] = None
        self.is_fitted = True

    def predict_per_arm(self, x: np.ndarray):
        """Return ensemble predictions shape (n_models, n_arms) for a single x"""
        preds = np.full((self.n_models, self.n_arms), np.nan)
        for m in range(self.n_models):
            for a in range(self.n_arms):
                model = self.ensemble[m][a]
                if model is None:
                    preds[m, a] = 0.0
                else:
                    try:
                        preds[m, a] = model.predict(x.reshape(1, -1))[0]
                    except Exception:
                        preds[m, a] = 0.0
        return preds

    def select_arm(self, x: np.ndarray) -> int:
        # Thompson-like: pick a random model and choose its best arm
        m = np.random.randint(0, self.n_models)
        preds = self.predict_per_arm(x)[m]  # (n_arms,)
        return int(np.nanargmax(preds))

    def predict_expected(self, X: np.ndarray):
        # average ensemble prediction per arm -> shape (n_samples, n_arms)
        n = X.shape[0]
        out = np.zeros((n, self.n_arms))
        for m in range(self.n_models):
            for a in range(self.n_arms):
                model = self.ensemble[m][a]
                if model is None:
                    out[:, a] += 0.0
                else:
                    try:
                        out[:, a] += model.predict(X)
                    except Exception:
                        out[:, a] += 0.0
        out /= float(self.n_models)
        return out

# ----------------------------
# Off-policy evaluators: IPS and Doubly Robust - unchanged
# ----------------------------
class OffPolicyEvaluator:
    @staticmethod
    def ips(rewards, actions, target_actions, propensities, clip=1.0):
        idx = (actions == target_actions)
        w = np.clip(1.0 / propensities, None, clip)
        return float(np.mean(w * rewards * idx))

    @staticmethod
    def dr(rewards, actions, target_actions, propensities, q_hat):
        n = len(rewards)
        dm = float(np.mean(q_hat[np.arange(n), target_actions]))
        correction = float(np.mean((rewards - q_hat[np.arange(n), actions]) * (actions == target_actions) / propensities))
        return dm + correction

# ----------------------------
# IPS-weighted regression trainer
# ----------------------------
def ips_weighted_regression(X, actions, rewards, propensities, n_arms):
    """Fit a single multi-output regression model with IPS sample weights for observed (x,a)->r.
    Returns a list of models, one per arm.
    """
    models = []
    for a in range(n_arms):
        mask = (actions == a)
        if mask.sum() < 10:
            model = None
            models.append(model)
            continue
        X_a = X[mask]
        y_a = rewards[mask]
        w_a = 1.0 / propensities[mask]
        reg = GradientBoostingRegressor(n_estimators=100)
        reg.fit(X_a, y_a, sample_weight=w_a)
        models.append(reg)
    return models

def predict_from_models(models, X):
    n = X.shape[0]
    n_arms = len(models)
    out = np.zeros((n, n_arms))
    for a, m in enumerate(models):
        if m is None:
            out[:, a] = 0.0
        else:
            out[:, a] = m.predict(X)
    return out

# ----------------------------
# Synthetic dataset generator (unchanged)
# ----------------------------
def generate_synthetic_data(n: int = 10000, n_arms: int = 5, d: int = 8, seed: Optional[int]=123):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    true_thetas = rng.normal(scale=0.75, size=(n_arms, d))
    logits = X @ true_thetas.T
    probs = expit(logits)
    logging_theta = rng.normal(scale=0.5, size=(n_arms, d))
    logging_logits = X @ logging_theta.T
    logging_scores = expit(logging_logits)
    exp_scores = np.exp(logging_scores)
    logging_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    logged_actions = np.array([rng.choice(n_arms, p=logging_probs[i]) for i in range(n)])
    rewards = np.array([rng.binomial(1, probs[i, logged_actions[i]]) for i in range(n)])
    propensities = logging_probs[np.arange(n), logged_actions]
    return {
        'X': X,
        'rewards': rewards,
        'actions': logged_actions,
        'propensities': propensities,
        'true_probs': probs,
        'logging_probs': logging_probs,
        'true_thetas': true_thetas
    }

# ----------------------------
# Reward model trainer for DR (unchanged)
# ----------------------------
def train_reward_model(X, actions, rewards, n_arms, alpha=1.0):
    n, d = X.shape
    q_hat = np.zeros((n, n_arms))
    for a in range(n_arms):
        idx = (actions == a)
        if idx.sum() < 10:
            q_hat[:, a] = rewards.mean()
            continue
        clf = LogisticRegression(max_iter=200)
        clf.fit(X[idx], rewards[idx])
        q_hat[:, a] = clf.predict_proba(X)[:, 1]
    return q_hat

# ----------------------------
# Save/Load helpers
# ----------------------------
def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
