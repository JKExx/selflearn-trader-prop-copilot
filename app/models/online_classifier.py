from __future__ import annotations
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def _to_numpy(X):
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=float)

class OnlineClassifier:
    """
    Binary online classifier predicting P(next_bar_up).
    - Works with pandas or numpy
    - Online partial_fit
    - Persists scaler/model + feature count
    - Auto-resets if feature count changes between runs
    """
    def __init__(self, model_path: str = "models_ckpt/online_clf.joblib",
                 random_state: int = 42, alpha: float = 1e-4):
        self.model_path = model_path
        self._random_state = random_state
        self._alpha = alpha
        self._reset()

    def _reset(self):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.clf = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=self._alpha,
            learning_rate="optimal", random_state=self._random_state,
        )
        self._initialized = False
        self._classes = np.array([0, 1], dtype=int)
        self._n_features = None

    def _ensure_feature_compat(self, X):
        n = X.shape[1]
        if self._initialized and self._n_features is not None and n != self._n_features:
            # stale checkpoint: soft reset
            self._reset()

    def partial_fit(self, X, y):
        X = _to_numpy(X)
        y = _to_numpy(y).ravel().astype(int)
        self._ensure_feature_compat(X)

        if not self._initialized:
            self.scaler.partial_fit(X)
            Xs = self.scaler.transform(X)
            self.clf.partial_fit(Xs, y, classes=self._classes)
            self._initialized = True
            self._n_features = X.shape[1]
        else:
            self.scaler.partial_fit(X)
            Xs = self.scaler.transform(X)
            self.clf.partial_fit(Xs, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = _to_numpy(X)
        if X.ndim == 1:
            X = X[None, :]
        if (not self._initialized) or (self._n_features is not None and X.shape[1] != self._n_features):
            # cold start or width mismatch: return neutral probs
            return np.tile([0.5, 0.5], (X.shape[0], 1))
        Xs = self.scaler.transform(X)
        try:
            return self.clf.predict_proba(Xs)
        except Exception:
            z = self.clf.decision_function(Xs)
            p1 = 1.0 / (1.0 + np.exp(-z))
            p1 = p1.reshape(-1, 1)
            return np.hstack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def decide_action(p_up, threshold: float) -> int:
        p = float(np.asarray(p_up).ravel()[0])
        if p >= threshold:
            return 1
        if p <= (1.0 - threshold):
            return -1
        return 0

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "clf": self.clf,
            "initialized": self._initialized,
            "classes": self._classes,
            "n_features": self._n_features,
            "alpha": self._alpha,
            "random_state": self._random_state,
        }, self.model_path)

    def load_if_exists(self) -> bool:
        if os.path.exists(self.model_path):
            obj = joblib.load(self.model_path)
            self.scaler = obj["scaler"]
            self.clf = obj["clf"]
            self._initialized = bool(obj.get("initialized", True))
            self._classes = obj.get("classes", np.array([0, 1], dtype=int))
            self._n_features = obj.get("n_features", None)
            self._alpha = obj.get("alpha", self._alpha)
            self._random_state = obj.get("random_state", self._random_state)
            return True
        return False
