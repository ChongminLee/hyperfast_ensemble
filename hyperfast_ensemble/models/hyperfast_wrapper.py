from __future__ import annotations
import numpy as np

class HyperfastWrapper:
    """Wrapper around HyperFastClassifier with a safe fallback.
    - If `from hyperfast import HyperFastClassifier` works, we use it.
    - Otherwise we fall back to a prototype (nearest-centroid-like) classifier in the transformed space.
    """
    def __init__(self, use_fallback: bool = False, n_jobs: int = -1):
        self.use_fallback = use_fallback
        self.model = None
        self._centroids = None
        self._labels = None
        self.n_jobs = n_jobs

        if not self.use_fallback:
            try:
                from hyperfast import HyperFastClassifier
                self.model = HyperFastClassifier()
            except Exception:
                self.use_fallback = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.use_fallback:
            # Prototype classifier: class centroids in feature space
            classes = np.unique(y)
            centroids = []
            for c in classes:
                centroids.append(X[y == c].mean(axis=0))
            self._centroids = np.vstack(centroids)  # (C, d)
            self._labels = classes
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_fallback:
            # nearest centroid
            # distances: (n, C)
            d2 = ((X[:, None, :] - self._centroids[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2, axis=1)
            return self._labels[idx]
        else:
            return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.use_fallback:
            # Softmax over negative distance as pseudo-probabilities
            d2 = ((X[:, None, :] - self._centroids[None, :, :]) ** 2).sum(axis=2)
            logits = -d2
            # softmax
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            return probs
        else:
            return self.model.predict_proba(X)
