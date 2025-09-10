from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Tuple, Optional

class PCATransformer:
    def __init__(self, n_components: int = 500, whiten: bool = False):
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=n_components, whiten=whiten, random_state=0)

    def fit(self, X: pd.DataFrame):
        X_imp = self.imputer.fit_transform(X)
        X_std = self.scaler.fit_transform(X_imp)
        self.pca.fit(X_std)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_std = self.scaler.transform(X_imp)
        return self.pca.transform(X_std)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imp = self.imputer.fit_transform(X)
        X_std = self.scaler.fit_transform(X_imp)
        return self.pca.fit_transform(X_std)

def tsne_2d(X_pca: np.ndarray, random_state: int = 0, perplexity: float = 30.0, learning_rate: float = 200.0) -> np.ndarray:
    return TSNE(n_components=2, random_state=random_state, perplexity=perplexity, learning_rate=learning_rate).fit_transform(X_pca)
