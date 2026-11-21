import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MassTakeRegressor(BaseEstimator):
    """Learn log mass as a target; expose a monotone score around the 500 g rule."""

    def __init__(self, alpha=8.0, temperature=0.45):
        self.alpha = alpha
        self.temperature = temperature

    def fit(self, x, mass_g, sample_weight=None):
        self.model_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
        self.model_.fit(x, np.log(np.asarray(mass_g)), ridge__sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        logits = np.clip((np.log(500.0) - self.model_.predict(x)) / self.temperature, -40, 40)
        take_score = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack((1 - take_score, take_score))
