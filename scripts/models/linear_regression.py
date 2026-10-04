from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import RegressionModel


class LinearRegression(RegressionModel):
    def __init__(self):
        self.model = Pipeline([
            ("scaling", StandardScaler()),
            ("regression", SklearnLinearRegression(fit_intercept=True)),
        ])
