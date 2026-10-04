from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .polynomial_ridge import PolynomialRidge


class HistoryPolynomialRidge(PolynomialRidge):
    requires_history = True

    def __init__(self):
        super().__init__()
        original_features = Pipeline([
            ("input_scaling", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=3, include_bias=False)),
            ("feature_scaling", StandardScaler()),
        ])
        feature_blocks = ColumnTransformer([
            ("original", original_features, make_column_selector(pattern=r"^\d+$")),
            ("history", StandardScaler(), make_column_selector(pattern=r"^sensor_")),
        ])
        pipeline = Pipeline([
            ("features", feature_blocks),
            ("ridge", Ridge()),
        ])
        self.model.set_params(estimator=pipeline)
