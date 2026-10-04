from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .base import GridSearchRegression


class PolynomialLasso(GridSearchRegression):
    def __init__(self):
        pipeline = Pipeline([
            ("input_scaling", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=3, include_bias=False)),
            ("feature_scaling", StandardScaler()),
            ("lasso", Lasso(max_iter=50000, tol=0.0001, precompute=True)),
        ])
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid={"lasso__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
