from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import GridSearchRegression
from .polynomial_features import polynomial_and_history


class PolynomialLasso(GridSearchRegression):
    def __init__(self):
        pipeline = Pipeline([
            ("input_scaling", StandardScaler().set_output(transform="pandas")),
            ("polynomial", polynomial_and_history()),
            ("feature_scaling", StandardScaler()),
            ("lasso", Lasso(max_iter=30000, tol=0.001, selection="random", random_state=42)),
        ])
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid={"lasso__alpha": [0.1, 0.3, 1, 3, 10, 30]},
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            verbose=2,
            error_score="raise",
        )
