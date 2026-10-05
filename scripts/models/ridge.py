from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import GridSearchRegression


class Ridge(GridSearchRegression):
    def __init__(self):
        pipeline = Pipeline([
            ("scaling", StandardScaler()),
            ("ridge", SklearnRidge()),
        ])
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid={"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
