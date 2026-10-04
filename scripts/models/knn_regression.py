from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import GridSearchRegression


class KNNRegression(GridSearchRegression):
    def __init__(self):
        pipeline = Pipeline([
            ("scaling", StandardScaler()),
            ("regression", KNeighborsRegressor(p=2)),
        ])
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid={
                "regression__n_neighbors": [3, 5, 10, 20, 50, 100],
                "regression__weights": ["uniform", "distance"],
            },
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
