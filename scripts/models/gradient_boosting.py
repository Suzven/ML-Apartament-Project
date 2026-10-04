from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold

from .base import GridSearchRegression


class GradientBoosting(GridSearchRegression):
    def __init__(self):
        self.model = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42, min_samples_leaf=10),
            param_grid={
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            error_score="raise",
            verbose=1,
        )
