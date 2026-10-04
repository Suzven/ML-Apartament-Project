from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeRegressor

from .base import GridSearchRegression


class DecisionTree(GridSearchRegression):
    def __init__(self):
        self.model = GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=42),
            param_grid={
                "max_depth": [3, 5, 10, None],
                "min_samples_leaf": [1, 10, 50],
                "min_samples_split": [2, 20, 100],
            },
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
