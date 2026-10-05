from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import GridSearchRegression
from .polynomial_features import polynomial_and_history


class GradientBoosting(GridSearchRegression):
    def __init__(self):
        pipeline = Pipeline([
            ("input_scaling", StandardScaler().set_output(transform="pandas")),
            ("polynomial", polynomial_and_history()),
            ("boosting", HistGradientBoostingRegressor(
                max_iter=200,
                min_samples_leaf=20,
                early_stopping=False,
                random_state=42,
            )),
        ])
        self.model = GridSearchCV(
            estimator=pipeline,
            param_grid={
                "boosting__learning_rate": [0.05, 0.1],
                "boosting__max_leaf_nodes": [15, 31],
                "boosting__l2_regularization": [0, 10],
            },
            scoring="neg_root_mean_squared_error",
            cv=GroupKFold(n_splits=5),
            refit=True,
            n_jobs=1,
            verbose=2,
            error_score="raise",
        )
