from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class KNNRegression:
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

    def fit(self, samples, targets, groups=None, validation=None):
        if groups is None:
            raise ValueError("Engine IDs are required for grouped cross-validation")
        self.model.fit(samples, targets, groups=groups)
        print("\nKNNRegression | best n_neighbors:", self.model.best_params_["regression__n_neighbors"])
        print("KNNRegression | best weights:", self.model.best_params_["regression__weights"])
        print("KNNRegression | best mean CV RMSE:", -self.model.best_score_)
        print("\nKNNRegression | CV results:")
        results = self.model.cv_results_
        for index in results["rank_test_score"].argsort():
            parameters = results["params"][index]
            rmse = -results["mean_test_score"][index]
            print(
                f"n_neighbors={parameters['regression__n_neighbors']}, "
                f"weights={parameters['regression__weights']}, mean CV RMSE={rmse:.6f}"
            )

    def predict(self, samples):
        return self.model.predict(samples)
