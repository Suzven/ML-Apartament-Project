from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold


class GradientBoosting:
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

    def fit(self, samples, targets, groups=None, validation=None):
        if groups is None:
            raise ValueError("Engine IDs are required for grouped cross-validation")
        self.model.fit(samples, targets, groups=groups)
        print("\nGradientBoosting | best parameters:", self.model.best_params_)
        print("GradientBoosting | best mean CV RMSE:", -self.model.best_score_)
        print("\nGradientBoosting | CV results:")
        results = self.model.cv_results_
        for index in results["rank_test_score"].argsort():
            print(
                f"parameters={results['params'][index]}, "
                f"mean CV RMSE={-results['mean_test_score'][index]:.6f}"
            )

    def predict(self, samples):
        return self.model.predict(samples)
