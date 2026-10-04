from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeRegressor


class DecisionTree:
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

    def fit(self, samples, targets, groups=None, validation=None):
        if groups is None:
            raise ValueError("Engine IDs are required for grouped cross-validation")
        self.model.fit(samples, targets, groups=groups)
        print("\nDecisionTree | best parameters:", self.model.best_params_)
        print("DecisionTree | best mean CV RMSE:", -self.model.best_score_)
        print("\nDecisionTree | CV results:")
        results = self.model.cv_results_
        for index in results["rank_test_score"].argsort():
            print(
                f"parameters={results['params'][index]}, "
                f"mean CV RMSE={-results['mean_test_score'][index]:.6f}"
            )

    def predict(self, samples):
        return self.model.predict(samples)
