class RegressionModel:
    requires_history = False

    def fit(self, samples, targets, groups=None, validation=None):
        self.model.fit(samples, targets)

    def predict(self, samples):
        return self.model.predict(samples)


class GridSearchRegression(RegressionModel):
    def fit(self, samples, targets, groups=None, validation=None):
        if groups is None:
            raise ValueError("Engine IDs are required for grouped cross-validation")

        self.model.fit(samples, targets, groups=groups)
        self.print_search_results()

    def print_search_results(self):
        model_name = type(self).__name__
        print(f"\n{model_name} | best parameters:", self.model.best_params_)
        print(f"{model_name} | best mean CV RMSE:", -self.model.best_score_)
        print(f"\n{model_name} | CV results:")

        results = self.model.cv_results_
        ranked_indices = results["rank_test_score"].argsort()
        for index in ranked_indices:
            parameters = results["params"][index]
            rmse = -results["mean_test_score"][index]
            print(f"parameters={parameters}, mean CV RMSE={rmse:.6f}")
