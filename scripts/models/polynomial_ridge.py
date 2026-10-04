from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class PolynomialRidge:
    uses_internal_scaling = True

    def __init__(self):
        pipeline = Pipeline([
            ("input_scaling", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("feature_scaling", StandardScaler()),
            ("ridge", Ridge()),
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

    def fit(self, samples, targets, groups=None):
        if groups is None:
            raise ValueError("Engine IDs are required for grouped cross-validation")
        self.model.fit(samples, targets, groups=groups)
        print("\nPolynomialRidge | best alpha:", self.model.best_params_["ridge__alpha"])
        print("PolynomialRidge | best mean CV RMSE:", -self.model.best_score_)

    def predict(self, samples):
        return self.model.predict(samples)
