from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearRegression:
    def __init__(self):
        self.model = Pipeline([
            ("scaling", StandardScaler()),
            ("regression", SklearnLinearRegression(fit_intercept=True)),
        ])

    def fit(self, samples, targets, groups=None, validation=None):
        self.model.fit(samples, targets)

    def predict(self, samples):
        return self.model.predict(samples)
