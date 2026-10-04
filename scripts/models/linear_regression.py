from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class LinearRegression:
    def __init__(self):
        self.model = SklearnLinearRegression(fit_intercept=True)

    def fit(self, samples, targets):
        self.model.fit(samples, targets)

    def predict(self, samples):
        return self.model.predict(samples)
