from sklearn.dummy import DummyRegressor


class Dummy:
    def __init__(self):
        self.model = DummyRegressor(strategy="mean")

    def fit(self, samples, targets, groups=None, validation=None):
        self.model.fit(samples, targets)

    def predict(self, samples):
        return self.model.predict(samples)
