from sklearn.dummy import DummyRegressor

from .base import RegressionModel


class Dummy(RegressionModel):
    def __init__(self):
        self.model = DummyRegressor(strategy="mean")
