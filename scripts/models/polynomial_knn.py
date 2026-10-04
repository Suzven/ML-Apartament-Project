from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .knn_regression import KNNRegression


class PolynomialKNN(KNNRegression):
    def __init__(self):
        super().__init__()
        pipeline = Pipeline([
            ("input_scaling", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("feature_scaling", StandardScaler()),
            ("regression", KNeighborsRegressor(p=2)),
        ])
        self.model.set_params(estimator=pipeline)
