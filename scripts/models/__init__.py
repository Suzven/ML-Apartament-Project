from .decision_tree import DecisionTree
from .dummy import Dummy
from .knn_regression import KNNRegression
from .linear_regression import LinearRegression
from .polynomial_ridge import PolynomialRidge

__all__ = [
    "Dummy",
    "LinearRegression",
    "PolynomialRidge",
    "KNNRegression",
    "DecisionTree",
]
