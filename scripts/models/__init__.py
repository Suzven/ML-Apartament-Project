from .decision_tree import DecisionTree
from .dummy import Dummy
from .knn_regression import KNNRegression
from .linear_regression import LinearRegression
from .polynomial_ridge import PolynomialRidge
from .ridge import Ridge
from .gradient_boosting import GradientBoosting
from .polynomial_lasso import PolynomialLasso

__all__ = [
    "Dummy",
    "LinearRegression",
    "PolynomialRidge",
    "Ridge",
    "GradientBoosting",
    "PolynomialLasso",
    "KNNRegression",
    "DecisionTree",
]
