from .decision_tree import DecisionTree
from .dummy import Dummy
from .knn_regression import KNNRegression
from .linear_regression import LinearRegression
from .polynomial_ridge import PolynomialRidge
from .ridge import Ridge
from .gradient_boosting import GradientBoosting
from .polynomial_lasso import PolynomialLasso
from .logistic_regression import LogisticRegression

__all__ = [
    "Dummy",
    "LinearRegression",
    "PolynomialRidge",
    "Ridge",
    "GradientBoosting",
    "PolynomialLasso",
    "LogisticRegression",
    "KNNRegression",
    "DecisionTree",
]
