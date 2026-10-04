from .decision_tree import DecisionTree
from .dummy import Dummy
from .gradient_boosting import GradientBoosting
from .history_polynomial_ridge import HistoryPolynomialRidge
from .knn_regression import KNNRegression
from .linear_regression import LinearRegression
from .polynomial_knn import PolynomialKNN
from .polynomial_lasso import PolynomialLasso
from .polynomial_ridge import PolynomialRidge

__all__ = [
    "Dummy",
    "LinearRegression",
    "PolynomialRidge",
    "HistoryPolynomialRidge",
    "PolynomialLasso",
    "KNNRegression",
    "PolynomialKNN",
    "DecisionTree",
    "GradientBoosting",
]
