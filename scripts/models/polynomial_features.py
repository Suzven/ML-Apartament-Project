from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import PolynomialFeatures


def polynomial_and_history():
    return ColumnTransformer([
        (
            "original_polynomial",
            PolynomialFeatures(degree=2, include_bias=False),
            make_column_selector(pattern=r"^\d+$"),
        ),
        (
            "history",
            "passthrough",
            make_column_selector(pattern=r"^features?_"),
        ),
    ])
