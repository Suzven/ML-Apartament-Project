if __package__:
    from .models import PolynomialRidge
else:
    from models import PolynomialRidge

MODEL_CLASS = PolynomialRidge
DRAW_EVALUATION_PLOTS = True
