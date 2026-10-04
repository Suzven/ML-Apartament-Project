if __package__:
    from .models import LinearRegression
else:
    from models import LinearRegression

MODEL_CLASS = LinearRegression
DRAW_EVALUATION_PLOTS = True
