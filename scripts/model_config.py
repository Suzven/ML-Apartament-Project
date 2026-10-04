if __package__:
    from .models import GradientBoosting
else:
    from models import GradientBoosting

MODEL_CLASS = GradientBoosting
DRAW_EVALUATION_PLOTS = True
