if __package__:
    from .models import Dummy
else:
    from models import Dummy

MODEL_CLASS = Dummy
DRAW_EVALUATION_PLOTS = True
