if __package__:
    from .models import LogisticRegression
else:
    from models import LogisticRegression

MODEL_CLASS = LogisticRegression
