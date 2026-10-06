import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_score


class ClassificationMetrics:
    def __init__(self, targets, predictions, probabilities, classes):
        self.targets = np.asarray(targets)
        self.predictions = np.asarray(predictions)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.classes = np.asarray(classes)
        if self.targets.ndim != 1 or self.targets.size == 0 or self.targets.shape != self.predictions.shape:
            raise ValueError("Targets and predictions must be matching nonempty vectors")
        if self.probabilities.shape != (self.targets.size, self.classes.size):
            raise ValueError("Probabilities must match samples and model class order")
        if not np.isfinite(self.probabilities).all() or (self.probabilities < 0).any() or (self.probabilities > 1).any():
            raise ValueError("Probabilities must be finite and within [0, 1]")
        if not np.allclose(self.probabilities.sum(axis=1), 1):
            raise ValueError("Class probabilities must sum to one")
        if not np.isin(self.targets, self.classes).all() or not np.isin(self.predictions, self.classes).all():
            raise ValueError("Targets and predictions must belong to model classes")

    def accuracy(self):
        return float(accuracy_score(self.targets, self.predictions))

    def precision(self):
        return float(precision_score(self.targets, self.predictions, labels=self.classes, average="macro", zero_division=0))

    def log_loss(self):
        return float(log_loss(self.targets, self.probabilities, labels=self.classes))


class ClassificationEvaluation:
    def __init__(self, targets, predictions, probabilities, classes, title=""):
        self.metrics = ClassificationMetrics(targets, predictions, probabilities, classes)
        self.title = title

    def run(self):
        print(f"\nCLASSIFICATION EVALUATION | {self.title}")
        print(f"Accuracy: {self.metrics.accuracy():.6f}")
        print(f"Precision (macro): {self.metrics.precision():.6f}")
        print(f"Log Loss: {self.metrics.log_loss():.6f}")
