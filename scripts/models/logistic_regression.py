from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self):
        self.model = Pipeline([
            ("scaling", StandardScaler()),
            ("classification", SklearnLogisticRegression(C=1.0, max_iter=5000, random_state=42)),
        ])

    @property
    def classes(self):
        return self.model.classes_

    def fit(self, samples, targets, groups=None):
        self.model.fit(samples.rename(columns=str), targets)

    def predict(self, samples):
        return self.model.predict(samples.rename(columns=str))

    def predict_proba(self, samples):
        return self.model.predict_proba(samples.rename(columns=str))
