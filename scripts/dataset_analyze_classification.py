from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import classification_model_config
    from .classification_evaluation_metrics import ClassificationEvaluation
    from .entity import ClassificationDatasetEntity
    from .preprocessing import DatasetPreprocessing
    from .analysis.grouped_classification_evaluation import GroupedClassificationEvaluation
else:
    import classification_model_config
    from classification_evaluation_metrics import ClassificationEvaluation
    from entity import ClassificationDatasetEntity
    from preprocessing import DatasetPreprocessing
    from analysis.grouped_classification_evaluation import GroupedClassificationEvaluation

DATASET_PATH = Path(__file__).with_name("train_FD001_classification.txt")
REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports" / "classification_by_group"


def main(model=None, grouped=False):
    data = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    if data.shape[1] != 27:
        raise ValueError("Classification file must contain 26 inputs and remaining_fraction")
    dataset = ClassificationDatasetEntity(data.iloc[:, :26])
    if not np.allclose(data[26], dataset.targets, rtol=0, atol=1e-12):
        raise ValueError("Stored remaining_fraction does not match engine trajectories")
    preprocessing = DatasetPreprocessing(dataset)
    preprocessing.run()
    if model is None:
        model = classification_model_config.MODEL_CLASS()
    model.fit(preprocessing.train.samples, preprocessing.train.class_targets())
    for name, subset in [("TRAIN", preprocessing.train), ("VALIDATION", preprocessing.validation)]:
        predictions = model.predict(subset.samples)
        probabilities = model.predict_proba(subset.samples)
        ClassificationEvaluation(
            subset.class_targets(),
            predictions,
            probabilities,
            model.classes,
            title=f"{type(model).__name__} | {name}",
        ).run()
        if grouped:
            GroupedClassificationEvaluation(
                subset, predictions, probabilities, model.classes, name,
                REPORT_ROOT / type(model).__name__,
            ).run()


if __name__ == "__main__":
    main()
