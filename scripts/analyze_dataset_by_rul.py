from pathlib import Path

import pandas as pd

if __package__:
    from . import model_config
    from .entity import DatasetEntity
    from .preprocessing import DatasetPreprocessing
    from .analysis.rul_stage_evaluation import RULStageEvaluation
else:
    import model_config
    from entity import DatasetEntity
    from preprocessing import DatasetPreprocessing
    from analysis.rul_stage_evaluation import RULStageEvaluation

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports" / "model_evaluation_by_rul"
ANALYZE_ENGINEERED_FEATURES = False


def main(model=None):
    samples = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(samples)
    preprocessing = DatasetPreprocessing(dataset)
    preprocessing.run()
    if model is None:
        model = model_config.MODEL_CLASS()
    model.fit(
        preprocessing.train.samples,
        preprocessing.train.targets,
        groups=preprocessing.train.engine_ids,
    )
    model_name = type(model).__name__
    for name, subset in [("TRAIN", preprocessing.train), ("VALIDATION", preprocessing.validation)]:
        features = list(subset.samples.columns) if ANALYZE_ENGINEERED_FEATURES else None
        predictions = model.predict(subset.samples)
        RULStageEvaluation(
            subset, predictions, name, REPORT_ROOT / model_name, features=features,
        ).run()


if __name__ == "__main__":
    main()
