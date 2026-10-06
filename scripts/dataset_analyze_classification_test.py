from pathlib import Path
import json

import joblib
import pandas as pd

if __package__:
    from . import classification_model_config
    from .classification_evaluation_metrics import ClassificationEvaluation
    from .entity import ClassificationDatasetEntity
    from .preprocessing import DataCleaning, FeatureEngineering
    from .analysis.grouped_classification_evaluation import GroupedClassificationEvaluation
else:
    import classification_model_config
    from classification_evaluation_metrics import ClassificationEvaluation
    from entity import ClassificationDatasetEntity
    from preprocessing import DataCleaning, FeatureEngineering
    from analysis.grouped_classification_evaluation import GroupedClassificationEvaluation

DATA_DIRECTORY = Path(__file__).resolve().parent
REPORT_DIRECTORY = DATA_DIRECTORY.parent / "reports" / "classification_test"


def load_datasets():
    train_samples = pd.read_csv(DATA_DIRECTORY / "train_FD001.txt", sep=r"\s+", header=None)
    test_samples = pd.read_csv(DATA_DIRECTORY / "test_FD001.txt", sep=r"\s+", header=None)
    remaining_cycles = pd.read_csv(DATA_DIRECTORY / "RUL_FD001.txt", sep=r"\s+", header=None)
    if train_samples.shape[1] != 26 or test_samples.shape[1] != 26:
        raise ValueError("Train and test must contain exactly 26 input columns")
    expected_ids = list(range(1, len(remaining_cycles) + 1))
    if remaining_cycles.shape[1] != 1 or sorted(test_samples[0].unique()) != expected_ids:
        raise ValueError("RUL file rows must match test engine IDs 1..N")
    remaining_by_engine = pd.Series(remaining_cycles[0].to_numpy(), index=expected_ids)
    train = ClassificationDatasetEntity(train_samples)
    test = ClassificationDatasetEntity(test_samples, remaining_cycles_by_engine=remaining_by_engine)
    return train, test


def main(model=None):
    train, test = load_datasets()
    for dataset in (train, test):
        DataCleaning(dataset).run()
        FeatureEngineering(dataset).run()
    if model is None:
        model = classification_model_config.MODEL_CLASS()
    model.fit(train.samples, train.class_targets())
    last_cycle = test.cycles.groupby(test.engine_ids).transform("max")
    last_samples = test.select_rows(test.cycles == last_cycle)
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = {
        "classes": train.resource_groups,
        "features": train.samples.shape[1],
        "training_engines": int(train.engine_ids.nunique()),
        "training_samples": len(train.samples),
    }
    for name, dataset in [("TRAIN_FULL", train), ("TEST_ALL_CYCLES", test), ("TEST_LAST_CYCLE", last_samples)]:
        predictions = model.predict(dataset.samples)
        probabilities = model.predict_proba(dataset.samples)
        evaluation = ClassificationEvaluation(
            dataset.class_targets(), predictions, probabilities, model.classes,
            title=f"{type(model).__name__} | {name}",
        )
        evaluation.run()
        grouped = GroupedClassificationEvaluation(
            dataset, predictions, probabilities, model.classes, name, REPORT_DIRECTORY,
        )
        grouped.run()
        summary[name.lower()] = {
            "samples": len(dataset.samples),
            "engines": int(dataset.engine_ids.nunique()),
            "accuracy": evaluation.metrics.accuracy(),
            "precision_macro": evaluation.metrics.precision(),
            "log_loss": evaluation.metrics.log_loss(),
            "groups": grouped.summary.to_dict(orient="records"),
        }
    (REPORT_DIRECTORY / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump(model, REPORT_DIRECTORY / "logistic_regression.joblib")
    print("\nReports and trained model:", REPORT_DIRECTORY)


if __name__ == "__main__":
    main()
