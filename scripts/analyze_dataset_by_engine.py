from pathlib import Path

import pandas as pd

if __package__:
    from . import model_config
    from .model_evaluation_metrics import ModelEvaluation
    from .entity import DatasetEntity
    from .preprocessing import DatasetPreprocessing
    from .analysis import SegmentedDatasetAnalysis
    from .analysis.grouped_model_evaluation import GroupedModelEvaluation
else:
    import model_config
    from model_evaluation_metrics import ModelEvaluation
    from entity import DatasetEntity
    from preprocessing import DatasetPreprocessing
    from analysis import SegmentedDatasetAnalysis
    from analysis.grouped_model_evaluation import GroupedModelEvaluation

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
ENGINE_ID_COLUMN = 0
CORRELATION_THRESHOLD = 0.8
DRAW_SCATTER = False
TOP_N_SCATTER = 5
ENGINES_TO_PLOT = [1]
REPORT_ROOT = Path(__file__).resolve().parent.parent / "reports" / "model_evaluation_by_engine"


def main(model=None):
    df = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(df, engine_id_column=ENGINE_ID_COLUMN)
    split = DatasetPreprocessing(dataset)
    split.run()

    for name, subset in [("TRAIN", split.train), ("VALIDATION", split.validation)]:
        print(f"\n{name}")
        print("\nDataset shape:")
        print(subset.samples.shape)
        SegmentedDatasetAnalysis(
            subset.samples,
            engine_id_column=ENGINE_ID_COLUMN,
            engine_ids=subset.engine_ids,
            threshold=CORRELATION_THRESHOLD,
            draw_scatter=DRAW_SCATTER,
            top_n=TOP_N_SCATTER,
            engines_to_plot=ENGINES_TO_PLOT,
        ).run()

    if model is None:
        model = model_config.MODEL_CLASS()
    model.fit(split.train.samples, split.train.targets, groups=split.train.engine_ids)
    model_name = type(model).__name__
    for name, subset in [("TRAIN", split.train), ("VALIDATION", split.validation)]:
        predictions = model.predict(subset.samples)
        ModelEvaluation(
            targets=subset.targets,
            predictions=predictions,
            draw_plots=False,
            title=f"{model_name} | {name}",
        ).run()
        GroupedModelEvaluation(
            subset, predictions, name, REPORT_ROOT / model_name
        ).run()


if __name__ == "__main__":
    main()
