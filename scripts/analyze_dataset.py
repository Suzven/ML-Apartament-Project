from pathlib import Path

import pandas as pd

if __package__:
    from . import model_config
    from .model_evaluation_metrics import ModelEvaluation
    from .entity import DatasetEntity
    from .preprocessing import DatasetPreprocessing
    from .analysis import DatasetAnalysis
else:
    import model_config
    from model_evaluation_metrics import ModelEvaluation
    from entity import DatasetEntity
    from preprocessing import DatasetPreprocessing
    from analysis import DatasetAnalysis

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
CORRELATION_THRESHOLD = 0.8
DRAW_SCATTER = False
TOP_N_SCATTER = 10


def main(model=None):
    df = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(df)
    split = DatasetPreprocessing(dataset)
    split.run()

    for name, subset in [("TRAIN", split.train), ("VALIDATION", split.validation)]:
        print(f"\n{name}")
        print("\nDataset shape:")
        print(subset.samples.shape)
        DatasetAnalysis(
            subset.samples,
            threshold=CORRELATION_THRESHOLD,
            draw_scatter=DRAW_SCATTER,
            top_n=TOP_N_SCATTER,
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
            draw_plots=model_config.DRAW_EVALUATION_PLOTS,
            title=f"{model_name} | {name}",
        ).run()


if __name__ == "__main__":
    main()
