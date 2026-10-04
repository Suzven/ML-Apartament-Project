from pathlib import Path

import pandas as pd

if __package__:
    from . import model_config
    from .model_training import ModelTraining
    from .entity import DatasetEntity
    from .preprocessing import DataCleaning, DatasetSplit, StandardScaling
    from .analysis import DatasetAnalysis
else:
    import model_config
    from model_training import ModelTraining
    from entity import DatasetEntity
    from preprocessing import DataCleaning, DatasetSplit, StandardScaling
    from analysis import DatasetAnalysis

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
CORRELATION_THRESHOLD = 0.8
DRAW_SCATTER = False
TOP_N_SCATTER = 10


def main(model=None):
    df = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(df)
    split = DatasetSplit(dataset)
    split.run()

    for name, subset in [("TRAIN", split.train), ("VALIDATION", split.validation)]:
        print(f"\n{name}")
        DataCleaning(subset).run()
        print("\nDataset shape:")
        print(subset.samples.shape)
        DatasetAnalysis(
            subset.samples,
            threshold=CORRELATION_THRESHOLD,
            draw_scatter=DRAW_SCATTER,
            top_n=TOP_N_SCATTER,
        ).run()

    scaling = StandardScaling(split.train, split.validation)
    scaling.run()

    if model is None:
        model = model_config.MODEL_CLASS()
    ModelTraining(
        model=model,
        train=split.train,
        validation=split.validation,
        draw_plots=model_config.DRAW_EVALUATION_PLOTS,
    ).run()


if __name__ == "__main__":
    main()
