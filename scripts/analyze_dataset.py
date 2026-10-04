from pathlib import Path

import pandas as pd

if __package__:
    from .dataset_entity import DatasetEntity
    from .analysis import DatasetAnalysis
else:
    from dataset_entity import DatasetEntity
    from analysis import DatasetAnalysis

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
CORRELATION_THRESHOLD = 0.8
DRAW_SCATTER = False
TOP_N_SCATTER = 10


def main():
    df = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(df)
    print("\nDataset shape:")
    print(dataset.samples.shape)
    DatasetAnalysis(
        dataset.samples,
        threshold=CORRELATION_THRESHOLD,
        draw_scatter=DRAW_SCATTER,
        top_n=TOP_N_SCATTER,
    ).run()


if __name__ == "__main__":
    main()
