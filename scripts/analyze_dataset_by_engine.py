from pathlib import Path

import pandas as pd

if __package__:
    from .analysis import SegmentedDatasetAnalysis
else:
    from analysis import SegmentedDatasetAnalysis

DATASET_PATH = Path(__file__).with_name("train_FD001.txt")
ENGINE_ID_COLUMN = 0
CORRELATION_THRESHOLD = 0.8
DRAW_SCATTER = False
TOP_N_SCATTER = 5
ENGINES_TO_PLOT = [1]


def main():
    df = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    print("\nDataset:")
    print(df.head())
    print("\nDataset shape:")
    print(df.shape)
    SegmentedDatasetAnalysis(
        df,
        engine_id_column=ENGINE_ID_COLUMN,
        threshold=CORRELATION_THRESHOLD,
        draw_scatter=DRAW_SCATTER,
        top_n=TOP_N_SCATTER,
        engines_to_plot=ENGINES_TO_PLOT,
    ).run()


if __name__ == "__main__":
    main()
