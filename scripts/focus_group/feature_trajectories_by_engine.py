from pathlib import Path
import sys

import pandas as pd

if __package__ and __package__.startswith("scripts."):
    from ..entity import DatasetEntity
    from ..preprocessing import DataCleaning
    from .focus_group_by_engine_id import ENGINE_IDS
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from entity import DatasetEntity
    from preprocessing import DataCleaning
    from focus_group.focus_group_by_engine_id import ENGINE_IDS

DATASET_PATH = Path(__file__).resolve().parents[1] / "train_FD001.txt"
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "focus_group" / "features"


class FocusGroupFeatureTrajectories:
    def __init__(self, dataset, engine_ids=None):
        self.dataset = dataset
        self.engine_ids = ENGINE_IDS if engine_ids is None else engine_ids
        self.rolling_window = 15

    def run(self, draw_plots=True, output_directory=OUTPUT_DIRECTORY):
        import matplotlib.pyplot as plt

        engine_samples = {}
        for engine_id in self.engine_ids:
            engine_mask = self.dataset.engine_ids == engine_id
            rows = self.dataset.samples.loc[engine_mask]
            if rows.empty:
                raise ValueError(f"Samples for engine {engine_id} are missing")
            engine_samples[engine_id] = rows.sort_values(self.dataset.cycle_column)

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for feature in self.dataset.samples.columns:
            figure, axis = plt.subplots(figsize=(12, 6))
            for engine_id in self.engine_ids:
                rows = engine_samples[engine_id]
                cycles = rows[self.dataset.cycle_column]
                values = rows[feature]
                rolling_mean = values.rolling(self.rolling_window, min_periods=1).mean()
                axis.plot(cycles, rolling_mean, linewidth=1, alpha=0.8, label=f"Engine {engine_id}")

            axis.set(
                title=f"Feature {feature} | Focus engines | Rolling mean ({self.rolling_window} cycles)",
                xlabel="Cycle",
                ylabel=f"Feature {feature}",
            )
            axis.grid(alpha=0.2)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output_directory / f"feature_{feature}_by_cycle.png", dpi=150)
            if draw_plots:
                plt.show()
            plt.close(figure)

        print("Focus engine IDs:", self.engine_ids)
        print("Plotted features:", self.dataset.samples.columns.tolist())
        print("Plots:", output_directory)


def main(draw_plots=True):
    samples = pd.read_csv(DATASET_PATH, sep=r"\s+", header=None)
    dataset = DatasetEntity(samples)
    DataCleaning(dataset).run()
    FocusGroupFeatureTrajectories(dataset).run(draw_plots=draw_plots)


if __name__ == "__main__":
    main()
