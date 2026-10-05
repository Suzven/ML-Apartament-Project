from pathlib import Path
import sys

import pandas as pd

if __package__ and __package__.startswith("scripts."):
    from .. import model_config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import model_config

ENGINE_IDS = [90, 92, 96, 100, 82]
REPORT_ROOT = Path(__file__).resolve().parents[2] / "reports" / "model_evaluation_by_engine"


class FocusGroupByEngineId:
    def __init__(self, predictions, engine_ids=None):
        self.predictions = predictions
        self.engine_ids = ENGINE_IDS if engine_ids is None else engine_ids

    def run(self, draw_plots=True, output_path=None):
        import matplotlib.pyplot as plt

        selected = self.predictions[self.predictions["engine_id"].isin(self.engine_ids)]
        for engine_id in self.engine_ids:
            if not (selected["engine_id"] == engine_id).any():
                raise ValueError(f"Predictions for engine {engine_id} are missing")

        figure, axes = plt.subplots(
            len(self.engine_ids), 1, figsize=(12, 4 * len(self.engine_ids)), squeeze=False
        )
        for engine_id, axis in zip(self.engine_ids, axes[:, 0]):
            rows = selected[selected["engine_id"] == engine_id].sort_values("cycle")
            axis.scatter(rows["cycle"], rows["target"], s=12, alpha=0.7, label="Target (y)")
            axis.scatter(
                rows["cycle"], rows["prediction"], s=12, alpha=0.7, label="Prediction (ŷ)"
            )
            axis.set(title=f"Engine {engine_id}", xlabel="Cycle", ylabel="RUL (cycles)")
            axis.grid(alpha=0.2)
            axis.legend()

        figure.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=150)
        if draw_plots:
            plt.show()
        plt.close(figure)


def main():
    model_name = model_config.MODEL_CLASS.__name__
    report_directory = REPORT_ROOT / model_name
    predictions_path = report_directory / "validation_predictions.txt"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction report not found: {predictions_path}. "
            "Run scripts/analyze_dataset_by_engine.py first."
        )

    predictions = pd.read_csv(predictions_path, sep=r"\s+")
    output_path = report_directory / "focus_group" / "predictions_by_cycle.png"
    FocusGroupByEngineId(predictions).run(output_path=output_path)
    print("Focus engine IDs:", ENGINE_IDS)
    print("Predictions:", predictions_path)
    print("Plot:", output_path)


if __name__ == "__main__":
    main()
