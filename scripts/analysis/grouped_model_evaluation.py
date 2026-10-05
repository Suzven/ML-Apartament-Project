from pathlib import Path

import pandas as pd

if __package__.startswith("scripts."):
    from ..model_evaluation_metrics import RegressionMetrics
else:
    from model_evaluation_metrics import RegressionMetrics


class GroupedModelEvaluation:
    def __init__(self, dataset, predictions, title, report_directory, mean_features=None):
        self.dataset = dataset
        self.predictions = predictions
        self.title = title
        self.report_directory = Path(report_directory)
        self.mean_features = [] if mean_features is None else mean_features

    def run(self):
        RegressionMetrics(self.dataset.targets, self.predictions)
        predictions_df = pd.DataFrame(index=self.dataset.samples.index)
        predictions_df["engine_id"] = self.dataset.engine_ids
        predictions_df["cycle"] = self.dataset.cycles
        predictions_df["target"] = self.dataset.targets
        predictions_df["prediction"] = self.predictions
        predictions_df["residual"] = predictions_df["prediction"] - predictions_df["target"]
        summary = []

        for engine_id, rows in predictions_df.groupby("engine_id", sort=True):
            metrics = RegressionMetrics(rows["target"], rows["prediction"])
            engine_summary = {
                "Group": engine_id,
                "N": rows["cycle"].nunique(),
                "MAE": metrics.mae(),
                "RMSE": metrics.rmse(),
                "Mean Residual": rows["residual"].mean(),
            }
            engine_samples = self.dataset.samples.loc[rows.index]
            for feature in self.mean_features:
                engine_summary[f"Mean Feature {feature}"] = engine_samples[feature].mean()
            summary.append(engine_summary)

        summary_df = pd.DataFrame(summary).sort_values("Group")
        summary_text = summary_df.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        print(f"\n{self.title} | METRICS AND FEATURE MEANS BY ENGINE")
        print(summary_text)
        predictions_path = self.report_directory / f"{self.title.lower()}_predictions.txt"
        metrics_path = self.report_directory / f"{self.title.lower()}_metrics.txt"
        self.report_directory.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(predictions_df.to_string(index=True) + "\n", encoding="utf-8")
        metrics_path.write_text(summary_text + "\n", encoding="utf-8")
