from pathlib import Path

import pandas as pd

if __package__.startswith("scripts."):
    from ..model_evaluation_metrics import RegressionMetrics
    from ..prediction_report import prediction_rows, write_table
else:
    from model_evaluation_metrics import RegressionMetrics
    from prediction_report import prediction_rows, write_table


class GroupedModelEvaluation:
    def __init__(self, dataset, predictions, title, report_directory):
        self.dataset = dataset
        self.predictions = predictions
        self.title = title
        self.report_directory = Path(report_directory)

    def run(self):
        predictions_df = prediction_rows(self.dataset, self.predictions)
        summary = []

        for engine_id, rows in predictions_df.groupby("engine_id", sort=True):
            metrics = RegressionMetrics(rows["target"], rows["prediction"])
            summary.append({
                "engine_id": engine_id,
                "rows": len(rows),
                "RSS": metrics.rss(),
                "RMSE": metrics.rmse(),
                "MAE": metrics.mae(),
                "RMSE_STD": metrics.rmse_to_std(),
                "RMSE_RANGE": metrics.rmse_to_min_max(),
                "mean_residual": rows["residual"].mean(),
                "max_absolute_error": rows["residual"].abs().max(),
            })
            print(f"\n{self.title} | ENGINE {engine_id} | prediction preview")
            print(rows.head(5).to_string(index=True))
            print(rows.tail(5).to_string(index=True))

        summary_df = pd.DataFrame(summary).sort_values("RMSE", ascending=False)
        print(f"\n{self.title} | METRICS BY ENGINE (worst RMSE first)")
        print(summary_df.to_string(index=False))
        predictions_path = self.report_directory / f"{self.title.lower()}_predictions.txt"
        metrics_path = self.report_directory / f"{self.title.lower()}_metrics.txt"
        write_table(predictions_path, predictions_df, include_index=True)
        write_table(metrics_path, summary_df)
