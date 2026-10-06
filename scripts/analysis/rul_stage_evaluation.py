from pathlib import Path

import pandas as pd

if __package__.startswith("scripts."):
    from ..model_evaluation_metrics import RegressionMetrics
else:
    from model_evaluation_metrics import RegressionMetrics


class RULStageEvaluation:
    def __init__(self, dataset, predictions, title, report_directory, features=None):
        self.dataset = dataset
        self.predictions = predictions
        self.title = title
        self.report_directory = Path(report_directory)
        self.features = features
        self.stages = ["early", "mid", "late"]
        self.summary = pd.DataFrame()
        self.feature_statistics = pd.DataFrame()
        self.observations = pd.DataFrame()

    def assign_stages(self):
        cycles = self.dataset.cycles
        targets = self.dataset.targets
        if (cycles <= 0).any() or (targets < 0).any():
            raise ValueError("Cycles must be positive and RUL must be nonnegative")
        lifetime = cycles + targets
        lifetime_counts = lifetime.groupby(self.dataset.engine_ids).nunique()
        if (lifetime_counts != 1).any():
            raise ValueError("Cycle + RUL must be constant within each engine")
        progress = cycles / lifetime
        stages = pd.Series("late", index=cycles.index, name="stage")
        stages.loc[progress <= 0.66] = "mid"
        stages.loc[progress <= 0.33] = "early"
        return progress, stages

    def run(self):
        overall = RegressionMetrics(self.dataset.targets, self.predictions)
        progress, stages = self.assign_stages()
        rows = pd.DataFrame(index=self.dataset.samples.index)
        rows["engine_id"] = self.dataset.engine_ids
        rows["cycle"] = self.dataset.cycles
        rows["target"] = self.dataset.targets
        rows["life_fraction"] = progress
        rows["stage"] = stages
        rows["prediction"] = self.predictions
        rows["residual"] = rows["prediction"] - rows["target"]
        self.observations = rows

        features = self.features
        if features is None:
            features = [column for column in self.dataset.samples.columns if isinstance(column, int)]
        summaries = []
        statistics = []
        total_rss = overall.rss()
        for stage in self.stages:
            stage_rows = rows.loc[rows["stage"] == stage]
            if stage_rows.empty:
                continue
            metrics = RegressionMetrics(stage_rows["target"], stage_rows["prediction"])
            rss_share = metrics.rss() / total_rss * 100 if total_rss > 0 else 0
            summaries.append({
                "Stage": stage,
                "Samples": len(stage_rows),
                "Engines": stage_rows["engine_id"].nunique(),
                "RUL min": stage_rows["target"].min(),
                "RUL max": stage_rows["target"].max(),
                "MAE": metrics.mae(),
                "RMSE": metrics.rmse(),
                "Mean Residual": stage_rows["residual"].mean(),
                "RSS": metrics.rss(),
                "RSS share %": rss_share,
            })
            for feature in features:
                values = self.dataset.samples.loc[stage_rows.index, feature]
                correlation = float("nan")
                if values.nunique() > 1 and stage_rows["target"].nunique() > 1:
                    correlation = values.corr(stage_rows["target"])
                statistics.append({
                    "Stage": stage,
                    "Feature": feature,
                    "Mean": values.mean(),
                    "Std": values.std(ddof=0),
                    "Correlation with RUL": correlation,
                })

        self.summary = pd.DataFrame(summaries)
        self.feature_statistics = pd.DataFrame(statistics)
        summary_text = self.summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        statistics_text = self.feature_statistics.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        print(f"\n{self.title} | EVALUATION BY LIFE STAGE")
        print("early: cycle / lifetime <= 0.33; mid: (0.33, 0.66]; late: (0.66, 1]")
        print(summary_text)
        print(f"\n{self.title} | FEATURE STATISTICS BY LIFE STAGE")
        print(statistics_text)
        self.report_directory.mkdir(parents=True, exist_ok=True)
        prefix = self.title.lower()
        (self.report_directory / f"{prefix}_metrics.txt").write_text(summary_text + "\n", encoding="utf-8")
        (self.report_directory / f"{prefix}_features.txt").write_text(statistics_text + "\n", encoding="utf-8")
        (self.report_directory / f"{prefix}_predictions.txt").write_text(rows.to_string(index=True) + "\n", encoding="utf-8")
