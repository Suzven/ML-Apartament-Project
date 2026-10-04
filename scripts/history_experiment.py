import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .entity import DatasetEntity
    from .model_evaluation_metrics import RegressionMetrics
    from .prediction_report import prediction_rows, write_table
    from .preprocessing import DatasetPreprocessing, SensorHistory
else:
    from entity import DatasetEntity
    from model_evaluation_metrics import RegressionMetrics
    from prediction_report import prediction_rows, write_table
    from preprocessing import DatasetPreprocessing, SensorHistory


class HistoryExperiment:
    def __init__(self, models, output_directory):
        self.models = models
        self.output_directory = Path(output_directory)
        self.aggregate_metrics = []
        self.engine_metrics = []
        self.search_results = []
        self.rul_range_metrics = []

    def run(self):
        dataset_path = Path(__file__).with_name("train_FD001.txt")
        samples = pd.read_csv(dataset_path, sep=r"\s+", header=None)
        preprocessing = DatasetPreprocessing(DatasetEntity(samples))
        preprocessing.run()

        train_history = self.add_history(preprocessing.train)
        validation_history = self.add_history(preprocessing.validation)

        for name, model, use_history in self.models:
            train = preprocessing.train
            validation = preprocessing.validation
            if use_history:
                train = train_history
                validation = validation_history

            print("\nEXPERIMENT:", name, flush=True)
            model.fit(train.samples, train.targets, groups=train.engine_ids)
            self.record_search(name, model)
            self.evaluate(name, model, "train", train)
            self.evaluate(name, model, "validation", validation)
            self.save_reports()

        self.plot_comparison()
        self.print_results()

    def add_history(self, dataset):
        augmented = dataset.copy()
        SensorHistory(augmented).run()
        pd.testing.assert_series_equal(dataset.targets, augmented.targets)
        assert dataset.samples.index.equals(augmented.samples.index)
        assert np.isfinite(augmented.samples.to_numpy()).all()
        return augmented

    def record_search(self, name, model):
        results = model.model.cv_results_
        number_of_folds = model.model.n_splits_
        for candidate_index, parameters in enumerate(results["params"]):
            row = {
                "experiment": name,
                "parameters": json.dumps(parameters, separators=(",", ":")),
            }
            for fold_index in range(number_of_folds):
                score_name = f"split{fold_index}_test_score"
                row[f"fold_{fold_index + 1}_RMSE"] = -results[score_name][candidate_index]
            self.search_results.append(row)

    def evaluate(self, name, model, split_name, dataset):
        predictions = model.predict(dataset.samples)
        metrics = RegressionMetrics(dataset.targets, predictions)
        self.aggregate_metrics.append({
            "experiment": name,
            "split": split_name,
            "best_parameters": json.dumps(model.model.best_params_, separators=(",", ":")),
            "CV_RMSE": -model.model.best_score_,
            "RMSE": metrics.rmse(),
            "MAE": metrics.mae(),
        })

        rows = prediction_rows(dataset, predictions)
        self.record_engine_metrics(name, split_name, rows)
        if split_name == "validation":
            self.record_rul_ranges(name, rows)

        path = self.output_directory / f"{name}_{split_name}_predictions.txt"
        write_table(path, rows, include_index=True)

    def record_engine_metrics(self, name, split_name, rows):
        for engine_id, engine_rows in rows.groupby("engine_id"):
            engine_rows = engine_rows.sort_values("cycle")
            metrics = RegressionMetrics(engine_rows["target"], engine_rows["prediction"])
            residuals = engine_rows["residual"]
            self.engine_metrics.append({
                "experiment": name,
                "split": split_name,
                "engine": engine_id,
                "RMSE": metrics.rmse(),
                "MAE": metrics.mae(),
                "bias": residuals.mean(),
                "first25_bias": residuals.head(25).mean(),
                "last30_bias": residuals.tail(30).mean(),
            })

    def record_rul_ranges(self, name, rows):
        rows["RUL_range"] = pd.cut(
            rows["target"],
            bins=[-1, 30, 100, 200, np.inf],
            labels=["0-30", "31-100", "101-200", "201+"],
        )
        for interval, interval_rows in rows.groupby("RUL_range", observed=True):
            metrics = RegressionMetrics(interval_rows["target"], interval_rows["prediction"])
            self.rul_range_metrics.append({
                "experiment": name,
                "RUL_range": interval,
                "rows": len(interval_rows),
                "RMSE": metrics.rmse(),
                "MAE": metrics.mae(),
                "bias": interval_rows["residual"].mean(),
            })

    def save_reports(self):
        write_table(
            self.output_directory / "aggregate_metrics.txt",
            pd.DataFrame(self.aggregate_metrics),
        )
        write_table(
            self.output_directory / "engine_metrics.txt",
            pd.DataFrame(self.engine_metrics),
        )
        write_table(
            self.output_directory / "cv_results.txt",
            pd.DataFrame(self.search_results),
        )
        write_table(
            self.output_directory / "rul_range_metrics.txt",
            pd.DataFrame(self.rul_range_metrics),
        )

    def print_results(self):
        aggregate = pd.DataFrame(self.aggregate_metrics)
        grouped = pd.DataFrame(self.engine_metrics)
        selected_engines = grouped["engine"].isin([92, 96, 90, 100])
        validation_rows = grouped["split"] == "validation"
        selected = grouped.loc[validation_rows & selected_engines]
        print("\nAGGREGATE RESULTS\n", aggregate.to_string(index=False))
        print("\nSELECTED VALIDATION ENGINES\n", selected.to_string(index=False))

    def plot_comparison(self):
        import matplotlib.pyplot as plt

        baseline_path = self.output_directory / "baseline_validation_predictions.txt"
        history_path = self.output_directory / "history_validation_predictions.txt"
        baseline = pd.read_csv(baseline_path, sep=r"\s+")
        history = pd.read_csv(history_path, sep=r"\s+")
        figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

        for engine_id, axis in zip([92, 96, 90, 100], axes.flat):
            original = baseline.loc[baseline["engine_id"] == engine_id].sort_values("cycle")
            augmented = history.loc[history["engine_id"] == engine_id].sort_values("cycle")
            baseline_mean = original["prediction"].rolling(15, min_periods=1).mean()
            history_mean = augmented["prediction"].rolling(15, min_periods=1).mean()
            axis.plot(
                original["cycle"], original["target"],
                color="black", linestyle="--", label="Actual RUL",
            )
            axis.plot(original["cycle"], baseline_mean, color="#999999", label="Baseline")
            axis.plot(
                augmented["cycle"], history_mean,
                color="#1769aa", label="With sensor history",
            )
            axis.set(title=f"Engine {engine_id}", xlabel="Cycle", ylabel="RUL (cycles)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)

        figure.suptitle("Held-out engines | predictions shown as trailing 15-cycle means")
        figure.tight_layout()
        figure.savefig(self.output_directory / "validation_comparison.png", dpi=150)
        plt.close(figure)
