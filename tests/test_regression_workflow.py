from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import unittest

import numpy as np
import pandas as pd

from scripts.analysis import CorrelationAnalysis
from scripts.analysis.grouped_model_evaluation import GroupedModelEvaluation
from scripts.entity import DatasetEntity
from scripts.history_experiment import HistoryExperiment
from scripts.model_evaluation_metrics import RegressionMetrics
from scripts.model_training import ModelTraining
from scripts.models import (
    DecisionTree,
    Dummy,
    GradientBoosting,
    HistoryPolynomialRidge,
    KNNRegression,
    LinearRegression,
    PolynomialKNN,
    PolynomialLasso,
    PolynomialRidge,
)
from scripts.preprocessing import DataCleaning, DatasetPreprocessing, SensorHistory


def raw_samples(engine_ids=(1, 2, 3, 4, 5, 81), cycles_per_engine=12):
    rows = []
    for engine_id in engine_ids:
        for cycle in range(1, cycles_per_engine + 1):
            row = [engine_id, cycle]
            for feature in range(2, 26):
                value = engine_id * 0.01 + cycle * (feature + 1) * 0.1
                row.append(value)
            rows.append(row)
    return pd.DataFrame(rows)


def prepared_split():
    with redirect_stdout(io.StringIO()):
        dataset = DatasetEntity(raw_samples())
        preprocessing = DatasetPreprocessing(dataset)
        preprocessing.run()
    return preprocessing


def small_grid(model):
    if isinstance(model, (KNNRegression, PolynomialKNN)):
        parameters = {
            "regression__n_neighbors": [3],
            "regression__weights": ["uniform"],
        }
    elif isinstance(model, (PolynomialRidge, HistoryPolynomialRidge)):
        parameters = {"ridge__alpha": [100]}
    elif isinstance(model, PolynomialLasso):
        parameters = {"lasso__alpha": [1]}
    elif isinstance(model, DecisionTree):
        parameters = {"max_depth": [3]}
    elif isinstance(model, GradientBoosting):
        parameters = {"n_estimators": [3], "max_depth": [2]}
    else:
        return
    model.model.set_params(param_grid=parameters, verbose=0)


class DatasetTests(unittest.TestCase):
    def test_split_cleaning_and_alignment(self):
        split = prepared_split()
        self.assertEqual(set(split.train.engine_ids), {1, 2, 3, 4, 5})
        self.assertEqual(set(split.validation.engine_ids), {81})
        self.assertEqual(list(split.train.samples.columns), [1, 6, 7, 8, 11, 13, 15, 16, 18, 21, 24, 25])
        pd.testing.assert_series_equal(split.train.targets, (12 - split.train.cycles).rename("RUL"))
        self.assertTrue(split.train.samples.index.equals(split.train.targets.index))
        self.assertTrue(split.train.samples.index.equals(split.train.engine_ids.index))

    def test_dataset_copy_is_independent(self):
        original = prepared_split().train
        copied = original.copy()
        copied.samples.iloc[0, 0] = 999
        copied.targets.iloc[0] = 999
        copied.engine_ids.iloc[0] = 999
        self.assertEqual(original.cycles.iloc[0], 1)
        self.assertEqual(original.targets.iloc[0], 11)
        self.assertEqual(original.engine_ids.iloc[0], 1)

    def test_unknown_engine_range_is_rejected(self):
        with redirect_stdout(io.StringIO()):
            dataset = DatasetEntity(raw_samples(engine_ids=(101,)))
            with self.assertRaises(ValueError):
                DatasetPreprocessing(dataset).run()


class HistoryTests(unittest.TestCase):
    def clean_dataset(self, samples):
        with redirect_stdout(io.StringIO()):
            dataset = DatasetEntity(samples)
            DataCleaning(dataset).run()
        return dataset

    def add_history(self, dataset):
        with redirect_stdout(io.StringIO()):
            SensorHistory(dataset).run()
        return dataset

    def test_future_changes_do_not_change_past(self):
        samples = raw_samples(engine_ids=(1, 2), cycles_per_engine=40)
        changed = samples.copy()
        changed.loc[changed[1] > 20, 6] += 1000
        baseline = self.add_history(self.clean_dataset(samples))
        modified = self.add_history(self.clean_dataset(changed))
        past_indices = samples.index[samples[1] <= 20]
        pd.testing.assert_frame_equal(
            baseline.samples.loc[past_indices], modified.samples.loc[past_indices]
        )

    def test_prefix_and_engine_isolation(self):
        samples = raw_samples(engine_ids=(1, 2), cycles_per_engine=40)
        full = self.add_history(self.clean_dataset(samples))
        prefix = samples.loc[(samples[0] == 1) & (samples[1] <= 20)]
        shortened = self.add_history(self.clean_dataset(prefix))
        pd.testing.assert_frame_equal(full.samples.loc[prefix.index], shortened.samples)
        changed = samples.copy()
        changed.loc[changed[0] == 2, 6] += 1000
        modified = self.add_history(self.clean_dataset(changed))
        first_engine = samples.index[samples[0] == 1]
        pd.testing.assert_frame_equal(
            full.samples.loc[first_engine], modified.samples.loc[first_engine]
        )

    def test_known_slope_initial_base_and_row_order(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=40)
        samples[6] = 2 * samples[1] + 10
        shuffled = samples.sample(frac=1, random_state=42)
        dataset = self.clean_dataset(shuffled)
        targets_before = dataset.targets.copy()
        self.add_history(dataset)
        self.assertEqual(dataset.samples.shape, (40, 89))
        self.assertTrue(dataset.samples.index.equals(shuffled.index))
        pd.testing.assert_series_equal(dataset.targets, targets_before)
        chronological = dataset.samples.sort_values("1")
        np.testing.assert_allclose(chronological["sensor_6_slope_10"].iloc[1:], 2)
        self.assertEqual(chronological["sensor_6_slope_10"].iloc[0], 0)
        self.assertEqual(chronological["sensor_6_std_10"].iloc[0], 0)
        self.assertEqual(chronological["sensor_6_delta_initial"].iloc[0], 0)
        self.assertEqual(chronological["sensor_6_delta_initial"].iloc[-1], 54)


class MetricsTests(unittest.TestCase):
    def test_metric_formulas(self):
        metrics = RegressionMetrics([0, 2, 4], [1, 1, 6])
        self.assertEqual(metrics.rss(), 6)
        self.assertAlmostEqual(metrics.rmse(), np.sqrt(2))
        self.assertAlmostEqual(metrics.mae(), 4 / 3)
        self.assertAlmostEqual(metrics.rmse_to_std(), np.sqrt(2) / np.std([0, 2, 4]))
        self.assertAlmostEqual(metrics.rmse_to_min_max(), np.sqrt(2) / 4)
        self.assertTrue(np.isnan(RegressionMetrics([1, 1], [1, 2]).rmse_to_std()))

    def test_invalid_predictions_are_rejected(self):
        for targets, predictions in [([], []), ([1], [1, 2]), ([1], [np.nan]), ([[1]], [[1]])]:
            with self.subTest(targets=targets, predictions=predictions):
                with self.assertRaises(ValueError):
                    RegressionMetrics(targets, predictions)
        targets = pd.Series([1, 2], index=[0, 1])
        predictions = pd.Series([1, 2], index=[1, 0])
        with self.assertRaises(ValueError):
            RegressionMetrics(targets, predictions)

    def test_correlation_skips_constants_and_resets_pairs(self):
        analysis = CorrelationAnalysis(pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "constant": [1, 1, 1]}))
        with redirect_stdout(io.StringIO()):
            analysis.run()
            analysis.run()
        self.assertEqual(len(analysis.all_pairs), 1)
        self.assertEqual(len(analysis.strong_pairs), 1)
        self.assertAlmostEqual(analysis.strong_pairs[0][2], 1)


class ModelTests(unittest.TestCase):
    def test_all_models_fit_and_predict_with_grouped_cv(self):
        split = prepared_split()
        model_classes = [
            Dummy, LinearRegression, PolynomialRidge, HistoryPolynomialRidge,
            PolynomialLasso, KNNRegression, PolynomialKNN, DecisionTree, GradientBoosting,
        ]
        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                train = split.train.copy()
                validation = split.validation.copy()
                model = model_class()
                small_grid(model)
                with redirect_stdout(io.StringIO()):
                    if model.requires_history:
                        SensorHistory(train).run()
                        SensorHistory(validation).run()
                    model.fit(train.samples, train.targets, groups=train.engine_ids)
                predictions = model.predict(validation.samples)
                self.assertEqual(predictions.shape, (12,))
                self.assertTrue(np.isfinite(predictions).all())
                if hasattr(model.model, "cv_results_"):
                    self.assertEqual(model.model.n_splits_, 5)
                    for train_indices, held_out_indices in model.model.cv.split(
                        train.samples, train.targets, groups=train.engine_ids
                    ):
                        training_engines = set(train.engine_ids.iloc[train_indices])
                        held_out_engines = set(train.engine_ids.iloc[held_out_indices])
                        self.assertFalse(training_engines.intersection(held_out_engines))

    def test_grid_models_require_engine_ids(self):
        split = prepared_split()
        with self.assertRaises(ValueError):
            PolynomialRidge().fit(split.train.samples, split.train.targets)

    def test_training_facade_adds_history_and_writes_group_reports(self):
        split = prepared_split()
        model = HistoryPolynomialRidge()
        small_grid(model)
        with TemporaryDirectory() as directory:
            training = ModelTraining(model, split.train, split.validation, False, directory)
            with redirect_stdout(io.StringIO()):
                training.run()
            self.assertEqual(training.train.samples.shape[1], 89)
            self.assertEqual(split.train.samples.shape[1], 12)
            self.assertTrue(Path(directory, "validation_metrics.txt").is_file())
            predictions = pd.read_csv(Path(directory, "validation_predictions.txt"), sep=r"\s+")
            self.assertEqual(len(predictions), 12)
            self.assertEqual(set(predictions.engine_id), {81})

    def test_experiment_reports_match_predictions(self):
        split = prepared_split()
        model = DecisionTree()
        small_grid(model)
        with redirect_stdout(io.StringIO()):
            model.fit(split.train.samples, split.train.targets, groups=split.train.engine_ids)
        with TemporaryDirectory() as directory:
            experiment = HistoryExperiment([], directory)
            experiment.record_search("baseline", model)
            experiment.evaluate("baseline", model, "train", split.train)
            experiment.evaluate("baseline", model, "validation", split.validation)
            experiment.save_reports()
            expected = RegressionMetrics(
                split.validation.targets, model.predict(split.validation.samples)
            ).rmse()
            self.assertAlmostEqual(experiment.aggregate_metrics[-1]["RMSE"], expected)
            self.assertEqual(sum(row["rows"] for row in experiment.rul_range_metrics), 12)
            self.assertEqual(len(experiment.search_results), 1)
            self.assertEqual(len(experiment.engine_metrics), 6)
            for name in ["aggregate_metrics.txt", "engine_metrics.txt", "cv_results.txt", "rul_range_metrics.txt"]:
                self.assertTrue(Path(directory, name).is_file())


if __name__ == "__main__":
    unittest.main()
