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
from scripts.model_evaluation_metrics import RegressionMetrics
from scripts.models import (
    DecisionTree,
    Dummy,
    KNNRegression,
    LinearRegression,
    PolynomialRidge,
)
from scripts.preprocessing import DatasetPreprocessing


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
    if isinstance(model, KNNRegression):
        parameters = {
            "regression__n_neighbors": [3],
            "regression__weights": ["uniform"],
        }
    elif isinstance(model, PolynomialRidge):
        parameters = {"ridge__alpha": [100]}
    elif isinstance(model, DecisionTree):
        parameters = {"max_depth": [3]}
    else:
        return
    model.model.set_params(param_grid=parameters, verbose=0)


class DatasetTests(unittest.TestCase):
    def test_split_cleaning_and_alignment(self):
        split = prepared_split()
        self.assertEqual(set(split.train.engine_ids), {1, 2, 3, 4, 5})
        self.assertEqual(set(split.validation.engine_ids), {81})
        self.assertEqual(list(split.train.samples.columns), [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 24, 25])
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
        model_classes = [Dummy, LinearRegression, PolynomialRidge, KNNRegression, DecisionTree]
        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                train = split.train.copy()
                validation = split.validation.copy()
                model = model_class()
                small_grid(model)
                with redirect_stdout(io.StringIO()):
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

    def test_group_reports_match_predictions(self):
        split = prepared_split()
        model = Dummy()
        model.fit(split.train.samples, split.train.targets)
        predictions = model.predict(split.validation.samples)
        with TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()):
                GroupedModelEvaluation(split.validation, predictions, "VALIDATION", directory).run()
            rows = pd.read_csv(Path(directory, "validation_predictions.txt"), sep=r"\s+")
            metrics = pd.read_csv(Path(directory, "validation_metrics.txt"), sep=r"\s+")
            self.assertEqual(len(rows), 12)
            self.assertEqual(set(rows.engine_id), {81})
            expected = RegressionMetrics(split.validation.targets, predictions).rmse()
            self.assertAlmostEqual(metrics.RMSE.iloc[0], expected, places=5)


if __name__ == "__main__":
    unittest.main()
