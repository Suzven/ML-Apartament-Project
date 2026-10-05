from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import pandas as pd

from scripts.entity import DatasetEntity
from scripts.models import GradientBoosting, LinearRegression, PolynomialLasso, PolynomialRidge, Ridge
from scripts.preprocessing import DataCleaning, DatasetPreprocessing, FeatureEngineering
from tests.test_regression_workflow import raw_samples


def engineered_dataset(samples):
    with redirect_stdout(io.StringIO()):
        dataset = DatasetEntity(samples)
        DataCleaning(dataset).run()
        FeatureEngineering(dataset).run()
    return dataset


class FeatureEngineeringTests(unittest.TestCase):
    def test_lasso_uses_604_features_and_selects_coefficients(self):
        dataset = engineered_dataset(raw_samples(engine_ids=(1, 2, 3, 4, 5), cycles_per_engine=12))
        model = PolynomialLasso()
        model.model.set_params(param_grid={"lasso__alpha": [1]}, verbose=0)
        with redirect_stdout(io.StringIO()):
            model.fit(dataset.samples, dataset.targets, groups=dataset.engine_ids)
        estimator = model.model.best_estimator_.named_steps["lasso"]
        self.assertEqual(estimator.n_features_in_, 604)
        self.assertEqual(model.model.n_splits_, 5)
        self.assertTrue(np.isfinite(model.predict(dataset.samples)).all())
        self.assertLess(np.count_nonzero(estimator.coef_), 604)

    def test_boosting_uses_same_604_features_and_grouped_search(self):
        dataset = engineered_dataset(raw_samples(engine_ids=(1, 2, 3, 4, 5), cycles_per_engine=12))
        model = GradientBoosting()
        model.model.set_params(
            param_grid={"boosting__learning_rate": [0.1]},
            estimator__boosting__max_iter=5,
            verbose=0,
        )
        with redirect_stdout(io.StringIO()):
            model.fit(dataset.samples, dataset.targets, groups=dataset.engine_ids)
        estimator = model.model.best_estimator_.named_steps["boosting"]
        self.assertEqual(estimator.n_features_in_, 604)
        self.assertFalse(estimator.early_stopping)
        self.assertEqual(model.model.n_splits_, 5)
        self.assertTrue(np.isfinite(model.predict(dataset.samples)).all())

    def test_future_rows_do_not_change_past_features(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=70)
        full = engineered_dataset(samples)
        for length in (1, 8, 15, 22, 45):
            with self.subTest(length=length):
                prefix = engineered_dataset(samples.iloc[:length])
                pd.testing.assert_frame_equal(prefix.samples, full.samples.iloc[:length])

    def test_other_engines_do_not_change_features_or_row_order(self):
        samples = raw_samples(engine_ids=(1, 81), cycles_per_engine=35)
        shuffled = samples.sample(frac=1, random_state=7)
        combined = engineered_dataset(shuffled)
        separate = engineered_dataset(samples.loc[samples[0] == 1])
        pd.testing.assert_frame_equal(combined.samples.loc[separate.samples.index], separate.samples)
        self.assertTrue(combined.samples.index.equals(shuffled.index))
        self.assertTrue(combined.samples.index.equals(combined.targets.index))

    def test_counts_and_constant_windows(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=35)
        samples.loc[:, 6] = 4.0
        dataset = engineered_dataset(samples)
        self.assertEqual(dataset.samples.shape[1], 394)
        self.assertTrue(np.isfinite(dataset.samples.to_numpy()).all())
        for name in ("feature_6_slope_30", "feature_6_slope_change_30", "feature_6_range_position_30"):
            self.assertTrue((dataset.samples[name] == 0).all())

    def test_known_slope_and_rare_events(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=35)
        samples.loc[:, 10] = 21.61
        samples.loc[4, 10] = 21.60
        dataset = engineered_dataset(samples)
        self.assertAlmostEqual(dataset.samples.loc[20, "feature_6_slope_15"], 0.7)
        self.assertAlmostEqual(dataset.samples.loc[20, "feature_10_low_fraction_30"], 1 / 21)
        self.assertEqual(dataset.samples.loc[20, "feature_10_cycles_since_low"], 16)
        self.assertEqual(dataset.samples.loc[3, "feature_10_low_seen"], 0)
        self.assertEqual(dataset.samples.loc[4, "feature_10_low_seen"], 1)
        self.assertAlmostEqual(dataset.samples.loc[20, "feature_6_initial_mean"], samples[6].iloc[:15].mean())

    def test_preprocessing_enables_features_and_preserves_targets(self):
        with redirect_stdout(io.StringIO()):
            dataset = DatasetEntity(raw_samples())
            preprocessing = DatasetPreprocessing(dataset)
            preprocessing.run()
        for subset in (preprocessing.train, preprocessing.validation):
            self.assertEqual(subset.samples.shape[1], 394)
            pd.testing.assert_series_equal(subset.targets, (12 - subset.cycles).rename("RUL"))

    def test_model_pipelines_accept_engineered_columns(self):
        dataset = engineered_dataset(raw_samples(engine_ids=(1, 2, 3, 4, 5), cycles_per_engine=12))
        for model in (LinearRegression(), Ridge(), PolynomialRidge()):
            if isinstance(model, (Ridge, PolynomialRidge)):
                model.model.set_params(param_grid={"ridge__alpha": [100]})
            with redirect_stdout(io.StringIO()):
                model.fit(dataset.samples, dataset.targets, groups=dataset.engine_ids)
            predictions = model.predict(dataset.samples)
            self.assertTrue(np.isfinite(predictions).all())
            if isinstance(model, Ridge):
                self.assertEqual(model.model.best_estimator_.named_steps["ridge"].n_features_in_, 394)
            if isinstance(model, PolynomialRidge):
                pipeline = model.model.best_estimator_
                self.assertEqual(pipeline.named_steps["ridge"].n_features_in_, 604)
                scaler = pipeline.named_steps["input_scaling"]
                scaled = scaler.transform(model.model_samples(dataset.samples))
                expansion = pipeline.named_steps["polynomial"]
                transformed = expansion.transform(scaled)
                history_columns = [column for column in scaled.columns if column.startswith("feature")]
                np.testing.assert_allclose(transformed[:, 230:], scaled[history_columns].to_numpy())


if __name__ == "__main__":
    unittest.main()
