import unittest

import numpy as np
import pandas as pd

from scripts.entity import DatasetEntity
from scripts.preprocessing import DataCleaning
from scripts.preprocessing.dl_feature_engineering import DLFeatureEngineering
from tests.test_regression_workflow import raw_samples


def prepare(samples, fill_missing=True):
    dataset = DatasetEntity(samples, verbose=False)
    DataCleaning(dataset).run(verbose=False)
    DLFeatureEngineering(dataset, fill_missing=fill_missing).run()
    return dataset


class DLFeatureEngineeringTests(unittest.TestCase):
    def test_nan_history_preserves_missing_windows_and_real_zeros(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=70)
        samples[6] = 7.0
        result = prepare(samples, fill_missing=False).samples
        for feature, required in [("mean_10", 10), ("std_20", 20), ("slope_30", 30),
                                  ("delta_30", 31), ("acceleration_30", 60)]:
            values = result[f"feature_6_{feature}"]
            self.assertTrue(values.iloc[:required].isna().all())
            self.assertTrue(np.isfinite(values.iloc[required:]).all())
        self.assertTrue((result["feature_6_std_20"].iloc[20:] == 0).all())
        prefix = prepare(samples.iloc[:45], fill_missing=False)
        pd.testing.assert_frame_equal(result.iloc[:45], prefix.samples)

    def test_rolling_windows_exclude_current_cycle_and_require_full_history(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=75)
        samples[6] = samples[1] * 2.0 + 5
        result = prepare(samples).samples
        self.assertEqual(result.shape, (75, 200))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        for window in (10, 20, 30):
            mean = f"feature_6_mean_{window}"
            std = f"feature_6_std_{window}"
            self.assertTrue((result[mean].iloc[:window] == 0).all())
            self.assertTrue((result[std].iloc[:window] == 0).all())
            history = samples[6].iloc[:window]
            self.assertAlmostEqual(result[mean].iloc[window], history.mean())
            self.assertAlmostEqual(result[std].iloc[window], history.std(ddof=0))
        self.assertTrue((result["feature_6_slope_30"].iloc[:30] == 0).all())
        np.testing.assert_allclose(result["feature_6_slope_30"].iloc[30:], 2)
        self.assertTrue((result["feature_6_delta_30"].iloc[:31] == 0).all())
        np.testing.assert_allclose(result["feature_6_delta_30"].iloc[31:], 60)
        self.assertTrue((result["feature_6_acceleration_30"].iloc[:60] == 0).all())
        np.testing.assert_allclose(result["feature_6_acceleration_30"].iloc[60:], 0, atol=1e-10)

    def test_acceleration_compares_two_complete_adjacent_windows(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=61)
        cycles = samples[1].to_numpy()
        samples[6] = np.where(cycles <= 30, cycles, 30 + 3 * (cycles - 30))
        result = prepare(samples).samples
        self.assertTrue((result["feature_6_acceleration_30"].iloc[:60] == 0).all())
        self.assertAlmostEqual(result["feature_6_slope_30"].iloc[60], 3)
        self.assertAlmostEqual(result["feature_6_acceleration_30"].iloc[60], 2)

    def test_future_and_current_values_do_not_change_history_features(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=90)
        original = prepare(samples)
        changed = samples.copy()
        changed.loc[changed[1] >= 65, 6] += 100000
        modified = prepare(changed)
        history_columns = [c for c in original.samples if isinstance(c, str)]
        # Включает cycle 65: его текущее значение тоже не входит в его историю.
        pd.testing.assert_frame_equal(
            original.samples[history_columns].iloc[:65],
            modified.samples[history_columns].iloc[:65],
        )
        prefix = prepare(samples.iloc[:65])
        pd.testing.assert_frame_equal(original.samples.iloc[:65], prefix.samples)

    def test_engines_are_independent_and_input_order_is_preserved(self):
        samples = raw_samples(engine_ids=(1, 2), cycles_per_engine=65)
        mixed = prepare(samples.sample(frac=1, random_state=42))
        separate = prepare(samples.loc[samples[0] == 1])
        pd.testing.assert_frame_equal(mixed.samples.loc[separate.samples.index], separate.samples)
        self.assertTrue(mixed.samples.index.equals(mixed.targets.index))
        for engine_id in (1, 2):
            first_index = samples.index[(samples[0] == engine_id) & (samples[1] == 1)][0]
            self.assertEqual(mixed.samples.loc[first_index, "feature_6_mean_10"], 0)

    def test_targets_are_not_used_and_constant_sensor_has_zero_std_slope_delta(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=70)
        samples[6] = 7.0
        dataset = DatasetEntity(samples, verbose=False)
        DataCleaning(dataset).run(verbose=False)
        dataset.targets[:] = -10000
        DLFeatureEngineering(dataset).run()
        expected = prepare(samples)
        pd.testing.assert_frame_equal(dataset.samples, expected.samples)
        for name in ("feature_6_std_30", "feature_6_slope_30", "feature_6_delta_30",
                     "feature_6_acceleration_30"):
            np.testing.assert_allclose(dataset.samples[name], 0)

    def test_missing_or_duplicate_cycles_and_repeated_engineering_are_rejected(self):
        samples = raw_samples(engine_ids=(1,), cycles_per_engine=65)
        with self.assertRaises(ValueError):
            prepare(samples.drop(index=10))
        with self.assertRaises(ValueError):
            prepare(pd.concat([samples, samples.iloc[[0]]], ignore_index=True))
        dataset = prepare(samples)
        with self.assertRaises(ValueError):
            DLFeatureEngineering(dataset).run()


if __name__ == "__main__":
    unittest.main()
