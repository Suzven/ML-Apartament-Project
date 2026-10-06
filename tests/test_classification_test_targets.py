from contextlib import redirect_stdout
import io
import unittest

import pandas as pd

from scripts.entity import ClassificationDatasetEntity
from scripts.dataset_analyze_classification_test import load_datasets
from tests.test_regression_workflow import raw_samples


class TestTargetsTests(unittest.TestCase):
    def test_test_fraction_uses_rul_at_observed_endpoint(self):
        with redirect_stdout(io.StringIO()):
            dataset = ClassificationDatasetEntity(raw_samples(engine_ids=(1,), cycles_per_engine=10), remaining_cycles_by_engine=pd.Series({1: 5}))
        self.assertAlmostEqual(dataset.targets.iloc[0], 14 / 15)
        self.assertAlmostEqual(dataset.targets.iloc[-1], 5 / 15)
        self.assertEqual(dataset.class_targets().iloc[-1], 'low')
        self.assertNotIn('remaining_fraction', dataset.samples.columns)

    def test_missing_test_engine_label_is_rejected(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                ClassificationDatasetEntity(raw_samples(engine_ids=(1, 2)), remaining_cycles_by_engine=pd.Series({1: 5}))

    def test_real_test_endpoints_reproduce_rul_file(self):
        with redirect_stdout(io.StringIO()):
            train, test = load_datasets()
        self.assertEqual(train.engine_ids.nunique(), 100)
        self.assertEqual(test.engine_ids.nunique(), 100)
        expected = pd.read_csv('scripts/RUL_FD001.txt', header=None)[0]
        for engine in (1, 50, 100):
            mask = test.engine_ids == engine
            last_cycle = test.cycles.loc[mask].max()
            last_fraction = test.targets.loc[mask & (test.cycles == last_cycle)].iloc[0]
            self.assertAlmostEqual(last_fraction, expected.iloc[engine - 1] / (last_cycle + expected.iloc[engine - 1]))
