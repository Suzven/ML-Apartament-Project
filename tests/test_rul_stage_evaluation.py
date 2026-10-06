from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts.analysis.rul_stage_evaluation import RULStageEvaluation
from scripts.entity import DatasetEntity
from tests.test_regression_workflow import raw_samples


class RULStageTests(unittest.TestCase):
    def dataset(self):
        short = raw_samples(engine_ids=(1,), cycles_per_engine=100)
        long = raw_samples(engine_ids=(81,), cycles_per_engine=200)
        samples = pd.concat([short, long], ignore_index=True)
        with redirect_stdout(io.StringIO()):
            return DatasetEntity(samples.sample(frac=1, random_state=7))

    def test_stages_use_each_engines_lifetime_and_keep_all_rows(self):
        dataset = self.dataset()
        with TemporaryDirectory() as directory:
            evaluation = RULStageEvaluation(dataset, dataset.targets, "VALIDATION", directory, features=[6])
            with redirect_stdout(io.StringIO()):
                evaluation.run()
            self.assertEqual(evaluation.summary["Samples"].tolist(), [99, 99, 102])
            self.assertEqual(evaluation.summary["Engines"].tolist(), [2, 2, 2])
            self.assertEqual(evaluation.summary["RMSE"].tolist(), [0, 0, 0])
            self.assertEqual(evaluation.summary["RSS share %"].tolist(), [0, 0, 0])
            self.assertTrue(evaluation.observations.index.equals(dataset.samples.index))
            for engine, cycles, expected in [(1, 33, "early"), (1, 34, "mid"), (1, 66, "mid"), (1, 67, "late"), (81, 66, "early"), (81, 67, "mid"), (81, 200, "late")]:
                row = evaluation.observations.loc[(dataset.engine_ids == engine) & (dataset.cycles == cycles)]
                self.assertEqual(row["stage"].iloc[0], expected)
            self.assertTrue(Path(directory, "validation_metrics.txt").exists())
            self.assertTrue(Path(directory, "validation_features.txt").exists())
            self.assertTrue(Path(directory, "validation_predictions.txt").exists())

    def test_stage_rss_sums_to_overall_and_bias_sign_is_prediction_minus_target(self):
        dataset = self.dataset()
        predictions = dataset.targets + 3
        with TemporaryDirectory() as directory:
            evaluation = RULStageEvaluation(dataset, predictions, "TRAIN", directory, features=[6])
            with redirect_stdout(io.StringIO()):
                evaluation.run()
            np.testing.assert_allclose(evaluation.summary["RMSE"], 3)
            np.testing.assert_allclose(evaluation.summary["Mean Residual"], 3)
            self.assertAlmostEqual(evaluation.summary["RSS"].sum(), 9 * len(dataset.samples))
            self.assertAlmostEqual(evaluation.summary["RSS share %"].sum(), 100)
            self.assertNotIn("life_fraction", dataset.samples.columns)
            self.assertNotIn("stage", dataset.samples.columns)

    def test_invalid_lifetime_is_rejected(self):
        dataset = self.dataset()
        dataset.targets.iloc[0] += 1
        with TemporaryDirectory() as directory:
            evaluation = RULStageEvaluation(dataset, dataset.targets, "TRAIN", directory)
            with self.assertRaises(ValueError):
                evaluation.assign_stages()
