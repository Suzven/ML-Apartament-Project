from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import pandas as pd

from scripts.classification_evaluation_metrics import ClassificationMetrics
from scripts.entity import ClassificationDatasetEntity
from scripts.models import LogisticRegression
from scripts.preprocessing import DatasetPreprocessing
from tests.test_regression_workflow import raw_samples


class ClassificationTests(unittest.TestCase):
    def test_fraction_and_class_boundaries(self):
        with redirect_stdout(io.StringIO()):
            dataset = ClassificationDatasetEntity(raw_samples(engine_ids=(1,), cycles_per_engine=100))
        self.assertAlmostEqual(dataset.targets.iloc[0], 0.99)
        self.assertEqual(dataset.targets.iloc[-1], 0)
        expected = {25: "medium", 50: "low", 75: "near_failure", 95: "near_failure", 100: "near_failure"}
        for cycle, label in expected.items():
            self.assertEqual(dataset.class_targets().loc[dataset.cycles == cycle].iloc[0], label)

    def test_pipeline_keeps_fraction_out_of_inputs_and_fits_classifier(self):
        with redirect_stdout(io.StringIO()):
            dataset = ClassificationDatasetEntity(raw_samples(cycles_per_engine=25))
            preprocessing = DatasetPreprocessing(dataset)
            preprocessing.run()
        train, validation = preprocessing.train, preprocessing.validation
        self.assertEqual(train.samples.shape[1], 394)
        self.assertNotIn("remaining_fraction", train.samples.columns)
        self.assertNotIn("resource_class", train.samples.columns)
        self.assertTrue(train.targets.index.equals(train.samples.index))
        model = LogisticRegression()
        model.fit(train.samples, train.class_targets())
        probabilities = model.predict_proba(validation.samples)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        metrics = ClassificationMetrics(validation.class_targets(), model.predict(validation.samples), probabilities, model.classes)
        self.assertTrue(0 <= metrics.accuracy() <= 1)
        self.assertTrue(0 <= metrics.precision() <= 1)
        self.assertTrue(np.isfinite(metrics.log_loss()))

    def test_known_metrics(self):
        metrics = ClassificationMetrics(['a', 'b'], ['a', 'a'], [[0.8, 0.2], [0.6, 0.4]], ['a', 'b'])
        self.assertEqual(metrics.accuracy(), 0.5)
        self.assertEqual(metrics.precision(), 0.25)
        self.assertAlmostEqual(metrics.log_loss(), -(np.log(0.8) + np.log(0.4)) / 2)

    def test_duplicated_file_targets_are_correct(self):
        data = pd.read_csv('scripts/train_FD001_classification.txt', sep=r'\s+', header=None)
        original = pd.read_csv('scripts/train_FD001.txt', sep=r'\s+', header=None)
        pd.testing.assert_frame_equal(data.iloc[:, :26], original)
        maximum = original.groupby(0)[1].transform('max')
        np.testing.assert_allclose(data[26], (maximum - original[1]) / maximum, atol=1e-12)
