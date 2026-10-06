from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analysis.grouped_classification_evaluation import GroupedClassificationEvaluation
from scripts.entity import ClassificationDatasetEntity
from tests.test_regression_workflow import raw_samples


class GroupedClassificationTests(unittest.TestCase):
    def test_precision_uses_false_positives_from_other_groups(self):
        with redirect_stdout(io.StringIO()):
            dataset = ClassificationDatasetEntity(raw_samples(engine_ids=(1,), cycles_per_engine=100))
        targets = dataset.class_targets()
        predictions = targets.to_numpy().copy()
        high_index = np.flatnonzero(targets.to_numpy() == 'high')[0]
        predictions[high_index] = 'near_failure'
        classes = np.array(sorted(targets.unique()))
        probabilities = np.full((len(targets), len(classes)), 0.01)
        for index, group in enumerate(predictions):
            probabilities[index, np.flatnonzero(classes == group)[0]] = 1 - 0.01 * (len(classes) - 1)
        with TemporaryDirectory() as directory:
            evaluation = GroupedClassificationEvaluation(dataset, predictions, probabilities, classes, 'VALIDATION', directory)
            with redirect_stdout(io.StringIO()):
                evaluation.run()
            summary = evaluation.summary.set_index('Group')
            self.assertEqual(summary.loc['near_failure', 'Samples'], 26)
            self.assertEqual(summary.loc['near_failure', 'Recall'], 1)
            self.assertAlmostEqual(summary.loc['near_failure', 'Precision'], 26 / 27)
            self.assertEqual(evaluation.confusion.loc['high', 'near_failure'], 1)
            self.assertEqual(evaluation.confusion.to_numpy().sum(), 100)
            for group in evaluation.groups:
                self.assertEqual(summary.loc[group, 'Accuracy within group'], summary.loc[group, 'Recall'])
            loss = -(np.log(probabilities[np.arange(len(targets)), [list(classes).index(group) for group in targets]])).mean()
            self.assertAlmostEqual((summary['Log Loss'] * summary['Samples']).sum() / 100, loss)
            self.assertTrue(Path(directory, 'validation_confusion.txt').exists())
