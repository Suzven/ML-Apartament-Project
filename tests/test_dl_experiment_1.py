from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import io
import unittest

import numpy as np
import pandas as pd

from tests.test_regression_workflow import raw_samples

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from scripts import dl_experiment_1 as experiment
    from scripts.entity import DatasetEntity


@unittest.skipUnless(TORCH_AVAILABLE, "Install torch to run DL tests")
class DeepLearningTests(unittest.TestCase):
    def test_split_and_scaler_fit_only_on_training_engines(self):
        with redirect_stdout(io.StringIO()):
            full_train = DatasetEntity(raw_samples(engine_ids=range(1, 101), cycles_per_engine=3))
            test = DatasetEntity(raw_samples(engine_ids=(1,), cycles_per_engine=3))
            datasets, prepared, preprocessing = experiment.prepare_datasets(full_train, test)
        self.assertEqual(set(datasets["TRAIN"].engine_ids), set(range(1, 81)))
        self.assertEqual(set(datasets["VALIDATION"].engine_ids), set(range(81, 101)))
        self.assertEqual(prepared["TRAIN"].shape, (240, 20))
        scaler = preprocessing
        self.assertEqual(scaler.n_samples_seen_, 240)
        np.testing.assert_allclose(scaler.mean_, datasets["TRAIN"].samples.mean().to_numpy())
        np.testing.assert_allclose(prepared["TRAIN"].mean(axis=0), 0, atol=1e-10)

    def test_real_test_targets_include_remaining_cycles(self):
        with redirect_stdout(io.StringIO()):
            train, test = experiment.load_datasets()
        remaining = pd.read_csv(experiment.DATA_DIRECTORY / "RUL_FD001.txt", header=None)[0]
        last_mask = test.cycles == test.cycles.groupby(test.engine_ids).transform("max")
        last_targets = test.targets.loc[last_mask].to_numpy()
        np.testing.assert_array_equal(last_targets, remaining.to_numpy())
        np.testing.assert_array_equal(train.targets + train.cycles,
                                      train.cycles.groupby(train.engine_ids).transform("max"))

    def test_early_stopping_restores_independent_best_weights(self):
        model = experiment.RegressionModel(2)
        stopping = experiment.EarlyStopping(patience=5, min_delta=0.001)
        self.assertFalse(stopping.step(10.0, model, 1))
        best = model.hidden.weight.detach().clone()
        with torch.no_grad():
            model.hidden.weight.add_(10)
        for epoch in range(2, 6):
            self.assertFalse(stopping.step(9.9995, model, epoch))
        self.assertTrue(stopping.step(9.9995, model, 6))
        stopping.restore_best_weights(model)
        torch.testing.assert_close(model.hidden.weight, best)
        self.assertEqual(stopping.best_epoch, 1)
        with self.assertRaises(ValueError):
            stopping.step(float("nan"), model, 7)

    def test_training_reports_mse_weighted_by_samples(self):
        # lr=0 сохраняет predictions: MSE должен быть общим по samples,
        # а не средним двух batch-метрик с размерами 64 и 1.
        from unittest.mock import patch

        X = np.zeros((65, 2))
        targets = np.zeros(65)
        targets[-1] = 100
        model = experiment.RegressionModel(2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        loader = experiment.create_loader(X, targets)
        with patch.object(experiment, "LEARNING_RATE", 0), redirect_stdout(io.StringIO()):
            history, best_epoch = experiment.train_model(model, loader, loader, num_epochs=2)
        np.testing.assert_allclose(history["train_mse"], 10000 / 65)
        np.testing.assert_allclose(history["validation_mse"], 10000 / 65)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(experiment.predict(model, loader).shape, (65,))

    def test_lifetime_boundaries_and_group_rmse(self):
        report = pd.DataFrame({
            "engine_id": [1, 1, 2, 3], "cycle": [1, 100, 1, 1],
            "target": [99, 0, 199, 299], "prediction": [102, 4, 197, 305],
            "residual": [3, 4, -2, 6], "squared_error": [9, 16, 4, 36],
            "engine_lifetime": [100, 100, 200, 300],
        })
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            experiment.print_evaluation("VALIDATION", report, Path(directory))
            groups = pd.read_csv(Path(directory) / "validation_by_lifetime.csv")
            engines = pd.read_csv(Path(directory) / "validation_by_engine.csv")
        self.assertEqual(groups["samples"].tolist(), [2, 1, 1])
        self.assertAlmostEqual(groups["rmse"].iloc[0], np.sqrt(12.5))
        self.assertAlmostEqual(groups["mean_residual"].iloc[0], 3.5)
        self.assertEqual(engines["engine_id"].tolist(), [3, 1, 2])


if __name__ == "__main__":
    unittest.main()
