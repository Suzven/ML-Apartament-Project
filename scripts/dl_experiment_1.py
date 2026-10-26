"""Первый DL regression experiment. Запуск: python scripts/dl_experiment_1.py."""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .entity import DatasetEntity
    from .preprocessing import DataCleaning
else:
    from entity import DatasetEntity
    from preprocessing import DataCleaning

DATA_DIRECTORY = Path(__file__).resolve().parent
REPORT_DIRECTORY = DATA_DIRECTORY.parent / "reports" / "dl_experiment_1"
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 5
MIN_DELTA = 0.001


class RegressionModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # nn.Linear сам выполняет default initialization weights и bias.
        self.hidden = nn.Linear(input_features, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return self.output(x)


class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.best_weights = None
        self.best_epoch = None
        self.counter = 0

    def step(self, validation_loss, model, epoch):
        if not np.isfinite(validation_loss):
            raise ValueError("Validation MSE is NaN or Inf")
        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def load_datasets():
    train_samples = pd.read_csv(DATA_DIRECTORY / "train_FD001.txt", sep=r"\s+", header=None)
    test_samples = pd.read_csv(DATA_DIRECTORY / "test_FD001.txt", sep=r"\s+", header=None)
    remaining_cycles = pd.read_csv(DATA_DIRECTORY / "RUL_FD001.txt", sep=r"\s+", header=None)
    if train_samples.shape[1] != 26 or test_samples.shape[1] != 26:
        raise ValueError("Train and test must contain exactly 26 input columns")
    expected_ids = list(range(1, len(remaining_cycles) + 1))
    if remaining_cycles.shape[1] != 1 or sorted(test_samples[0].unique()) != expected_ids:
        raise ValueError("RUL file rows must match test engine IDs 1..N")
    remaining = pd.Series(remaining_cycles[0].to_numpy(), index=expected_ids)
    if not np.isfinite(remaining).all() or (remaining < 0).any():
        raise ValueError("Test RUL must be finite and nonnegative")

    full_train = DatasetEntity(train_samples, verbose=False)
    test = DatasetEntity(test_samples, verbose=False)
    # Test обрывается ДО отказа: к расстоянию до последнего cycle добавляем RUL из файла.
    test.targets = (test.targets + test.engine_ids.map(remaining)).rename("RUL")
    return full_train, test


def prepare_datasets(full_train, test):
    # 80% ДВИГАТЕЛЕЙ, а не случайных строк: engine не пересекаются между train/val.
    engine_ids = np.sort(full_train.engine_ids.unique())
    train_count = int(len(engine_ids) * 0.80)
    if train_count == 0 or train_count == len(engine_ids):
        raise ValueError("At least two engines are required for train/validation split")
    train_ids = engine_ids[:train_count]
    train = full_train.select_rows(full_train.engine_ids.isin(train_ids))
    validation = full_train.select_rows(~full_train.engine_ids.isin(train_ids))
    datasets = {"TRAIN": train, "VALIDATION": validation, "TEST": test}

    for name, dataset in datasets.items():
        print(f"\n{name} | engines: {dataset.engine_ids.nunique()} | samples: {len(dataset.samples)}")
        # Только исходные колонки: удаляем engine_id и прежние constant features.
        removed_features = DataCleaning(dataset).run(verbose=False)
        print("Removed features (engine_id + constants):", removed_features)
        print(dataset.samples.head())

    # Без генерации признаков: fit scaler только на Train.
    preprocessing = StandardScaler()
    prepared = {}
    train_samples = train.samples.rename(columns=str)
    prepared["TRAIN"] = preprocessing.fit_transform(train_samples)
    for name in ("VALIDATION", "TEST"):
        prepared[name] = preprocessing.transform(datasets[name].samples.rename(columns=str))
    for name, values in prepared.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf after preprocessing")
    print(f"\nModel input features: {prepared['TRAIN'].shape[1]}")
    return datasets, prepared, preprocessing


def create_loader(features, targets, shuffle=False):
    X = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(model, train_loader, validation_loader, num_epochs=NUM_EPOCHS):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    early_stopping = EarlyStopping()
    history = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            if not torch.isfinite(loss).item():
                raise ValueError("Training MSE is NaN or Inf")
            loss.backward()
            optimizer.step()
            # Взвешиваем по размеру batch, включая последний неполный batch.
            train_loss_sum += loss.item() * X_batch.size(0)
        train_mse = train_loss_sum / len(train_loader.dataset)

        model.eval()
        validation_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                validation_loss_sum += loss.item() * X_batch.size(0)
        validation_mse = validation_loss_sum / len(validation_loader.dataset)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse})
        print(
            f"Epoch {epoch:02d}/{num_epochs} | Train MSE: {train_mse:.6f} | "
            f"Validation MSE: {validation_mse:.6f}", flush=True,
        )
        if early_stopping.step(validation_mse, model, epoch):
            print(f"Early stopping: {PATIENCE} epochs without improvement > {MIN_DELTA}")
            break

    early_stopping.restore_best_weights(model)
    model.eval()
    print(f"\nRestored epoch {early_stopping.best_epoch} | Validation MSE: {early_stopping.best_loss:.6f}")
    return pd.DataFrame(history), early_stopping.best_epoch


def predict(model, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions.append(model(X_batch).numpy().reshape(-1))
    return np.concatenate(predictions)


def prediction_report(dataset, predictions):
    report = pd.DataFrame({
        "engine_id": dataset.engine_ids.to_numpy(),
        "cycle": dataset.cycles.to_numpy(),
        "target": dataset.targets.to_numpy(),
        "prediction": predictions,
    })
    # Совпадает со знаком residual в существующей ML-версии: prediction - target.
    report["residual"] = report["prediction"] - report["target"]
    report["squared_error"] = report["residual"] ** 2
    report["engine_lifetime"] = report["cycle"] + report["target"]
    return report


def metrics(report):
    return np.sqrt(report["squared_error"].mean()), report["residual"].mean()


def print_evaluation(name, report, report_directory=REPORT_DIRECTORY):
    rmse, mean_residual = metrics(report)
    print(f"\n{name} | samples: {len(report)} | RMSE: {rmse:.6f} | Mean Residual: {mean_residual:.6f}")
    by_engine = report.groupby("engine_id").agg(
        samples=("residual", "size"),
        mse=("squared_error", "mean"),
        mean_residual=("residual", "mean"),
    )
    by_engine["rmse"] = np.sqrt(by_engine["mse"])
    by_engine = by_engine[["samples", "rmse", "mean_residual"]].sort_values("rmse", ascending=False)
    print("\nTop 25 engines by RMSE (descending):")
    print(by_engine.head(25).to_string(float_format="%.6f"))

    # Группируем целые двигатели по полной длине жизни, а не отдельные cycles/RUL.
    lifetime = report["engine_lifetime"]
    groups = [("100-200 cycles [100, 200)", (lifetime >= 100) & (lifetime < 200)),
              ("200-300 cycles [200, 300)", (lifetime >= 200) & (lifetime < 300)),
              ("300+ cycles", lifetime >= 300)]
    group_rows = []
    print("\nGroups by engine lifetime | residual = prediction - target")
    for label, mask in groups:
        subset = report.loc[mask]
        group_rmse, group_residual = metrics(subset)
        group_rows.append({
            "group": label, "engines": subset["engine_id"].nunique(), "samples": len(subset),
            "rmse": group_rmse, "mean_residual": group_residual,
        })
    grouped = pd.DataFrame(group_rows)
    print(grouped.to_string(index=False, float_format="%.6f", na_rep="N/A"))
    report.to_csv(report_directory / f"{name.lower()}_predictions.csv", index=False)
    by_engine.to_csv(report_directory / f"{name.lower()}_by_engine.csv")
    grouped.to_csv(report_directory / f"{name.lower()}_by_lifetime.csv", index=False)


def draw_learning_curves(history, best_epoch, report_directory=REPORT_DIRECTORY, show=True):
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_mse"], label="Train MSE")
    plt.plot(history["epoch"], history["validation_mse"], label="Validation MSE")
    plt.axvline(best_epoch, linestyle="--", color="gray", label=f"Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("dl_experiment_1 | Learning Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(report_directory / "learning_curves.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    torch.manual_seed(RANDOM_SEED)
    full_train, test = load_datasets()
    datasets, prepared, preprocessing = prepare_datasets(full_train, test)
    train_loader = create_loader(prepared["TRAIN"], datasets["TRAIN"].targets, shuffle=True)
    validation_loader = create_loader(prepared["VALIDATION"], datasets["VALIDATION"].targets)
    model = RegressionModel(input_features=prepared["TRAIN"].shape[1])
    history, best_epoch = train_model(model, train_loader, validation_loader)

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    history.to_csv(REPORT_DIRECTORY / "learning_curves.csv", index=False)
    # Для отчётов используем shuffle=False, чтобы сохранить соответствие engine/cycle.
    for name in ("TRAIN", "VALIDATION"):
        loader = create_loader(prepared[name], datasets[name].targets)
        report = prediction_report(datasets[name], predict(model, loader))
        print_evaluation(name, report)

    # Test участвует только в финальной оценке, после выбора лучшей validation epoch.
    test_loader = create_loader(prepared["TEST"], datasets["TEST"].targets)
    test_report = prediction_report(datasets["TEST"], predict(model, test_loader))
    print_evaluation("TEST", test_report)
    last_cycle = test_report.groupby("engine_id")["cycle"].transform("max")
    last_report = test_report.loc[test_report["cycle"] == last_cycle]
    last_report.to_csv(REPORT_DIRECTORY / "test_last_cycle_predictions.csv", index=False)

    torch.save(model.state_dict(), REPORT_DIRECTORY / "model.pth")
    # Сохраняем train-fitted preprocessing для inference с теми же преобразованиями.
    import joblib

    joblib.dump(preprocessing, REPORT_DIRECTORY / "preprocessing.joblib")
    draw_learning_curves(history, best_epoch)
    print("\nReports and best model:", REPORT_DIRECTORY)


if __name__ == "__main__":
    main()
