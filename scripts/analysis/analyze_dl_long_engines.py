"""Разбор длинных engines без переобучения и без использования Test.

Запуск: python -m scripts.analysis.analyze_dl_long_engines
"""

from pathlib import Path
import hashlib
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..dl_experiment_2 import RegressionModel
from ..entity import DatasetEntity
from ..preprocessing import DataCleaning
from ..preprocessing.dl_feature_engineering import DLFeatureEngineering

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports" / "dl_experiment_2"
OUTPUT = RUN / "long_engine_diagnostics"
ENGINES = (69, 67, 92, 96)
SENSORS = (6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25)
COLORS = {69: "tab:blue", 67: "tab:orange", 92: "tab:green", 96: "tab:red"}


def load_run():
    preprocessing = joblib.load(RUN / "preprocessing.joblib")
    if hasattr(preprocessing, "named_steps"):
        scaler = preprocessing.named_steps["scaling"]
        fill_missing = False
    else:
        scaler = preprocessing
        fill_missing = True
    samples = pd.read_csv(ROOT / "scripts" / "train_FD001.txt", sep=r"\s+", header=None)
    dataset = DatasetEntity(samples, verbose=False)
    datasets = {
        "TRAIN": dataset.select_rows(dataset.engine_ids <= 80),
        "VALIDATION": dataset.select_rows(dataset.engine_ids > 80),
    }
    for subset in datasets.values():
        DataCleaning(subset).run(verbose=False)
        DLFeatureEngineering(subset, fill_missing=fill_missing).run()
    reports = []
    for split, filename in [("TRAIN", "train_predictions.csv"),
                            ("VALIDATION", "validation_predictions.csv")]:
        report = pd.read_csv(RUN / filename)
        subset = datasets[split]
        np.testing.assert_array_equal(report["engine_id"], subset.engine_ids)
        np.testing.assert_array_equal(report["cycle"], subset.cycles)
        np.testing.assert_array_equal(report["target"], subset.targets)
        np.testing.assert_allclose(report["residual"], report["prediction"] - report["target"], atol=1e-4)
        reports.append(report.assign(split=split))

    # Проверяем, что checkpoint, scaler и CSV относятся к одному запуску.
    model = RegressionModel(len(scaler.feature_names_in_))
    model.load_state_dict(torch.load(RUN / "model.pth", map_location="cpu", weights_only=True))
    model.eval()
    for split, report in zip(datasets, reports):
        features = preprocessing.transform(datasets[split].samples.rename(columns=str))
        with torch.no_grad():
            predictions = model(torch.tensor(features, dtype=torch.float32)).numpy().reshape(-1)
        np.testing.assert_allclose(predictions, report["prediction"], atol=1e-3, rtol=1e-5)
    metadata = {
        "model_sha256": hashlib.sha256((RUN / "model.pth").read_bytes()).hexdigest(),
        "model_architecture": str(model),
        "checkpoint_predictions_verified": True,
        "data_used": ["Train engines 1-80", "Validation engines 81-100"],
        "test_used": False,
        "candidate_associations": "Exploratory, Train only, 80 engines at cycle 100",
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return datasets, pd.concat(reports, ignore_index=True), scaler


def error_tables(reports):
    summaries = []
    periods = []
    for engine_id in ENGINES:
        rows = reports.loc[reports["engine_id"] == engine_id].sort_values("cycle")
        total_rss = rows["squared_error"].sum()
        early = rows.loc[rows["cycle"] <= 100]
        late = rows.loc[rows["cycle"] > 200]
        summaries.append({
            "engine_id": engine_id, "split": rows["split"].iloc[0],
            "lifetime": rows["engine_lifetime"].iloc[0],
            "rmse": np.sqrt(rows["squared_error"].mean()),
            "mean_residual": rows["residual"].mean(),
            "first100_rmse": np.sqrt(early["squared_error"].mean()),
            "first100_rss_percent": 100 * early["squared_error"].sum() / total_rss,
            "after200_rmse": np.sqrt(late["squared_error"].mean()),
            "after200_mean_residual": late["residual"].mean(),
        })
        for start, end in [(1, 30), (31, 60), (61, 100), (101, 200), (201, 300), (301, 10000)]:
            subset = rows.loc[rows["cycle"].between(start, end)]
            if not subset.empty:
                periods.append({
                    "engine_id": engine_id, "start_cycle": start,
                    "end_cycle": int(subset["cycle"].max()), "samples": len(subset),
                    "rmse": np.sqrt(subset["squared_error"].mean()),
                    "mean_residual": subset["residual"].mean(),
                    "rss_percent": 100 * subset["squared_error"].sum() / total_rss,
                })
    pd.DataFrame(periods).to_csv(OUTPUT / "errors_by_period.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "engine_summary.csv", index=False)
    reports.loc[reports["engine_id"].isin(ENGINES)].to_csv(OUTPUT / "focus_predictions.csv", index=False)
    return summary


def scaling_table(train, scaler):
    rows = []
    for column in SENSORS:
        for window in (10, 20, 30):
            name = f"feature_{column}_mean_{window}"
            valid = train.samples.loc[train.cycles > window, name]
            scale = scaler.scale_[list(scaler.feature_names_in_).index(name)]
            real_std = valid.std(ddof=0)
            rows.append({
                "feature": column, "window": window, "scaler_std": scale,
                "complete_window_std": real_std,
                "scale_ratio": scale / real_std,
                "missing_history_rows": int((train.cycles <= window).sum()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "rolling_mean_scaling.csv", index=False)
    return result


def snapshot_features(datasets, cycle=100):
    """Одна строка на engine при одинаковом возрасте; все Train engines живы на cycle 100.

    Lifetime используется только как ответ для ретроспективного анализа.
    Кандидатные признаки используют исключительно cycles < 100.
    """
    records = []
    for split, dataset in datasets.items():
        for engine_id in dataset.engine_ids.unique():
            rows = dataset.samples.loc[dataset.engine_ids == engine_id].sort_values(1)
            past = rows.loc[rows[1] < cycle]
            current = rows.loc[rows[1] == cycle]
            if len(past) != cycle - 1 or current.empty:
                raise ValueError("Matched-cycle comparison requires complete history for every engine")
            record = {"engine_id": engine_id, "split": split,
                      "lifetime": float(dataset.targets.loc[rows.index[0]] + rows[1].iloc[0])}
            for column in SENSORS:
                values = past[column].to_numpy()
                cycles = past[1].to_numpy()
                baseline = values[:30].mean()
                recent = values[-30:].mean()
                record[f"current_{column}"] = current[column].iloc[0]
                record[f"mean30_{column}"] = recent
                record[f"slope30_{column}"] = np.polyfit(cycles[-30:], values[-30:], 1)[0]
                record[f"baseline30_{column}"] = baseline
                record[f"delta_baseline_{column}"] = recent - baseline
                record[f"slope90_{column}"] = np.polyfit(cycles[-90:], values[-90:], 1)[0]
                record[f"mean_deviation_{column}"] = (values[30:] - baseline).mean()
            records.append(record)
    snapshot = pd.DataFrame(records).set_index("engine_id")
    snapshot.to_csv(OUTPUT / "snapshot_cycle100.csv")
    train = snapshot.loc[snapshot["split"] == "TRAIN"]
    feature_names = [c for c in snapshot.columns if c not in ("split", "lifetime")]
    correlations = train[feature_names].corrwith(train["lifetime"], method="spearman")
    associations = correlations.rename("spearman_lifetime").to_frame()
    associations["train_engines"] = len(train)
    associations = associations.sort_values("spearman_lifetime", key=lambda s: s.abs(), ascending=False)
    associations.to_csv(OUTPUT / "train_only_candidate_associations.csv")

    # Сходство при одинаковом возрасте: 14 текущих сенсоров + их mean30.
    # Масштабы для расстояний оцениваем по 80 Train engines, без Validation.
    comparison = [f"{prefix}_{c}" for prefix in ("current", "mean30") for c in SENSORS]
    scale = train[comparison].std(ddof=0)
    neighbor_rows = []
    for engine_id in ENGINES:
        difference = (train[comparison] - snapshot.loc[engine_id, comparison]) / scale
        distance = np.sqrt(difference.pow(2).mean(axis=1).astype(float))
        nearest = distance.drop(engine_id, errors="ignore").sort_values().head(5)
        for other, value in nearest.items():
            neighbor_rows.append({
                "focus_engine": engine_id, "train_neighbor": other,
                "standardized_rms_distance": value,
                "neighbor_lifetime": train.loc[other, "lifetime"],
                "focus_lifetime": snapshot.loc[engine_id, "lifetime"],
            })
    neighbors = pd.DataFrame(neighbor_rows)
    neighbors.to_csv(OUTPUT / "neighbors_cycle100.csv", index=False)
    return snapshot, associations, neighbors


def plots(reports, snapshot):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, engine_id in zip(axes.flat, ENGINES):
        rows = reports.loc[reports["engine_id"] == engine_id]
        axis.axvspan(1, 100, color="tab:orange", alpha=0.12, label="First 100 cycles")
        axis.plot(rows["cycle"], rows["target"], color="black", linewidth=2, label="True RUL")
        axis.plot(rows["cycle"], rows["prediction"], color=COLORS[engine_id], label="Predicted RUL")
        axis.set_title(f"Engine {engine_id} | {rows['split'].iloc[0]} | lifetime {int(rows['engine_lifetime'].iloc[0])}")
        axis.set(xlabel="Observed cycle", ylabel="RUL (cycles)")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Stress-test checkpoint: most underprediction occurs early")
    figure.tight_layout()
    figure.savefig(OUTPUT / "rul_trajectories.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    train = snapshot.loc[snapshot["split"] == "TRAIN"]
    for axis, feature in zip(axes, ("delta_baseline_16", "slope90_24")):
        axis.scatter(train[feature], train["lifetime"], color="gray", alpha=0.6, label="80 Train engines")
        for engine_id in ENGINES:
            row = snapshot.loc[engine_id]
            axis.scatter(row[feature], row["lifetime"], color=COLORS[engine_id], s=65,
                         marker="o" if row["split"] == "TRAIN" else "^", zorder=3)
            axis.annotate(str(engine_id), (row[feature], row["lifetime"]), xytext=(5, 4), textcoords="offset points")
        axis.set(xlabel=feature, ylabel="True lifetime (label only)")
        axis.margins(x=0.15, y=0.1)
        axis.grid(alpha=0.2)
    figure.suptitle("Candidate signals at cycle 100: history uses cycles 1-99 only")
    figure.tight_layout()
    figure.savefig(OUTPUT / "candidate_signals.png", dpi=160)
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets, reports, scaler = load_run()
    summary = error_tables(reports)
    scales = scaling_table(datasets["TRAIN"], scaler)
    snapshot, associations, neighbors = snapshot_features(datasets)
    plots(reports, snapshot)
    print("Verified saved model reproduces Train/Validation predictions; Test was not loaded.")
    print("\nFocus engines:\n", summary.round(3).to_string(index=False))
    print("\nRolling mean 30 scaling:\n", scales.loc[scales["window"] == 30].round(3).to_string(index=False))
    print("\nTrain-only candidate associations at cycle 100:\n", associations.head(12).round(3).to_string())
    print("\nNearest Train trajectories at cycle 100:\n", neighbors.round(3).to_string(index=False))
    print("\nDiagnostics:", OUTPUT)


if __name__ == "__main__":
    main()
