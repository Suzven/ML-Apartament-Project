from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "reports" / "engine_trajectory_analysis"
ENGINES = [92, 96, 90, 100]
FEATURES = [6, 7, 8, 11, 13, 15, 16, 18, 21, 24, 25]
COLORS = {92: "#d62728", 96: "#ff7f0e", 90: "#1f77b4", 100: "#2ca02c"}


def load_predictions(samples):
    path = OUTPUT.parent / "polynomial_ridge_by_engine" / "validation_predictions.txt"
    columns = ["sample_index", "engine_id", "cycle", "target", "prediction", "residual"]
    predictions = pd.read_csv(path, sep=r"\s+", skiprows=1, names=columns)
    predictions = predictions.set_index("sample_index")
    selected = predictions.loc[predictions["engine_id"].isin(ENGINES)]

    expected_ids = samples.loc[selected.index, 0]
    expected_cycles = samples.loc[selected.index, 1]
    maximum_cycles = samples.groupby(0)[1].transform("max")
    expected_targets = maximum_cycles - samples[1]
    np.testing.assert_array_equal(selected["engine_id"], expected_ids)
    np.testing.assert_array_equal(selected["cycle"], expected_cycles)
    np.testing.assert_array_equal(selected["target"], expected_targets.loc[selected.index])
    residuals = selected["prediction"] - selected["target"]
    np.testing.assert_allclose(residuals, selected["residual"], atol=2e-6)
    return selected


def plot_rul_trajectories(predictions):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    summary = []

    for engine_id, axis in zip(ENGINES, axes.flat):
        rows = predictions.loc[predictions["engine_id"] == engine_id].sort_values("cycle")
        cycles = rows["cycle"]
        targets = rows["target"]
        predicted_rul = rows["prediction"]
        residuals = rows["residual"]
        smoothed = predicted_rul.rolling(15, min_periods=1).mean()

        axis.plot(cycles, targets, color="black", linestyle="--", label="Actual RUL")
        axis.plot(cycles, predicted_rul, color=COLORS[engine_id], alpha=0.35, label="Prediction")
        axis.plot(
            cycles, smoothed, color=COLORS[engine_id],
            label="Prediction: trailing mean (15)",
        )
        axis.set(
            title=f"Engine {engine_id} | lifetime {len(rows)}",
            xlabel="Cycle", ylabel="RUL (cycles)",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

        summary.append({
            "engine": engine_id,
            "lifetime": len(rows),
            "RMSE": np.sqrt(np.mean(residuals ** 2)),
            "MAE": residuals.abs().mean(),
            "bias": residuals.mean(),
            "first25_target": targets.head(25).mean(),
            "first25_prediction": predicted_rul.head(25).mean(),
            "first25_bias": residuals.head(25).mean(),
            "last30_bias": residuals.tail(30).mean(),
        })

    figure.suptitle("PolynomialRidge degree 3 | held-out engines", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT / "rul_trajectories.png", dpi=160)
    plt.close(figure)
    return pd.DataFrame(summary)


def sensor_metrics(rows, feature, engine_id, initial_mean, initial_std, train_std):
    values = rows[feature]
    initial_median = values.head(25).median()
    early_rows = rows.head(50)
    late_rows = rows.tail(50)
    early_slope = np.polyfit(early_rows[1], early_rows[feature], 1)[0]
    late_slope = np.polyfit(late_rows[1], late_rows[feature], 1)[0]
    initial_difference = initial_median - initial_mean[feature]

    return {
        "feature": feature,
        "engine": engine_id,
        "first25_median": initial_median,
        "initial_z_vs_train_engines": initial_difference / initial_std[feature],
        "last25_median": values.tail(25).median(),
        "early_slope_per100_train_std": early_slope * 100 / train_std[feature],
        "late_slope_per100_train_std": late_slope * 100 / train_std[feature],
        "cycle_correlation": rows[1].corr(values),
    }


def plot_sensor_trajectories(samples):
    train = samples.loc[samples[0].between(1, 80)]
    train_mean = train[FEATURES].mean()
    train_std = train[FEATURES].std(ddof=0)
    train_initial = train.groupby(0).head(25).groupby(0)[FEATURES].median()
    initial_mean = train_initial.mean()
    initial_std = train_initial.std(ddof=0)
    summary = []
    figure, axes = plt.subplots(4, 3, figsize=(15, 14))

    for feature, axis in zip(FEATURES, axes.flat):
        for engine_id in ENGINES:
            rows = samples.loc[samples[0] == engine_id].sort_values(1)
            smoothed = rows[feature].rolling(15, min_periods=15).mean()
            standardized = (smoothed - train_mean[feature]) / train_std[feature]
            axis.plot(rows[1], standardized, color=COLORS[engine_id], label=str(engine_id))
            summary.append(sensor_metrics(
                rows, feature, engine_id, initial_mean, initial_std, train_std
            ))

        axis.set(title=f"Feature {feature}", xlabel="Cycle", ylabel="Sensor (train-standardized)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)

    axes.flat[-1].axis("off")
    figure.suptitle("Sensor trajectories | trailing mean (15 cycles) | shared train scaling", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT / "sensor_trajectories.png", dpi=160)
    plt.close(figure)
    return pd.DataFrame(summary)


def fit_retrospective_bend(rows, engine_id):
    cycles = rows[1].to_numpy(dtype=float)
    values = rows[15].to_numpy(dtype=float)
    intercept_column = np.ones(len(cycles))
    linear_design = np.column_stack([intercept_column, cycles])
    linear_coefficients = np.linalg.lstsq(linear_design, values, rcond=None)[0]
    linear_prediction = linear_design @ linear_coefficients
    linear_squared_error = np.sum((values - linear_prediction) ** 2)

    best_squared_error = float("inf")
    best_knot = None
    best_coefficients = None
    for knot in cycles[25:-25]:
        cycles_after_knot = np.maximum(0, cycles - knot)
        design = np.column_stack([intercept_column, cycles, cycles_after_knot])
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        prediction = design @ coefficients
        squared_error = np.sum((values - prediction) ** 2)
        if squared_error < best_squared_error:
            best_squared_error = squared_error
            best_knot = knot
            best_coefficients = coefficients

    if best_coefficients is None:
        raise ValueError("At least 51 observations are required for a retrospective bend")

    return {
        "engine": engine_id,
        "feature": 15,
        "retrospective_bend_cycle": best_knot,
        "slope_before": best_coefficients[1],
        "slope_after": best_coefficients[1] + best_coefficients[2],
        "SSE_reduction_vs_line_percent": (1 - best_squared_error / linear_squared_error) * 100,
    }


def retrospective_bends(samples):
    summary = []
    for engine_id in ENGINES:
        rows = samples.loc[samples[0] == engine_id].sort_values(1)
        summary.append(fit_retrospective_bend(rows, engine_id))
    return pd.DataFrame(summary)


def save_analysis(trajectory_summary, sensor_summary, bend_summary):
    notes = (
        "Model: saved PolynomialRidge degree=3, alpha=100 predictions; validation engines only.\n"
        "First 25 cycles: initial level. First/last 50: linear slope.\n"
        "Initial z: compared with first-25 medians of the 80 train engines.\n"
        "Sensor plots: trailing 15-cycle means, scaled with train-only mean/std.\n"
        "Bend: retrospective continuous two-segment fit to feature 15; not a labeled degradation onset.\n"
        "Full-life endpoints and bend locations are diagnostic only, not inference features.\n\n"
    )
    text = notes + trajectory_summary.to_string(index=False)
    text += "\n\nSENSOR COMPARISON\n" + sensor_summary.to_string(index=False)
    text += "\n\nRETROSPECTIVE TREND BENDS\n" + bend_summary.to_string(index=False)
    (OUTPUT / "trajectory_analysis.txt").write_text(text + "\n", encoding="utf-8")


def main():
    samples = pd.read_csv(ROOT / "train_FD001.txt", sep=r"\s+", header=None)
    predictions = load_predictions(samples)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trajectory_summary = plot_rul_trajectories(predictions)
    sensor_summary = plot_sensor_trajectories(samples)
    bend_summary = retrospective_bends(samples)
    save_analysis(trajectory_summary, sensor_summary, bend_summary)

    print(trajectory_summary.to_string(index=False))
    print(bend_summary.to_string(index=False))
    print("\nInitial standardized differences vs train engines:")
    initial_differences = sensor_summary.pivot(
        index="feature", columns="engine", values="initial_z_vs_train_engines"
    )
    print(initial_differences.round(2).to_string())
    print("\nEarly/late slope, train std per 100 cycles:")
    selected_sensors = sensor_summary["feature"].isin([13, 15, 16, 18])
    print(sensor_summary.loc[selected_sensors].to_string(index=False))
    print("\nReports:", OUTPUT)


if __name__ == "__main__":
    main()
