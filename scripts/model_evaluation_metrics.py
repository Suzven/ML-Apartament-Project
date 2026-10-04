import numpy as np
import pandas as pd


class RegressionMetrics:
    def __init__(self, targets, predictions):
        if isinstance(targets, pd.Series) and isinstance(predictions, pd.Series):
            if not targets.index.equals(predictions.index):
                raise ValueError("Targets and predictions must have matching indices")

        self.targets = np.asarray(targets, dtype=float)
        self.predictions = np.asarray(predictions, dtype=float)
        if self.targets.ndim != 1 or self.predictions.ndim != 1:
            raise ValueError("Targets and predictions must be one-dimensional")
        if self.targets.shape != self.predictions.shape or self.targets.size == 0:
            raise ValueError("Targets and predictions must have the same non-zero length")
        if not np.isfinite(self.targets).all() or not np.isfinite(self.predictions).all():
            raise ValueError("Targets and predictions must contain only finite values")

        self.residuals = self.predictions - self.targets

    def rss(self) -> float:
        return float(np.sum(self.residuals ** 2))

    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2)))

    def mae(self) -> float:
        return float(np.mean(np.abs(self.residuals)))

    def rmse_to_std(self) -> float:
        target_std = np.std(self.targets, ddof=0)
        if target_std == 0:
            return float("nan")
        return self.rmse() / target_std

    def rmse_to_min_max(self) -> float:
        target_range = self.targets.max() - self.targets.min()
        if target_range == 0:
            return float("nan")
        return self.rmse() / target_range


class ModelEvaluation:
    def __init__(self, targets, predictions, draw_plots: bool = True, title: str = ""):
        self.metrics = RegressionMetrics(targets, predictions)
        self.draw_plots = draw_plots
        self.title = title

    def run(self):
        print(f"\nMODEL EVALUATION | {self.title}" if self.title else "\nMODEL EVALUATION")
        print(f"RSS: {self.metrics.rss():.6f}")
        print(f"RMSE: {self.metrics.rmse():.6f}")
        print(f"MAE: {self.metrics.mae():.6f}")
        std_ratio = self.metrics.rmse_to_std()
        range_ratio = self.metrics.rmse_to_min_max()
        if np.isnan(std_ratio):
            print("RMSE / STD(target): undefined (constant target)")
        else:
            print(f"RMSE / STD(target): {std_ratio:.6f}")
        if np.isnan(range_ratio):
            print("RMSE / MinMax(target): undefined (constant target)")
        else:
            print(f"RMSE / MinMax(target): {range_ratio:.6f} ({range_ratio:.2%})")

        if self.draw_plots:
            self.prediction_vs_target()
            self.residual_histogram()

    def prediction_vs_target(self):
        import matplotlib.pyplot as plt

        targets = self.metrics.targets
        predictions = self.metrics.predictions
        lower = min(targets.min(), predictions.min())
        upper = max(targets.max(), predictions.max())
        if lower == upper:
            lower -= 0.5
            upper += 0.5

        figure, axes = plt.subplots(figsize=(8, 6))
        axes.scatter(targets, predictions, alpha=0.3, s=10)
        axes.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="Perfect prediction")
        axes.set(
            xlabel="Target (y)",
            ylabel="Prediction (ŷ)",
            title=f"{self.title} | Prediction vs Target" if self.title else "Prediction vs Target",
            xlim=(lower, upper),
            ylim=(lower, upper),
        )
        axes.set_aspect("equal", adjustable="box")
        axes.grid(alpha=0.2)
        axes.legend()
        figure.tight_layout()
        plt.show()
        plt.close(figure)

    def residual_histogram(self):
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(8, 6))
        axes.hist(self.metrics.residuals, bins=30, edgecolor="black", alpha=0.7)
        axes.axvline(0, color="red", linestyle="--", label="Zero residual")
        axes.set(
            xlabel="Residual (ŷ − y)",
            ylabel="Sample count",
            title=f"{self.title} | Residual Histogram" if self.title else "Residual Histogram",
        )
        axes.legend()
        axes.grid(alpha=0.2)
        figure.tight_layout()
        plt.show()
        plt.close(figure)
