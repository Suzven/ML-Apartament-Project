import numpy as np
import pandas as pd


class SensorFeatures:
    def __init__(self, windows=(15, 30), baseline_window=15):
        self.windows = windows
        self.baseline_window = baseline_window

    def rolling_slope(self, values, cycles, window):
        rolling_cycles = cycles.rolling(window, min_periods=1)
        rolling_values = values.rolling(window, min_periods=1)
        count = rolling_values.count()
        cycle_sum = rolling_cycles.sum()
        value_sum = rolling_values.sum()
        product_sum = (cycles * values).rolling(window, min_periods=1).sum()
        square_sum = cycles.pow(2).rolling(window, min_periods=1).sum()
        numerator = count * product_sum - cycle_sum * value_sum
        denominator = count * square_sum - cycle_sum.pow(2)
        return numerator.div(denominator.where(denominator > 0)).fillna(0)

    def cycles_since_extremum(self, values, cycles, window, minimum):
        ages = pd.Series(0.0, index=values.index)
        for position in range(len(values)):
            start = max(0, position - window + 1)
            history = values.iloc[start:position + 1].to_numpy()
            extreme = history.min() if minimum else history.max()
            last_match = np.flatnonzero(history == extreme)[-1]
            extreme_position = start + last_match
            ages.iloc[position] = cycles.iloc[position] - cycles.iloc[extreme_position]
        return ages

    def create(self, values, cycles, feature):
        features = {}
        baseline_mean = values.expanding().mean()
        baseline_std = values.expanding().std(ddof=0)
        if len(values) > self.baseline_window:
            baseline_mean.iloc[self.baseline_window:] = baseline_mean.iloc[self.baseline_window - 1]
            baseline_std.iloc[self.baseline_window:] = baseline_std.iloc[self.baseline_window - 1]

        delta = values - baseline_mean
        outside_baseline = delta.abs() > 2 * baseline_std
        prefix = f"feature_{feature}"
        features[f"{prefix}_initial_mean"] = baseline_mean
        features[f"{prefix}_delta_initial"] = delta
        features[f"{prefix}_outside_initial_fraction_30"] = outside_baseline.astype(float).rolling(30, min_periods=1).mean()

        for window in self.windows:
            rolling = values.rolling(window, min_periods=1)
            minimum = rolling.min()
            maximum = rolling.max()
            value_range = maximum - minimum
            slope = self.rolling_slope(values, cycles, window)
            suffix = f"_{window}"
            features[f"{prefix}_mean{suffix}"] = rolling.mean()
            features[f"{prefix}_std{suffix}"] = rolling.std(ddof=0)
            features[f"{prefix}_slope{suffix}"] = slope
            features[f"{prefix}_slope_change{suffix}"] = (slope - slope.shift(window)).fillna(0)
            features[f"{prefix}_distance_min{suffix}"] = values - minimum
            features[f"{prefix}_distance_max{suffix}"] = maximum - values
            features[f"{prefix}_range{suffix}"] = value_range
            features[f"{prefix}_range_position{suffix}"] = (values - minimum).div(value_range.where(value_range > 0)).fillna(0)
            features[f"{prefix}_cycles_since_min{suffix}"] = self.cycles_since_extremum(values, cycles, window, minimum=True)
            features[f"{prefix}_cycles_since_max{suffix}"] = self.cycles_since_extremum(values, cycles, window, minimum=False)
        return features
