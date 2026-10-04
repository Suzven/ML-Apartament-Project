import pandas as pd


class SensorHistory:
    def __init__(self, dataset):
        self.dataset = dataset
        self.windows = [10, 30]
        self.initial_window = 25
        self.sensors = [6, 7, 8, 11, 13, 15, 16, 18, 21, 24, 25]

    def run(self):
        samples = self.dataset.samples
        history = pd.DataFrame(index=samples.index)

        for engine_id in self.dataset.engine_ids.unique():
            engine_mask = self.dataset.engine_ids == engine_id
            engine_indices = self.dataset.engine_ids.loc[engine_mask].index
            engine_rows = samples.loc[engine_indices].sort_values(self.dataset.cycle_column)
            engine_history = self.create_engine_history(engine_rows)
            if history.empty:
                history = pd.DataFrame(
                    index=samples.index, columns=engine_history.columns, dtype=float
                )
            history.loc[engine_rows.index, engine_history.columns] = engine_history

        original_samples = samples.rename(columns=str)
        self.dataset.samples = pd.concat([original_samples, history], axis=1)
        print("Sensor history features:", len(history.columns))
        print("Samples with history:", self.dataset.samples.shape)

    def create_engine_history(self, engine_rows):
        history = pd.DataFrame(index=engine_rows.index)
        cycles = engine_rows[self.dataset.cycle_column]

        for sensor in self.sensors:
            values = engine_rows[sensor]
            initial_level = self.initial_level(values)
            history[f"sensor_{sensor}_delta_initial"] = values - initial_level

            for window in self.windows:
                window_features = self.create_window_features(values, cycles, sensor, window)
                history[window_features.columns] = window_features

        return history

    def initial_level(self, values):
        initial_mean = values.expanding().mean()
        if len(initial_mean) > self.initial_window:
            fixed_initial_mean = initial_mean.iloc[self.initial_window - 1]
            initial_mean.iloc[self.initial_window:] = fixed_initial_mean
        return initial_mean

    def create_window_features(self, values, cycles, sensor, window):
        rolling_values = values.rolling(window, min_periods=1)
        rolling_cycles = cycles.rolling(window, min_periods=1)
        value_mean = rolling_values.mean()
        cycle_mean = rolling_cycles.mean()
        cycle_variance = rolling_cycles.var(ddof=0)

        value_cycle_product = values * cycles
        product_mean = value_cycle_product.rolling(window, min_periods=1).mean()
        covariance = product_mean - value_mean * cycle_mean
        slope = covariance / cycle_variance
        slope = slope.fillna(0.0)

        features = pd.DataFrame(index=values.index)
        features[f"sensor_{sensor}_mean_{window}"] = value_mean
        features[f"sensor_{sensor}_std_{window}"] = rolling_values.std(ddof=0)
        features[f"sensor_{sensor}_slope_{window}"] = slope
        return features
