"""Признаки истории для DL: только предыдущие циклы внутри одного engine."""

import numpy as np
import pandas as pd


class DLFeatureEngineering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.windows = (10, 20, 30)
        self.trend_window = 30

    def rolling_slope(self, values, cycles):
        # Наклон линейной регрессии sensor = slope * cycle + intercept.
        # NaN сохраняем до конца: неполное окно не должно участвовать в acceleration.
        window = self.trend_window
        cycle_sum = cycles.rolling(window, min_periods=window).sum()
        value_sum = values.rolling(window, min_periods=window).sum()
        product_sum = (cycles * values).rolling(window, min_periods=window).sum()
        square_sum = cycles.pow(2).rolling(window, min_periods=window).sum()
        numerator = window * product_sum - cycle_sum * value_sum
        denominator = window * square_sum - cycle_sum.pow(2)
        return numerator / denominator.where(denominator > 0)

    def create_engine_features(self, samples):
        cycles = samples[self.dataset.cycle_column].astype(float)
        if not cycles.diff().dropna().eq(1).all():
            raise ValueError("History windows require consecutive cycles without duplicates")
        past_cycles = cycles.shift(1)
        features = {}
        for column in samples.columns:
            # Для строки t доступны текущие исходные признаки, но история
            # использует только t-1, t-2, ... . Текущий cycle в окна не входит.
            past_values = samples[column].astype(float).shift(1)
            prefix = f"feature_{column}"
            for window in self.windows:
                rolling = past_values.rolling(window, min_periods=window)
                features[f"{prefix}_mean_{window}"] = rolling.mean()
                features[f"{prefix}_std_{window}"] = rolling.std(ddof=0)

            slope = self.rolling_slope(past_values, past_cycles)
            features[f"{prefix}_slope_30"] = slope
            # Изменение между двумя прошлыми измерениями с расстоянием 30 циклов:
            # x[t-1] - x[t-31]. Нужен 31 предыдущий cycle.
            features[f"{prefix}_delta_30"] = past_values - past_values.shift(30)
            # recent: t-30..t-1; previous: t-60..t-31.
            # Нужны два ПОЛНЫХ соседних окна, то есть 60 предыдущих циклов.
            features[f"{prefix}_acceleration_30"] = slope - slope.shift(30)

        # Только отсутствие полной истории заменяем нулём.
        return pd.DataFrame(features, index=samples.index).fillna(0.0)

    def run(self):
        samples = self.dataset.samples
        if samples.empty or not samples.index.is_unique:
            raise ValueError("Feature engineering requires nonempty samples with unique indices")
        if not samples.index.equals(self.dataset.engine_ids.index):
            raise ValueError("Samples and engine IDs must have matching indices")
        if self.dataset.engine_id_column in samples.columns:
            raise ValueError("Remove engine_id before feature engineering")
        if any(not isinstance(column, int) for column in samples.columns):
            raise ValueError("Expected original columns only; history may already be present")
        if not np.isfinite(samples.to_numpy(dtype=float)).all():
            raise ValueError("Original features must be finite")

        engine_features = []
        for engine_id in self.dataset.engine_ids.unique():
            mask = self.dataset.engine_ids == engine_id
            engine_samples = samples.loc[mask].sort_values(self.dataset.cycle_column)
            engine_features.append(self.create_engine_features(engine_samples))

        engineered = pd.concat(engine_features).reindex(samples.index)
        self.dataset.samples = pd.concat([samples, engineered], axis=1)
