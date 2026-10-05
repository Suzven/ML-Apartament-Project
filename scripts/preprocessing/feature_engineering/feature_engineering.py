import numpy as np
import pandas as pd

from .sensor_features import SensorFeatures


class FeatureEngineering:
    def __init__(self, dataset):
        self.dataset = dataset
        self.continuous_features = [2, 3, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
        self.sensor_features = SensorFeatures()

    def rare_events(self, values, cycles):
        events = values < 21.605
        last_event_cycle = cycles.where(events).ffill()
        elapsed = (cycles - last_event_cycle).fillna(cycles - cycles.iloc[0])
        return {
            "feature_10_low_fraction_15": events.astype(float).rolling(15, min_periods=1).mean(),
            "feature_10_low_fraction_30": events.astype(float).rolling(30, min_periods=1).mean(),
            "feature_10_cycles_since_low": elapsed,
            "feature_10_low_seen": events.cummax().astype(float),
        }

    def create_engine_features(self, samples):
        cycles = samples[self.dataset.cycle_column].astype(float)
        features = {}
        for feature in self.continuous_features:
            sensor = samples[feature].astype(float)
            features.update(self.sensor_features.create(sensor, cycles, feature))
        features.update(self.rare_events(samples[10], cycles))
        for window in self.sensor_features.windows:
            slope_13 = features[f"feature_13_slope_{window}"]
            slope_18 = features[f"feature_18_slope_{window}"]
            features[f"features_13_18_direction_agreement_{window}"] = np.sign(slope_13) * np.sign(slope_18)
        return pd.DataFrame(features, index=samples.index)

    def run(self):
        samples = self.dataset.samples
        required = self.continuous_features + [10, self.dataset.cycle_column]
        if not samples.index.is_unique:
            raise ValueError("Feature engineering requires unique sample indices")
        if not samples.index.equals(self.dataset.engine_ids.index):
            raise ValueError("Samples and engine IDs must have matching indices")
        if samples[required].isna().any().any() or not np.isfinite(samples[required].to_numpy()).all():
            raise ValueError("Sensor values and cycles must be finite")
        if "feature_2_initial_mean" in samples.columns:
            raise ValueError("Feature engineering has already been applied")

        engine_features = []
        for engine_id in self.dataset.engine_ids.unique():
            engine_mask = self.dataset.engine_ids == engine_id
            engine_samples = samples.loc[engine_mask].sort_values(self.dataset.cycle_column)
            if engine_samples[self.dataset.cycle_column].duplicated().any():
                raise ValueError(f"Duplicate cycles for engine {engine_id}")
            engine_features.append(self.create_engine_features(engine_samples))
        if not engine_features:
            raise ValueError("Feature engineering requires a nonempty dataset")

        engineered = pd.concat(engine_features).reindex(samples.index)
        self.dataset.samples = pd.concat([samples, engineered], axis=1)
        print("\nFeature engineering | original features:", samples.shape[1])
        print("Feature engineering | added features:", engineered.shape[1])
        print("Feature engineering | total features:", self.dataset.samples.shape[1])
        self.dataset.print_head(rows=3)
