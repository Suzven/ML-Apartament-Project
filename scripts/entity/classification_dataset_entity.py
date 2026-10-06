from copy import copy

import numpy as np
import pandas as pd


class ClassificationDatasetEntity:
    resource_groups = ["high", "medium", "low", "near_failure"]

    def __init__(self, df, engine_id_column=0, cycle_column=1, remaining_cycles_by_engine=None):
        self.samples = df.copy()
        self.engine_id_column = engine_id_column
        self.cycle_column = cycle_column
        self.engine_ids = self.samples[engine_id_column].copy()
        cycles = self.samples[cycle_column]
        if self.engine_ids.isna().any() or not np.isfinite(cycles).all() or (cycles <= 0).any():
            raise ValueError("Engine IDs must be present and cycles must be positive and finite")
        cycle_max = cycles.groupby(self.engine_ids).transform("max")
        if remaining_cycles_by_engine is not None:
            remaining = self.engine_ids.map(remaining_cycles_by_engine)
            if remaining.isna().any() or not np.isfinite(remaining).all() or (remaining < 0).any():
                raise ValueError("Each test engine requires finite nonnegative remaining cycles")
            cycle_max = cycle_max + remaining
        self.targets = ((cycle_max - cycles) / cycle_max).rename("remaining_fraction")
        self.print_head()

    @property
    def cycles(self):
        return self.samples[self.cycle_column]

    def class_targets(self):
        return pd.cut(
            self.targets,
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=["near_failure", "low", "medium", "high"],
            include_lowest=True,
            right=True,
        ).astype(str).rename("resource_class")

    def copy(self):
        dataset = copy(self)
        dataset.samples = self.samples.copy()
        dataset.targets = self.targets.copy()
        dataset.engine_ids = self.engine_ids.copy()
        return dataset

    def select_rows(self, mask):
        dataset = copy(self)
        dataset.samples = self.samples.loc[mask].copy()
        dataset.targets = self.targets.loc[mask].copy()
        dataset.engine_ids = self.engine_ids.loc[mask].copy()
        return dataset

    def print_head(self, rows=5):
        print("\nSamples:")
        print(self.samples.head(rows).to_string())
        preview = self.cycles.head(rows).to_frame(name="cycle")
        preview.insert(0, "engine_id", self.engine_ids.head(rows))
        preview["remaining_fraction"] = self.targets.head(rows)
        preview["resource_class"] = self.class_targets().head(rows)
        print("\nEngine / cycle / fraction / class alignment:")
        print(preview.to_string())
