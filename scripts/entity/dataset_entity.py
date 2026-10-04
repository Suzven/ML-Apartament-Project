from copy import copy

import pandas as pd


class DatasetEntity:
    def __init__(self, df: pd.DataFrame, engine_id_column=0, cycle_column=1):
        if df[[engine_id_column, cycle_column]].isna().any().any():
            raise ValueError("Engine IDs and cycles must not contain missing values")
        if not pd.api.types.is_numeric_dtype(df[cycle_column]):
            raise ValueError("Cycles must be numeric")

        self.samples = df.copy()
        self.engine_id_column = engine_id_column
        self.cycle_column = cycle_column
        self.engine_ids = self.samples[engine_id_column].copy()
        cycle_max = self.samples.groupby(engine_id_column)[cycle_column].transform("max")
        cycle_current = self.samples[cycle_column]
        self.targets = (cycle_max - cycle_current).rename("RUL")
        self.print_head()

    @property
    def cycles(self):
        if self.cycle_column in self.samples.columns:
            return self.samples[self.cycle_column]
        return self.samples[str(self.cycle_column)]

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

    def print_head(self, rows: int = 5):
        print("\nSamples:")
        print(self.samples.head(rows).to_string())
        print("\nTargets:")
        print(self.targets.head(rows).to_string())
        print("\nEngine / current cycle / RUL alignment:")
        preview = self.cycles.head(rows).to_frame(name=self.cycle_column)
        preview.insert(0, self.engine_id_column, self.engine_ids.head(rows))
        preview["RUL"] = self.targets.head(rows)
        print(preview.to_string())
