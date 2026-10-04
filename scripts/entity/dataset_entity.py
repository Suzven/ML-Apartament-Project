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
        cycle_max = self.samples.groupby(engine_id_column)[cycle_column].transform("max")
        cycle_current = self.samples[cycle_column]
        self.targets = (cycle_max - cycle_current).rename("RUL")
        self.print_head()

    def print_head(self, rows: int = 5):
        print("\nSamples:")
        print(self.samples.head(rows).to_string())
        print("\nTargets:")
        print(self.targets.head(rows).to_string())
        print("\nEngine / current cycle / RUL alignment:")
        preview = self.samples[[self.cycle_column]].head(rows).copy()
        if self.engine_id_column in self.samples.columns:
            engine_ids = self.samples[self.engine_id_column]
        else:
            engine_ids = self.engine_ids
        preview.insert(0, self.engine_id_column, engine_ids.head(rows))
        preview["RUL"] = self.targets.head(rows)
        print(preview.to_string())
