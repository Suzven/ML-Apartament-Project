import pandas as pd


class DatasetStatistics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def print_quality_checks(self):
        print("\nMissing values:")
        print(self.df.isna().sum())
        print("\nDuplicate rows:")
        print(self.df.duplicated().sum())

    def run(self):
        self.print_quality_checks()
        feature_std = pd.Series(dtype=float)

        for column, values in self.df.select_dtypes(include="number").items():
            values = values.dropna()
            mean = values.mean()
            median = values.median()
            modes = values.mode().tolist()
            std = values.std(ddof=0)
            variance = values.var(ddof=0)
            min_value = values.min()
            max_value = values.max()
            value_range = max_value - min_value
            percentile_95 = values.quantile(0.95)
            feature_std.loc[column] = std

            print(f"\n{'=' * 50}\nFeature {column}\n{'=' * 50}")
            print("mean:", mean)
            print("median:", median)
            print("mode:", modes)
            print("std:", std)
            print("variance:", variance)
            print("min:", min_value)
            print("max:", max_value)
            print("range:", value_range)
            print("95 percentile:", percentile_95)

        sorted_std = feature_std.sort_values(ascending=False, kind="stable")
        print(f"\n{'=' * 50}\nFEATURES SORTED BY STD\n{'=' * 50}")
        for column, std in sorted_std.items():
            print(f"Feature {column}: std = {std}")
        print("\nFeature order by std:")
        print(sorted_std.index.tolist())
