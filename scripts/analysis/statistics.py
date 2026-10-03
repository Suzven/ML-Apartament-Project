import pandas as pd


class DatasetStatistics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def print_quality_checks(self):
        print("\nMissing values:")
        print(self.df.isna().sum())
        print("\nDuplicate rows:")
        print(self.df.duplicated().sum())

    def run(self) -> pd.DataFrame:
        self.print_quality_checks()
        numeric_df = self.df.select_dtypes(include="number")
        statistics = pd.DataFrame({
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "min": numeric_df.min(),
            "max": numeric_df.max(),
        })
        statistics["mode"] = pd.Series(
            {column: values.mode().tolist() for column, values in numeric_df.items()},
            dtype=object,
        )
        statistics["std"] = numeric_df.std(ddof=0)
        statistics["variance"] = numeric_df.var(ddof=0)
        statistics["range"] = statistics["max"] - statistics["min"]
        statistics["95 percentile"] = numeric_df.quantile(0.95)
        statistics = statistics[
            ["mean", "median", "mode", "std", "variance", "min", "max", "range", "95 percentile"]
        ]

        for column, values in statistics.iterrows():
            print(f"\n{'=' * 50}\nFeature {column}\n{'=' * 50}")
            for name, value in values.items():
                print(f"{name}: {value}")

        sorted_std = statistics["std"].sort_values(ascending=False, kind="stable")
        print(f"\n{'=' * 50}\nFEATURES SORTED BY STD\n{'=' * 50}")
        for column, std in sorted_std.items():
            print(f"Feature {column}: std = {std}")
        print("\nFeature order by std:")
        print(sorted_std.index.tolist())
        return statistics
