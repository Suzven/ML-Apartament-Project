import numpy as np
import pandas as pd


class CorrelationSummary:
    def __init__(self, engine_correlations, threshold: float = 0.8):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.engine_correlations = engine_correlations
        self.threshold = threshold

    def run(self):
        pair_correlations = {}
        for engine_pairs in self.engine_correlations:
            for first, second, correlation in engine_pairs:
                pair = (first, second)
                if pair not in pair_correlations:
                    pair_correlations[pair] = []
                pair_correlations[pair].append(correlation)

        summary_df = pd.DataFrame(columns=[
            "feature_1", "feature_2", "mean_corr", "median_corr", "std_corr",
            "min_corr", "max_corr", "strong_count", "engines_with_pair",
            "total_engines", "strong_percent", "positive_percent", "negative_percent",
        ])
        total_engines = len(self.engine_correlations)

        for (first, second), values in pair_correlations.items():
            correlations = np.asarray(values)
            mean_corr = correlations.mean()
            median_corr = np.median(correlations)
            std_corr = correlations.std()
            min_corr = correlations.min()
            max_corr = correlations.max()
            strong_count = int(np.count_nonzero(np.abs(correlations) >= self.threshold))
            engines_with_pair = len(correlations)
            strong_percent = strong_count / total_engines * 100
            positive_percent = np.count_nonzero(correlations > 0) / engines_with_pair * 100
            negative_percent = np.count_nonzero(correlations < 0) / engines_with_pair * 100

            summary_df.loc[len(summary_df)] = {
                "feature_1": first,
                "feature_2": second,
                "mean_corr": mean_corr,
                "median_corr": median_corr,
                "std_corr": std_corr,
                "min_corr": min_corr,
                "max_corr": max_corr,
                "strong_count": strong_count,
                "engines_with_pair": engines_with_pair,
                "total_engines": total_engines,
                "strong_percent": strong_percent,
                "positive_percent": positive_percent,
                "negative_percent": negative_percent,
            }

        print("\nCORRELATION STABILITY ACROSS ALL ENGINES")
        if summary_df.empty:
            print("No valid correlation pairs found.")
            return

        summary_df["abs_median_corr"] = summary_df["median_corr"].abs()
        summary_df = summary_df.sort_values(
            ["strong_percent", "abs_median_corr", "std_corr"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        display_columns = summary_df.columns.drop(["engines_with_pair", "abs_median_corr"])
        print(summary_df[display_columns].to_string(index=False))
        print("\nTOP STABLE STRONG CORRELATIONS")
        top_stable = summary_df.loc[summary_df["strong_percent"] > 0].head(30)
        print(top_stable[display_columns].to_string(index=False))
