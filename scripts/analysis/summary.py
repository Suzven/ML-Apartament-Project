from collections import defaultdict

import numpy as np
import pandas as pd


class CorrelationSummary:
    def __init__(self, results, threshold: float = 0.8):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.results = list(results)
        self.threshold = threshold

    def run(self) -> pd.DataFrame:
        pair_correlations = defaultdict(list)
        for result in self.results:
            for first, second, correlation in result["all_correlated_pairs"]:
                pair_correlations[first, second].append(correlation)

        summary = []
        for (first, second), values in pair_correlations.items():
            correlations = np.asarray(values)
            strong_count = int(np.count_nonzero(np.abs(correlations) >= self.threshold))
            summary.append({
                "feature_1": first,
                "feature_2": second,
                "mean_corr": correlations.mean(),
                "median_corr": np.median(correlations),
                "std_corr": correlations.std(),
                "min_corr": correlations.min(),
                "max_corr": correlations.max(),
                "strong_count": strong_count,
                "engines_with_pair": len(correlations),
                "total_engines": len(self.results),
                "strong_percent": strong_count / len(self.results) * 100,
                "positive_percent": np.count_nonzero(correlations > 0) / len(correlations) * 100,
                "negative_percent": np.count_nonzero(correlations < 0) / len(correlations) * 100,
            })

        summary_df = pd.DataFrame(summary)
        print("\nCORRELATION STABILITY ACROSS ALL ENGINES")
        if summary_df.empty:
            print("No valid correlation pairs found.")
            return summary_df

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
        return summary_df
