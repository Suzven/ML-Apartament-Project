from itertools import combinations

import pandas as pd


class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame, threshold: float = 0.8):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.df = df
        self.threshold = threshold
        self.correlation_matrix = pd.DataFrame()
        self.all_pairs = []
        self.strong_pairs = []

    def run(self):
        self.correlation_matrix = self.df.corr(method="pearson", numeric_only=True)
        self.all_pairs = []
        self.strong_pairs = []
        for first, second in combinations(self.correlation_matrix.columns, 2):
            correlation = self.correlation_matrix.loc[first, second]
            if pd.isna(correlation):
                continue

            pair = (first, second, correlation)
            self.all_pairs.append(pair)
            if abs(correlation) >= self.threshold:
                self.strong_pairs.append(pair)

        self.strong_pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)
        print(f"\n{'=' * 50}\nCORRELATION MATRIX\n{'=' * 50}")
        print(self.correlation_matrix)
        print(
            f"\n{'=' * 50}\nCORRELATED FEATURES | "
            f"abs(corr) >= {self.threshold}\n{'=' * 50}"
        )
        if not self.strong_pairs:
            print("No strongly correlated feature pairs found.")
        for first, second, correlation in self.strong_pairs:
            print(f"Feature {first} <-> Feature {second}: corr = {correlation:.6f}")
        return self.strong_pairs
