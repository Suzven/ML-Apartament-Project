import pandas as pd


class ScatterPlots:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_pairs,
        enabled: bool = True,
        top_n: int | None = 10,
        title: str = "",
    ):
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be non-negative or None")
        self.df = df
        self.feature_pairs = list(feature_pairs)
        self.enabled = enabled
        self.top_n = top_n
        self.title = title

    def run(self):
        if not self.enabled:
            return

        import matplotlib.pyplot as plt

        pairs = self.feature_pairs[:self.top_n]
        print(f"\nSCATTER PLOTS FOR TOP {len(pairs)} CORRELATIONS")
        title_prefix = f"{self.title}\n" if self.title else ""
        for first, second, correlation in pairs:
            figure, axes = plt.subplots(figsize=(8, 6))
            axes.scatter(self.df[first], self.df[second], alpha=0.3, s=10)
            axes.set(
                xlabel=f"Feature {first}",
                ylabel=f"Feature {second}",
                title=(
                    f"{title_prefix}"
                    f"Feature {first} vs Feature {second}\n"
                    f"Pearson correlation = {correlation:.4f}"
                ),
            )
            axes.grid(alpha=0.2)
            figure.tight_layout()
            plt.show()
            plt.close(figure)
