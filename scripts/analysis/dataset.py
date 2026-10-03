import pandas as pd

from .correlation import CorrelationAnalysis
from .scatter import ScatterPlots
from .statistics import DatasetStatistics


class DatasetAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        threshold: float = 0.8,
        draw_scatter: bool = True,
        top_n: int | None = 10,
        title: str = "",
    ):
        self.statistics = DatasetStatistics(df)
        self.correlation = CorrelationAnalysis(df, threshold)
        self.df = df
        self.draw_scatter = draw_scatter
        self.top_n = top_n
        self.title = title

    def run(self):
        self.statistics.run()
        pairs = self.correlation.run()
        ScatterPlots(self.df, pairs, self.draw_scatter, self.top_n, self.title).run()
