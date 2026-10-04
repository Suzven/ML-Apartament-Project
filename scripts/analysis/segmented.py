import pandas as pd

from .dataset import DatasetAnalysis
from .statistics import DatasetStatistics
from .summary import CorrelationSummary


class SegmentedDatasetAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        engine_id_column=0,
        threshold: float = 0.8,
        draw_scatter: bool = False,
        top_n: int | None = 5,
        engines_to_plot=None,
        engine_ids: pd.Series | None = None,
    ):
        self.df = df
        self.engine_id_column = engine_id_column
        self.engine_ids = engine_ids
        self.threshold = threshold
        self.draw_scatter = draw_scatter
        self.top_n = top_n
        self.engines_to_plot = None if engines_to_plot is None else set(engines_to_plot)

    def run(self):
        group_key = self.engine_id_column if self.engine_ids is None else self.engine_ids
        groups = self.df.groupby(group_key, sort=True)
        print(f"\nNumber of engines: {groups.ngroups}")
        print("Engine IDs:", list(groups.groups))
        DatasetStatistics(self.df).print_quality_checks()
        engine_correlations = []
        for engine_id, engine_df in groups:
            title = f"ENGINE ID: {engine_id}"
            print(f"\n{'#' * 80}\n{title}\nROWS: {len(engine_df)}\n{'#' * 80}")
            should_plot = self.draw_scatter and (
                self.engines_to_plot is None or engine_id in self.engines_to_plot
            )
            features_df = engine_df
            if self.engine_id_column in features_df.columns:
                features_df = features_df.drop(columns=[self.engine_id_column])
            analysis = DatasetAnalysis(
                features_df,
                threshold=self.threshold,
                draw_scatter=should_plot,
                top_n=self.top_n,
                title=title,
            )
            analysis.run()
            engine_correlations.append(analysis.correlation.all_pairs)

        CorrelationSummary(engine_correlations, self.threshold).run()
        print(f"\nANALYSIS FINISHED\nAnalyzed engines: {len(engine_correlations)}")
