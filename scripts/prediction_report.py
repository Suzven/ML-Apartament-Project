from pathlib import Path

import pandas as pd

if __package__:
    from .model_evaluation_metrics import RegressionMetrics
else:
    from model_evaluation_metrics import RegressionMetrics


def prediction_rows(dataset, predictions):
    RegressionMetrics(dataset.targets, predictions)
    rows = pd.DataFrame(index=dataset.samples.index)
    rows["engine_id"] = dataset.engine_ids
    rows["cycle"] = dataset.cycles
    rows["target"] = dataset.targets
    rows["prediction"] = predictions
    rows["residual"] = rows["prediction"] - rows["target"]
    return rows


def write_table(path, table, include_index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = table.to_string(index=include_index)
    path.write_text(text + "\n", encoding="utf-8")
