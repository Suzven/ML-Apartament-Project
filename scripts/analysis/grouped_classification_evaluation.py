from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

if __package__.startswith("scripts."):
    from ..classification_evaluation_metrics import ClassificationMetrics
else:
    from classification_evaluation_metrics import ClassificationMetrics


class GroupedClassificationEvaluation:
    def __init__(self, dataset, predictions, probabilities, classes, title, report_directory):
        self.dataset = dataset
        self.predictions = predictions
        self.probabilities = probabilities
        self.classes = classes
        self.title = title
        self.report_directory = Path(report_directory)
        self.groups = dataset.resource_groups
        self.summary = pd.DataFrame()
        self.confusion = pd.DataFrame()

    def run(self):
        targets = self.dataset.class_targets()
        overall = ClassificationMetrics(targets, self.predictions, self.probabilities, self.classes)
        predictions = overall.predictions
        probabilities = overall.probabilities
        summaries = []
        for group in self.groups:
            mask = targets.to_numpy() == group
            predicted_mask = predictions == group
            count = int(mask.sum())
            predicted_count = int(predicted_mask.sum())
            correct = int((mask & predicted_mask).sum())
            recall = correct / count if count else float("nan")
            precision = correct / predicted_count if predicted_count else float("nan")
            group_loss = float("nan")
            if count:
                group_metrics = ClassificationMetrics(targets.loc[mask], predictions[mask], probabilities[mask], self.classes)
                group_loss = group_metrics.log_loss()
            summaries.append({
                "Group": group,
                "Samples": count,
                "Engines": self.dataset.engine_ids.loc[mask].nunique(),
                "Predicted samples": predicted_count,
                "Correct": correct,
                "Accuracy within group": recall,
                "Precision": precision,
                "Recall": recall,
                "Log Loss": group_loss,
            })
        self.summary = pd.DataFrame(summaries)
        matrix = confusion_matrix(targets, predictions, labels=self.groups)
        self.confusion = pd.DataFrame(matrix, index=self.groups, columns=self.groups)
        self.confusion.index.name = "True group"
        self.confusion.columns.name = "Predicted group"
        summary_text = self.summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
        confusion_text = self.confusion.to_string()
        print(f"\n{self.title} | CLASSIFICATION BY RESOURCE GROUP")
        print("Groups are assigned by true remaining_fraction.")
        print("Accuracy within group equals Recall; Precision includes predictions from all groups.")
        print("Log Loss uses all class probabilities for samples of each true group.")
        print(summary_text)
        print(f"\n{self.title} | CONFUSION MATRIX | rows=true, columns=predicted")
        print(confusion_text)
        observations = pd.DataFrame(index=self.dataset.samples.index)
        observations["engine_id"] = self.dataset.engine_ids
        observations["cycle"] = self.dataset.cycles
        observations["remaining_fraction"] = self.dataset.targets
        observations["target_class"] = targets
        observations["predicted_class"] = predictions
        for position, group in enumerate(self.classes):
            observations[f"probability_{group}"] = probabilities[:, position]
        self.report_directory.mkdir(parents=True, exist_ok=True)
        prefix = self.title.lower()
        (self.report_directory / f"{prefix}_metrics.txt").write_text(summary_text + "\n", encoding="utf-8")
        (self.report_directory / f"{prefix}_confusion.txt").write_text(confusion_text + "\n", encoding="utf-8")
        (self.report_directory / f"{prefix}_predictions.txt").write_text(observations.to_string(index=True) + "\n", encoding="utf-8")
