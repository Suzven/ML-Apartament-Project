if __package__:
    from .model_evaluation_metrics import ModelEvaluation
    from .analysis.grouped_model_evaluation import GroupedModelEvaluation
    from .preprocessing import SensorHistory
else:
    from model_evaluation_metrics import ModelEvaluation
    from analysis.grouped_model_evaluation import GroupedModelEvaluation
    from preprocessing import SensorHistory


class ModelTraining:
    def __init__(self, model, train, validation, draw_plots: bool = True, report_directory=None):
        self.model = model
        self.train = train
        self.validation = validation
        self.draw_plots = draw_plots
        self.report_directory = report_directory

    def run(self):
        if getattr(self.model, "requires_history", False):
            self.train = self.train.copy()
            self.validation = self.validation.copy()
            SensorHistory(self.train).run()
            SensorHistory(self.validation).run()

        self.model.fit(
            self.train.samples,
            self.train.targets,
            groups=self.train.engine_ids,
            validation=self.validation,
        )
        model_name = type(self.model).__name__
        for name, dataset in [("TRAIN", self.train), ("VALIDATION", self.validation)]:
            self.evaluate_dataset(name, dataset, model_name)

    def evaluate_dataset(self, name, dataset, model_name):
        predictions = self.model.predict(dataset.samples)
        ModelEvaluation(
            targets=dataset.targets,
            predictions=predictions,
            draw_plots=self.draw_plots,
            title=f"{model_name} | {name}",
        ).run()
        if self.report_directory is not None:
            GroupedModelEvaluation(
                dataset, predictions, name, self.report_directory
            ).run()
