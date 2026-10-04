if __package__:
    from .model_evaluation_metrics import ModelEvaluation
else:
    from model_evaluation_metrics import ModelEvaluation


class ModelTraining:
    def __init__(self, model, train, validation, draw_plots: bool = True):
        self.model = model
        self.train = train
        self.validation = validation
        self.draw_plots = draw_plots

    def run(self):
        self.model.fit(
            self.train.samples,
            self.train.targets,
            groups=self.train.engine_ids,
            validation=self.validation,
        )
        model_name = type(self.model).__name__
        for name, dataset in [("TRAIN", self.train), ("VALIDATION", self.validation)]:
            predictions = self.model.predict(dataset.samples)
            ModelEvaluation(
                targets=dataset.targets,
                predictions=predictions,
                draw_plots=self.draw_plots,
                title=f"{model_name} | {name}",
            ).run()
