from .data_cleaning import DataCleaning
from .dataset_split import DatasetSplit
from .feature_engineering import FeatureEngineering


class DatasetPreprocessing:
    def __init__(self, dataset, engineer_features=True):
        self.dataset = dataset
        self.engineer_features = engineer_features
        self.train = None
        self.validation = None

    def run(self):
        split = DatasetSplit(self.dataset)
        split.run()
        self.train = split.train
        self.validation = split.validation

        DataCleaning(self.train).run()
        DataCleaning(self.validation).run()
        if self.engineer_features:
            FeatureEngineering(self.train).run()
            FeatureEngineering(self.validation).run()
