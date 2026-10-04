from .data_cleaning import DataCleaning
from .dataset_split import DatasetSplit


class DatasetPreprocessing:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train = None
        self.validation = None

    def run(self):
        split = DatasetSplit(self.dataset)
        split.run()
        self.train = split.train
        self.validation = split.validation

        DataCleaning(self.train).run()
        DataCleaning(self.validation).run()
