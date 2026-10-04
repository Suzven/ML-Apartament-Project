import pandas as pd
from sklearn.preprocessing import StandardScaler


class StandardScaling:
    def __init__(self, train, validation):
        self.train = train
        self.validation = validation
        self.columns = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
        self.scaler = StandardScaler()

    def run(self):
        train_scaled = self.scaler.fit_transform(self.train.samples[self.columns])
        validation_scaled = self.scaler.transform(self.validation.samples[self.columns])

        self.train.samples[self.columns] = pd.DataFrame(
            train_scaled,
            index=self.train.samples.index,
            columns=self.columns,
        )
        self.validation.samples[self.columns] = pd.DataFrame(
            validation_scaled,
            index=self.validation.samples.index,
            columns=self.columns,
        )

        print("\nStandardScaler features:", self.columns)
        print("\nMODEL READY TRAIN | scaled samples:")
        print(self.train.samples.head().to_string())
        print("\nTrain targets (cycles):")
        print(self.train.targets.head().to_string())
        print("\nMODEL READY VALIDATION | scaled samples:")
        print(self.validation.samples.head().to_string())
        print("\nValidation targets (cycles):")
        print(self.validation.targets.head().to_string())
