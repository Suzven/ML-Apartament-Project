from sklearn.model_selection import train_test_split

class DatasetEntity:

    def __init__(
            self,
            df,
            x: list[str],
            y: str
        ) -> None:

        self.df = df
        self.x = self.df[x]
        self.y = self.df[y]
        self.splitDataset()

    def splitDataset(self) -> None:

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.2,
            random_state=20
        )