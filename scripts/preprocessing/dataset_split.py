class DatasetSplit:
    def __init__(self, dataset):
        self.dataset = dataset
        self.train = None
        self.validation = None

    def run(self):
        engine_ids = self.dataset.engine_ids
        train_mask = engine_ids.between(1, 80)
        validation_mask = engine_ids.between(81, 100)
        if not (train_mask | validation_mask).all():
            raise ValueError("Engine IDs must be in ranges 1–80 or 81–100")

        self.train = self.dataset.select_rows(train_mask)
        self.validation = self.dataset.select_rows(validation_mask)

        print("\nTRAIN | engines 1–80 | rows:", len(self.train.samples))
        self.train.print_head()
        print("\nVALIDATION | engines 81–100 | rows:", len(self.validation.samples))
        self.validation.print_head()
