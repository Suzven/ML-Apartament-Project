class DataCleaning:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        self.dataset.engine_ids = self.dataset.samples[0].copy()
        columns_to_drop = [0, 4, 5, 14, 22, 23]
        self.dataset.samples = self.dataset.samples.drop(columns=columns_to_drop)
        print("\nRemoved features:", columns_to_drop)
        print("Samples shape after cleaning:", self.dataset.samples.shape)
        self.dataset.print_head()
