class DataCleaning:
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        constant_features = [4, 5, 14, 22, 23]
        excluded_features = [20, 9, 3, 10, 2, 19, 12, 17]
        columns_to_drop = [self.dataset.engine_id_column]
        columns_to_drop.extend(constant_features)
        columns_to_drop.extend(excluded_features)
        columns_to_drop.sort()
        self.dataset.samples = self.dataset.samples.drop(columns=columns_to_drop)
        print("\nRemoved features:", columns_to_drop)
        print("Samples shape after cleaning:", self.dataset.samples.shape)
        self.dataset.print_head()
