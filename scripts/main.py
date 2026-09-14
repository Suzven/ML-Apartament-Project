from model import Model

def main():

    x = [
        "city",
        "district",
        "area_sqm",
        "rooms",
        "floor",
        "total_floors",
        "building_age_years",
        "distance_to_center_km",
        "condition_score",
        "metro_distance_min"
    ]

    y = "price_usd"

    model = Model()
    model.loadDataset(
        "../dataset/apartments_regression_1000.csv",
        x,
        y
    )

    categorical_features = [
        "city",
        "district"
    ]
    
    numeric_features = [
        "area_sqm",
        "rooms",
        "floor",
        "total_floors",
        "building_age_years",
        "distance_to_center_km",
        "condition_score",
        "metro_distance_min"
    ]

    model.preprocessing(
        categorical_features,
        numeric_features
    )

    model.train()
    model.predictTest()
    model.exportModel()

if __name__ == "__main__":
    main()