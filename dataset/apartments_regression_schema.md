# Apartments Regression Dataset

Synthetic dataset for practicing regression pipelines, hyperparameter tuning, Ridge, Lasso, ElasticNet, PolynomialFeatures and Core ML conversion.

## Size
- Rows: 1000
- ML features: 10
- Target: `price_usd`
- Extra identifier: `sample_id`
- Missing values: none

## Features
1. `area_sqm` — apartment area in square meters
2. `rooms` — number of rooms
3. `floor` — apartment floor
4. `total_floors` — floors in the building
5. `building_age_years` — building age
6. `distance_to_center_km` — distance to city center
7. `condition_score` — condition/renovation score from 1 to 10
8. `metro_distance_min` — approximate walking time to metro/public transit
9. `city` — categorical feature
10. `district` — categorical feature

## Target
- `price_usd`

## Dataset characteristics
- Mostly linear signal with moderate noise.
- Mild interaction/non-linear effects are included.
- Suitable for comparing LinearRegression, Ridge, Lasso, ElasticNet and PolynomialFeatures(degree=2).
- Categorical columns are suitable for OneHotEncoder.
- Numeric columns are suitable for StandardScaler.
- No missing values.
- Price range is constrained to approximately $20,000–$200,000.

## Quick stats
- Price min: $41,200
- Price max: $200,000
- Price mean: $130,237
- Area min: 30.6 m²
- Area max: 142.7 m²

## Suggested sklearn pipeline
Use a `ColumnTransformer`:
- numeric features -> `StandardScaler`
- categorical features -> `OneHotEncoder(handle_unknown="ignore")`

Then test:
- `LinearRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`
- optionally `PolynomialFeatures(degree=2)` on numeric features + `Ridge`

For tuning, try `GridSearchCV` or `RandomizedSearchCV`.
