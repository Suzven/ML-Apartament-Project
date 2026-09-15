# Apartment Price Regression

Small ML project for predicting apartment prices from 10 apartment and location features.

The main goal of the project was to go through a complete classical regression workflow: dataset preprocessing, categorical encoding, feature scaling, model comparison, regularization, polynomial feature generation, hyperparameter tuning, evaluation, and saving the final trained pipeline.

## Dataset

The dataset contains 1,000 apartment samples.

The model uses 10 input features:

### Numeric features

- `area_sqm` — apartment area in square meters
- `rooms` — number of rooms
- `floor` — apartment floor
- `total_floors` — total number of floors in the building
- `building_age_years` — age of the building
- `distance_to_center_km` — distance from the city center
- `condition_score` — apartment condition / renovation score
- `metro_distance_min` — walking time to the nearest metro / public transport

### Categorical features

- `city` — city where the apartment is located
- `district` — city district

Target:

- `price_usd` — apartment price in USD

Categorical features are encoded with `OneHotEncoder`, while numeric features are processed separately using a Scikit-learn pipeline.

## Model experiments

I started with regularized linear regression models to establish a baseline.

### Ridge Regression

`alpha = 1.0`

Test dataset:

| Metric | Result |
| --- | ---: |
| RMSE | $8,775.92 |
| RMSE / STD | 0.2485 |
| RMSE / Min-Max | 5.53% |

### Lasso Regression

`alpha = 1.0`

Test dataset:

| Metric | Result |
| --- | ---: |
| RMSE | $8,774.50 |
| RMSE / STD | 0.2485 |
| RMSE / Min-Max | 5.53% |

Ridge and Lasso produced almost identical results.

This suggested that the main limitation was probably not the choice between L1 and L2 regularization.

## Polynomial Features

The next experiment was to increase the expressive power of the model by generating polynomial and interaction features.

### Polynomial Features + Lasso

Polynomial degree: `2`  
Lasso alpha: `1.0`

Test dataset:

| Metric | Result |
| --- | ---: |
| RMSE | $7,964.15 |
| RMSE / STD | 0.2255 |
| RMSE / Min-Max | 5.02% |

This produced a significantly larger improvement than switching between Ridge, Lasso, and ElasticNet.

The experiment suggests that the original feature representation was a bigger limitation than the regularization strategy. The dataset contains relationships and feature interactions that are not represented well enough by the original linear feature space.

I also tested ElasticNet, but it did not improve the result compared with Lasso.

## Hyperparameter tuning

After introducing polynomial features, I used `GridSearchCV` with cross-validation to tune the Lasso regularization strength.

The preprocessing and regression steps are kept inside Scikit-learn pipelines so that preprocessing is fitted only on the corresponding training data during cross-validation.

### Polynomial Features + Lasso + GridSearchCV

After tuning and increasing the optimization iteration limit, the final model produced:

#### Train dataset

| Metric | Result |
| --- | ---: |
| RMSE | $6,537.85 |
| RMSE / STD | 0.1940 |
| RMSE / Min-Max | 4.40% |

#### Test dataset

| Metric | Result |
| --- | ---: |
| RMSE | **$7,417.16** |
| RMSE / STD | **0.2100** |
| RMSE / Min-Max | **4.67%** |

The difference between train and test RMSE is present but relatively small, so I stopped increasing model complexity at this point.

## Result

The initial regularized linear model had a test RMSE of approximately:

`$8,775`

The final polynomial Lasso model reduced it to:

`$7,417`

That is roughly a **15.5% reduction in test RMSE** compared with the initial baseline.

The most useful conclusion from the experiments was that changing the regularization method itself had very little effect:

`Ridge ≈ Lasso`

while changing the feature representation produced a much larger improvement:

`Linear features → Polynomial / interaction features`

So for this dataset, feature representation turned out to be more important than choosing a more complicated regularization strategy.

## Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib
