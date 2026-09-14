import os

import pandas as pd
import numpy as np

from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso

from dataset_entity import DatasetEntity
import model_evaluation_metrics as me


class Model:

    def __init__(self):
        self.dataset = None
        self.best_model = None
        self.preprocessing_flow = None

    def loadDataset(
        self,
        csv_name: str,
        features: list[str],
        targets: str,
    ) -> None:

        df = pd.read_csv(csv_name)

        self.dataset = DatasetEntity(
            df=df,
            x=features,
            y=targets
        )

        #Check if loading finished correctly
        print(self.dataset.df.head())


    def preprocessing(
        self,
        categorical_features: list[str],
        numeric_features: list[str]
    ) -> None:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=3,
                        include_bias=False
                    )
                ),
                (
                    "standartScaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "oneHotEncoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        self.preprocessing_flow = ColumnTransformer(
            transformers=[
                (
                    "numeric_pipeline",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical_pipeline",
                    categorical_pipeline,
                    categorical_features
                )
            ]
        )


    def train(self) -> None:

        model_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self.preprocessing_flow
                ),
                (
                    "regression",
                    Lasso(
                        max_iter=15000
                    )
                )
            ]
        )

        param_grid = {
            "regression__alpha": [
                0.01,
                0.1,
                0.3,
                1.0,
                1.5,
                2.0,
                10.0,
            ],
        }

        search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )

        search.fit(
            self.dataset.x_train,
            self.dataset.y_train
        )

        #Check best params
        print(search.best_params_)

        #Get best model and import it
        self.best_model = search.best_estimator_

        y_train_pred = self.best_model.predict(
            self.dataset.x_train
        )

        self.evaluateMetrics(
            self.dataset.y_train,
            y_train_pred,
            "TRAIN"
        )


    def predictTest(self):

        y_test_pred = self.best_model.predict(
            self.dataset.x_test
        )

        self.evaluateMetrics(
            self.dataset.y_test,
            y_test_pred,
            "TEST"
        )


    def evaluateMetrics(
        self,
        y_true,
        y_pred,
        dataset_name: str
    ) -> None:

        print(
            f"______{dataset_name} DATASET METRICS:"
        )

        rmse = me.RMSE(
            y_true,
            y_pred
        )

        me.minMaxRMSE(
            rmse,
            y_true
        )

        me.stdRMSE(
            rmse,
            y_true
        )


    def exportModel(
        self,
        file_name: str = "best_house_price_model.joblib"
    ) -> None:

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        model_path = os.path.join(
            current_dir,
            "..",
            "models",
            file_name
        )

        model_path = os.path.abspath(
            model_path
        )

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        dump(
            self.best_model,
            model_path
        )

        print(
            f"Model saved to: {model_path}"
        )