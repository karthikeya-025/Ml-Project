import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate, save_obj


@dataclass
class ModelTrainConfig:
    train_model_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "Linear Regressor": LinearRegression(),
                "K-NN": KNeighborsRegressor(),
                "XGboost": XGBRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
            }

            model_report: dict = evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_score = max(sorted(list(model_report.values())))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info("Model training completed")
            save_obj(
                file_path=self.model_trainer_config.train_model_path, obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2
        except Exception as e:
            raise CustomException(e, sys)
