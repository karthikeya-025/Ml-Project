import os
import sys

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

from dataclasses import dataclass


@dataclass
class DataTranformConfig:
    preprocessor_obj_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransform:
    def __init__(self):
        self.preprocessor_path = DataTranformConfig()

    def get_data_transformer_obj(self):
        """
        This function is responsible for data transformation
        """
        try:
            num_vars = ["writing_score", "reading_score"]
            cat_vars = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("The data is encoded and standardized")
            preprocessor = ColumnTransformer(
                [("num_tsfr", num_pipe, num_vars), ("cat_tsfr", cat_pipe, cat_vars)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def data_tranform_initiate(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The train and test path is read as DataFrame")
            logging.info("The preprocessing object is obtained")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col = ["math_score"]
            num_vars = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(target_col, axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(target_col, axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Preprocessing Phase has been Initiated")

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.preprocessor_path.preprocessor_obj_path,
                obj=preprocessing_obj,
            )

            logging.info("The preprocessing phase has been completed and saved")

            return (train_arr, test_arr, self.preprocessor_path.preprocessor_obj_path)

        except Exception as e:
            raise CustomException(e, sys)
