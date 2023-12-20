import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransform, DataTranformConfig
from src.components.model_trainer import ModelTrainer, ModelTrainConfig


@dataclass
class DataIngestConfig:
    train_data_path = os.path.join("artifact", "train.csv")
    test_data_path = os.path.join("artifact", "test.csv")
    raw_data_path = os.path.join("artifact", "data.csv")


class DataIngest:
    def __init__(self):
        self.ingestion_config = DataIngestConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Phase")
        try:
            df = pd.read_csv(
                r"C:\Users\Lenovo\Desktop\webapps\MLproject\notebooks\data\stud.csv"
            )
            logging.info("Dataset is exported")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            logging.info("Train and test split is initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info("Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngest()
    train_path, test_path = obj.initiate_data_ingestion()
    data_transform = DataTransform()
    train, test, preprocessor = data_transform.data_tranform_initiate(
        train_path, test_path
    )
    model = ModelTrainer()
    r2 = model.initiate_model_trainer(train, test, preprocessor)
    print(r2)
