"""Component responsible for data ingestion."""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Data ingestion configuration."""

    source_data_path: str = os.path.join("notebook", "data", "stud.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """Data ingestion class."""

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """Initialize the data ingestion class."""
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        """Is the responsible for the data ingestion."""
        logging.info("Entered the data ingestion method or component")
        try:
            # df=pd.read_csv('notebook/data/stud.csv')
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info("Read data set as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True,
            )

            df.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=24
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


## if __name__ == "__main__":
##
##     obj=DataIngestion()
##     train_data,test_data = obj.initiate_data_ingestion()
##
##     data_transformation = DataTransformation()
##     train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
##
##     model_trainer = ModelTrainer()
##     print(model_trainer.initiate_model_trainer(train_arr, test_arr))
