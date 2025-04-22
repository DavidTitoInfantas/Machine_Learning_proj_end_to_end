import pandas as pd
import numpy as np
import os 
import sys

from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

def continuous_training():
    ''''
    This function is responsible for the continuous training of the model.
    '''
    try:
        logging.info("Starting the training pipeline")
        logging.info("Loading the data ingestion component")
        obj=DataIngestion()
        train_data,test_data = obj.initiate_data_ingestion()

        logging.info("Loading the data transformation component")
        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

        logging.info("Loading the model trainer component")
        model_trainer = ModelTrainer()
        (r2_square, formatted_datetime, 
         model_name) = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Model training completed successfully")

        # Message to be logged
        mensagem = f"\n{formatted_datetime}// With model: {model_name}, has a R2 Square value: {r2_square:.4f}"

        # Abre (ou cria) o arquivo e escreve a linha
        caminho_arquivo = "results/metrics.txt"
        with open(caminho_arquivo, "a") as arquivo:
            arquivo.write(mensagem)

        logging.info(f"Model trained and saved as {model_name} with R2 score: {r2_square:.6f}")

    except Exception as e:
        raise CustomException(e, sys)
