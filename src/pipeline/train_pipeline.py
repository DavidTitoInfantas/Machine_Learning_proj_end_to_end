"""Module to handle the training pipeline of the model."""

import os
import sys

import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


def continuous_training():
    """Is responsible for the continuous training of the model."""
    try:
        logging.info("Starting the training pipeline")
        logging.info("Loading the data ingestion component")
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        logging.info("Loading the data transformation component")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data, test_data
            )
        )

        logging.info("Loading the model trainer component")
        model_trainer = ModelTrainer()
        (r2_square, formatted_datetime, model_name) = (
            model_trainer.initiate_model_trainer(train_arr, test_arr)
        )

        logging.info("Model training completed successfully")

        # Message to be logged
        mensagem = f"\n{formatted_datetime}// With model: {model_name}, has a R2 Square value: {r2_square:.4f}"

        # Abre (ou cria) o arquivo e escreve a linha
        caminho_arquivo = "results/metrics.txt"
        with open(caminho_arquivo, "a") as arquivo:
            arquivo.write(mensagem)

        logging.info(
            f"Model trained and saved as {model_name} with R2 score: {r2_square:.6f}"
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    """Execute the continuous training."""
    continuous_training()
