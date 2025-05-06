"""Component responsible for training the model."""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models, evaluate_models_tunn_Grid


@dataclass
class ModelTrainerConfig:
    """Model trainer configuration."""

    # Current date
    current_datetime = datetime.now()

    # Format date
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

    # Model name
    model_name = f"model_{formatted_datetime}.pkl"

    # Create the path
    trained_model_file_path = os.path.join("artifacts", model_name)


class ModelTrainer:
    """Model trainer class."""

    def __init__(self):
        """Initialize the model trainer."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Is the responsible for the model training."""
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
            }

            params = {
                "Linear regression": {
                    ##'fit_intercept': [True, False],
                    ##'normalize': [True, False]
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "XGB Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Flag to check if hyperparameter tuning is required
            tunning_method = True

            if tunning_method:
                # Available models with hyperparameter tuning using GridSearchCV
                model_report: dict = evaluate_models_tunn_Grid(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    param=params,
                )
            else:
                # Available models
                model_report: dict = evaluate_models(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    models=models,
                    param=params,
                )

            ## to get best model score from dict
            best_model_score = max(model_report.values())

            ## to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(
                f"Best found model on both training and test dataset: {best_model_name}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            plt.scatter(y_test, predicted)
            plt.plot(
                [min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--"
            )  # Diagonal
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(
                f"Predicted vs Actual with model: {self.model_trainer_config.model_name}"
            )

            # Save the graphic
            plt.savefig(
                f"results/model_results_{self.model_trainer_config.formatted_datetime}.png"
            )
            plt.show()

            r2_square = r2_score(y_test, predicted)

            return (
                r2_square,
                self.model_trainer_config.formatted_datetime,
                self.model_trainer_config.model_name,
            )

        except Exception as e:
            raise CustomException(e, sys)
