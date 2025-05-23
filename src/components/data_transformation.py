"""Component for data transformation."""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Data transformation configuration."""

    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Data transformation class."""

    def __init__(self):
        """Initialize the data transformation class."""
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Is the responsible for the data transformation."""
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="warn",
                        ),
                    ),
                    ("std_scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Is the responsible for the data transformation."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            imput_feature_train_df = train_df.drop(
                columns=[target_column_name], axis=1
            )
            target_feature_train_df = train_df[target_column_name]

            imput_feature_test_df = test_df.drop(
                columns=[target_column_name], axis=1
            )
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training "
                "dataframe and testing dataframe"
            )

            imput_feature_train_arr = preprocessing_obj.fit_transform(
                imput_feature_train_df
            )
            imput_feature_test_arr = preprocessing_obj.transform(
                imput_feature_test_df
            )

            train_arr = np.c_[
                imput_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                imput_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
