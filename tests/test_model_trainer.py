"""Modules for testing the model trainer component."""

import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.utils import load_object
import pytest


class TestModelTrainer:
    """Test class for ModelTrainer."""

    @pytest.fixture()
    def sample_data(self, tmp_path):
        """Create data for test the functions."""
        # Create data for test
        data_train = {
            "gender": [
                "male",
                "male",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "female",
                "female",
                "female",
            ],
            "race_ethnicity": [
                "group B",
                "group A",
                "group B",
                "group E",
                "group E",
                "group C",
                "group C",
                "group B",
                "group D",
                "group A",
                "group B",
                "group D",
                "group E",
                "group C",
                "group B",
            ],
            "parental_level_of_education": [
                "associate's degree",
                "master's degree",
                "some college",
                "associate's degree",
                "associate's degree",
                "associate's degree",
                "some college",
                "high school",
                "high school",
                "associate's degree",
                "some college",
                "bachelor's degree",
                "master's degree",
                "associate's degree",
                "some high school",
            ],
            "lunch": [
                "free/reduced",
                "free/reduced",
                "standard",
                "free/reduced",
                "standard",
                "standard",
                "standard",
                "standard",
                "free/reduced",
                "free/reduced",
                "standard",
                "free/reduced",
                "free/reduced",
                "free/reduced",
                "free/reduced",
            ],
            "test_preparation_course": [
                "none",
                "none",
                "none",
                "none",
                "completed",
                "none",
                "none",
                "none",
                "completed",
                "none",
                "completed",
                "completed",
                "none",
                "none",
                "none",
            ],
            "reading_score": [
                56,
                74,
                54,
                56,
                81,
                73,
                78,
                81,
                64,
                57,
                95,
                71,
                72,
                58,
                32,
            ],
            "writing_score": [
                57,
                72,
                55,
                54,
                79,
                68,
                75,
                73,
                67,
                44,
                92,
                80,
                65,
                61,
                28,
            ],
            "math_score": [57, 73, 69, 50, 81, 58, 76, 65, 64, 47, 88, 74, 56, 54, 18],
        }
        data_test = {
            "gender": ["male", "female", "male", "female", "male"],
            "race_ethnicity": ["group A", "group C", "group D", "group B", "group E"],
            "parental_level_of_education": [
                "some college",
                "bachelor's degree",
                "associate's degree",
                "some college",
                "high school",
            ],
            "lunch": [
                "standard",
                "standard",
                "free/reduced",
                "standard",
                "free/reduced",
            ],
            "test_preparation_course": [
                "completed",
                "none",
                "none",
                "completed",
                "none",
            ],
            "math_score": [78, 67, 40, 88, 54],
            "reading_score": [72, 69, 52, 95, 58],
            "writing_score": [70, 75, 43, 92, 61],
        }
        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)

        # Target column name
        target_column_name = "math_score"

        # Separate input and target
        imput_feature_train_df = df_train.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = df_train[target_column_name]

        imput_feature_test_df = df_test.drop(columns=[target_column_name], axis=1)
        target_feature_test_df = df_test[target_column_name]

        # Load preprocessor object
        transformation_config = DataTransformationConfig()
        preprocessing_obj = load_object(
            path_file=transformation_config.preprocessor_obj_file_path
        )

        # Transform data
        imput_feature_train_arr = preprocessing_obj.transform(imput_feature_train_df)
        imput_feature_test_arr = preprocessing_obj.transform(imput_feature_test_df)

        # Concat array with the target
        train_arr = np.c_[imput_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[imput_feature_test_arr, np.array(target_feature_test_df)]

        path_save_numpy = os.path.join(tmp_path, "arrays_train_test.npz")
        np.savez(path_save_numpy, array1=train_arr, array2=test_arr)

        return path_save_numpy

    @pytest.fixture(scope="class")
    def trainer(self):
        """Create an instance for ModelTrainer."""
        return ModelTrainer()

    def test_initiate_model_trainer(self, trainer, sample_data):
        """Test the function initiate_model_trainer with data sample."""
        # path_save_numpy = os.path.join(tmp_path,'arrays_train_test.npz')
        path_save_numpy = sample_data
        data_arr = np.load(path_save_numpy)
        train_arr = data_arr["array1"]
        test_arr = data_arr["array2"]

        r2_score, formatted_datetime, model_name = trainer.initiate_model_trainer(
            train_arr, test_arr
        )

        assert isinstance(r2_score, float)
        assert 0 <= r2_score <= 1
        assert isinstance(formatted_datetime, str)
        assert isinstance(formatted_datetime, str)
