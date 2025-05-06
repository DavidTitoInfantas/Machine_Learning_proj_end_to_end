"""Module for making predictions using a trained model."""

import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """Class for making predictions using a trained model."""

    def __init__(self):
        """Initialize the PredictPipeline class."""
        pass

    def predict(self, features):
        """Make predictions using the trained model."""
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(path_file=model_path)
            preprocessor = load_object(path_file=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Class for handling custom data input for predictions."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """Initialize the CustomData class with input features."""
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """Convert the custom data input into a pandas DataFrame."""
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
