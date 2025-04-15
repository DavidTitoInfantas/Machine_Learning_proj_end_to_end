import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
import os
from sklearn.compose import ColumnTransformer
import pytest

class TestDataTransformation:

    @pytest.fixture
    def sample_data(self, tmp_path):
        """
        Create data for train and test for the tests
        """
        # Create data for train and test
        train_data={
            'gender': ['male', 'feamle'],
            'race_ethnicity': ['group A', 'group B'], 
            'parental_level_of_education': ["bachelor's degree", "some college"],
            'lunch': ["standard", "free/reduced"], 
            'test_preparation_course': ["none", "completed"],
            'writing_score': [74, 88],
            'reading_score': [70, 85],
            'math_score': [40, 80],
        }

        test_data = {
            'gender': ['female'],
            'race_ethnicity': ['group C'],
            'parental_level_of_education': ['master\'s degree'],
            'lunch': ['standard'],
            'test_preparation_course': ['none'],
            'reading_score': [95],
            'writing_score': [93],
            'math_score': [60],
        }

        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        train_path = os.path.join(tmp_path,"train_teste.csv")
        test_path = os.path.join(tmp_path,"test_teste.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        return str(train_path), str(test_path)

    @pytest.fixture(scope="class")
    def transformer(self):
        """
        Create an instance for DataTrandformation object
        """
        return DataTransformation()

    def test_get_data_transform_object(self, transformer):
        """
        Test the object of transformation
        """
        preprocessor = transformer.get_data_transformer_object()
        assert isinstance(preprocessor, ColumnTransformer)


    def test_initiate_data_transformation(self, sample_data):
        """
        Test the tranformations of data and the types of arrays
        """
        train_path, test_path = sample_data
        transformer = DataTransformation()

        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

        # Validate if array was created
        assert isinstance(train_arr, np.ndarray)
        assert isinstance(test_arr, np.ndarray)

        # Validade if the file was saved 
        assert os.path.exists(preprocessor_path)
