import pandas as pd
import numpy as np
import os

from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
import pytest

class TestModelTrainer:

    @pytest.fixture(scope="class")
    def sample_data(self, tmp_path):
        """
        Create data for rutine
        """
        # Create data for test
        data = {
            'gender': 
                ['male', 'male', 'male', 'female', 'male', 'female', 
                 'male', 'female', 'male', 'male', 'female', 'male', 
                 'female', 'female', 'female'],
            'race_ethnicity': 
                ['group B', 'group A', 'group B', 'group E', 
                 'group E', 'group C', 'group C', 'group B', 
                 'group D', 'group A', 'group B', 'group D', 
                 'group E', 'group C', 'group B'],
            'parental_level_of_education': 
                ["associate's degree", "master's degree", 
                 'some college', "associate's degree", 
                 "associate's degree", "associate's degree", 
                 'some college', 'high school', 'high school', 
                 "associate's degree", 'some college', 
                 "bachelor's degree", "master's degree", 
                 "associate's degree", 'some high school'],
            'lunch': 
                ['free/reduced', 'free/reduced', 'standard', 
                 'free/reduced', 'standard', 'standard', 
                 'standard', 'standard', 'free/reduced', 
                 'free/reduced', 'standard', 'free/reduced', 
                 'free/reduced', 'free/reduced', 'free/reduced'],
            'test_preparation_course': 
                ['none', 'none', 'none', 'none', 'completed', 
                 'none', 'none', 'none', 'completed', 'none', 
                 'completed', 'completed', 'none', 'none', 'none'],
            'reading_score': 
                [56, 74, 54, 56, 81, 73, 78, 81, 64, 57, 95, 71, 
                 72, 58, 32],
            'writing_score': 
                [57, 72, 55, 54, 79, 68, 75, 73, 67, 44, 92, 80, 
                 65, 61, 28],
            'math_score': 
                [57, 73, 69, 50, 81, 58, 76, 65, 64, 47, 88, 74, 
                 56, 54, 18],
            }
        df_sample = pd.DataFrame(data)
        df_sample.to_csv(os.path.join(
                tmp_path, 'notebook', 'data', 'data_sample.csv'
                )
            )


    @pytest.fixture(scope="class")
    def ingestion(self):
        '''
        Create an instance of DataIngestion
        '''
        return DataIngestion()

    def test_initiate_data_ingestion(self, tmp_path, ingestion):
        '''
        Test the function with dataframe test
        '''

        # Update paths to use temporary directory
        config = DataIngestionConfig(
            source_data_path=os.path.join(
                tmp_path, 'notebook', 'data', 'data_sample.csv'
                ),
            train_data_path=os.path.join(
                tmp_path, 'artifacts','train.csv'
                ),
            test_data_path=os.path.join(
                tmp_path, 'artifacts', 'test.csv'
                ),
            raw_data_path=os.path.join(
                tmp_path, 'artifacts', 'data.csv'
                )
            )
        
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # Validate if files exists
        assert os.path.exists(train_data_path)
        assert os.path.exists(test_data_path)










    









