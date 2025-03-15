import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
import dill

def save_object(file_path, obj):
    '''
    this function is the responsible for the saving the object
    '''
    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''
    this function is the responsible for the evaluation diferents algorithms
    '''
    try:
        report = {}

        # for i in range(len(list(models))):
        for name_dic, obj_dic in models.items():    
            model = obj_dic
            model.fit(X_train, y_train)

            # Making predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate model
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name_dic]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models_tunn_Grid(X_train, y_train, X_test, y_test, models, param):
    '''
    this function is the responsible for the evaluation diferents algorithms
    and hyperparameters to find the best model
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]  #list(models.values())[i]
            para= param[model_name]    ##param[list(models.keys())[i]]

            # Tuning hyperparameters
            gs = GridSearchCV(model, para, cv=5) #, n_jobs=-1, verbose=verbose, refit=refit)
            gs.fit(X_train, y_train)

            # traing the model with best hyperparameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Making predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate model
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name]=test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
