import os
import sys

import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path , obj):
    '''
    This function is used to save the object
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , 'wb') as file:
            dill.dump(obj , file)
    except Exception as e:
        raise CustomException(e , sys)

def evaluate_model(X_train , y_train , X_test , y_test , models , param ):
    '''
    This function is used to evaluate the model
    '''
    try:
        model_report = {}
        for model_name , model in models.items():
            # model.fit(X_train , y_train)
            params = param[model_name]

            gs = GridSearchCV(model , params , cv = 5 , n_jobs = -1 , verbose = 1)
            gs.fit(X_train , y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train , y_train)

            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            model_report[model_name] = {
                'Mean Squared Error': np.round(mean_squared_error(y_test , y_pred) , 2),
                'R2 Score': np.round(r2_score(y_test , y_pred) , 2),
                'Train Mean Squared Error': np.round(mean_squared_error(y_train , y_train_pred) , 2),
                'Train R2 Score': np.round(r2_score(y_train , y_train_pred) , 2)
            }

        return model_report
    
    except Exception as e:
        raise CustomException(e , sys)

def load_object(file_path):
    try:
        with open(file_path , "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e , sys)