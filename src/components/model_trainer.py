import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor

from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object , evaluate_model

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info("Split the data into training and testing")
            X_train  , y_train , X_test , y_test = train_array[:,:-1] , train_array[:,-1] , test_array[:,:-1] , test_array[:,-1]

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression':{},
                'Lasso':{},
                'Ridge':{},
                'KNN':{
                    'n_neighbors': [2,3,4,5,6,7,8],
                    'weights': ['uniform','distance'],
                    'algorithm': ['auto','ball_tree','kd_tree','brute']
                },
                "XGBoost":{
                    # 'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    # 'depth': [6,8,10],
                    # 'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost":{
                    # 'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report = evaluate_model(X_train = X_train , y_train = y_train , X_test = X_test , y_test = y_test , models=models , param = params)

            # Get the best model name based on R2 score
            best_model_name = max(model_report , key = lambda x : model_report[x]['R2 Score'])

            # Get the best model score
            best_model_score = model_report[best_model_name]['R2 Score']

            if best_model_score < 0.6:
                raise CustomException("Best model score is less than 0.6")

            logging.info(f"Best model is {best_model_name} with R2 Score {best_model_score}")

            save_object(self.model_trainer_config.model_path , models[best_model_name])

            best_model = models[best_model_name]
            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test , predicted)

            return r2_score_value

        except Exception as e:
            raise CustomException(e , sys)