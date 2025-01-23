
import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer , ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts" , "train.csv")
    test_data_path: str = os.path.join("artifacts" , "test.csv")
    raw_data_path: str = os.path.join("artifacts" , "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            data = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info("Data loaded successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok= True)
            data.to_csv(self.ingestion_config.raw_data_path , index = False , header = True)

            logging.info("Training and Testing data split started")

            train_data , test_data = train_test_split(data , test_size = 0.2 , random_state = 42)

            train_data.to_csv(self.ingestion_config.train_data_path , index = False , header = True)

            test_data.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info("Training and Testing data split completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error("Error occured while ingesting data")
            raise CustomException(e , sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data , test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr , test_arr , _ = data_transformation.initiate_data_transformation(train_data , test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr , test_arr , )