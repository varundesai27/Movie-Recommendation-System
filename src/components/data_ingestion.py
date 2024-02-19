import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from pickle4 import pickle

@dataclass
class DataIngestionConfig:
    merged_data_path: str = os.path.join('artifacts','movies_data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # Reading the data
            movies_df = pd.read_csv('notebook/data/tmdb_5000_movies.csv')
            credits_df = pd.read_csv('notebook/data/tmdb_5000_credits.csv')
            logging.info("Read Data Successfully")

            merge_data = pd.merge(movies_df,credits_df,on='title')
            logging.info("Datasets merged successfully")

            #Creating Data
            os.makedirs(os.path.dirname(self.ingestion_config.merged_data_path),exist_ok=True)

            merge_data.to_csv(self.ingestion_config.merged_data_path,index=False,header=True)

            logging.info("Data Ingestion is Completed")

            return self.ingestion_config.merged_data_path

        except Exception as e:
            raise CustomException(sys,e)