import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# This line added because the system is not recognizing the src module
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifacts_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'artifacts'))
    train_data_path: str = os.path.join(artifacts_dir, 'train.csv')
    test_data_path: str = os.path.join(artifacts_dir, 'test.csv')
    raw_data_path: str = os.path.join(artifacts_dir, 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')
        try:
            logging.info('Loading the data set')
            df = pd.read_csv('../../data/skewed_data.csv')
            os.makedirs(os.path.dirname(os.path.abspath(self.ingestion_config.raw_data_path)), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Data loaded successfully, about to split into train and test sets')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print('Train data saved at: ', train_data)
    print('Test data saved: ', test_data)


