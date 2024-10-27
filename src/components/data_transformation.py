import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import boxcox

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def skew_log_transform(self,x):
        return np.log1p(x)
    
    def skew_boxcox_transform(self,x):
        #return np.array([boxcox(col + 1)[0] for col in x.T]).T
        # Check for constant columns and apply Box-Cox only on non-constant columns
        transformed = []
        for col in x.T:
            if np.var(col) == 0:
                # If the column is constant, append the original column or handle accordingly
                transformed.append(col)  # or use np.zeros_like(col) or some other method
            else:
                transformed.append(boxcox(col + 1)[0])  # Use boxcox transformation
        return np.array(transformed).T
    
    def skew_power_transform(self,x):
        return x ** 4

    def get_data_transformer_object(self):
        '''
        This function si responsible for data transformation
        
        '''
        try:
            logging.info('Creating the Pipeline')

            log_columns = ['Internal_storage(GB)']

            boxcox_columns = ['Rear_Camera(MP)', 'Front_Camera(MP)', 'RAM(GB)']

            power_columns = ['Number of SIMs']

            numerical_columns = ['Battery_capacity(mAh)', 'Screen_size(inches)', 'Processor',
                                 'Resolution_width(px)', 'Resolution_height(px)']
            
            categorical_columns =  ['Brand', 'Touchscreen', 'Operating system', 'Wi-Fi',
                             'Bluetooth', 'GPS', '3G', '4G/ LTE']
            

            log_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('log_transform', FunctionTransformer(self.skew_log_transform)),
                    ('scaler', StandardScaler())
                ]
            )

            boxcox_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('boxcox_transform', FunctionTransformer(self.skew_boxcox_transform)),
                    ('scaler', StandardScaler())
                ]
            )

            power_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('power_transform', FunctionTransformer(self.skew_power_transform)),
                    ('scaler', StandardScaler())
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('log_pipeline', log_pipeline, log_columns),
                    ('boxcox_pipeline', boxcox_pipeline, boxcox_columns),
                    ('power_pipeline', power_pipeline, power_columns),
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info('Entered initiate_data_transformation')
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
           
            logging.info("Read train and test data completed")

            train_df['Price'] = np.log1p(train_df['Price'])
            test_df['Price'] = np.log1p(test_df['Price'])
            logging.info(f'Skew of price train data {train_df["Price"].skew()}')
            logging.info(f'Skew of price test data {test_df["Price"].skew()}')

            
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Price"

            X_train_df = train_df.drop(columns=[target_column_name], axis=1)
            y_train_data = train_df[target_column_name]


            X_test_df = test_df.drop(columns=[target_column_name], axis=1)
            y_test_data = test_df[target_column_name]

            logging.info(
            f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            processed_train_data = preprocessing_obj.fit_transform(X_train_df)
            processed_test_data = preprocessing_obj.transform(X_test_df)


            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                processed_train_data,
                y_train_data,
                processed_test_data,
                y_test_data,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            raise CustomException(e,sys) 
