import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model, save_object
from sklearn.metrics import r2_score



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train_data, y_train_data, X_test_data, y_test_data):
        try:
            logging.info('Split training and testing data')
            X_train, y_train, X_test, y_test = (X_train_data, y_train_data, X_test_data, y_test_data)
            models = {
                'Linear Regression': LinearRegression(),
                'Lasso Regression':  Lasso(),
                'Ridge Regression': Ridge(),
                'KNeighbors Regression': KNeighborsRegressor(),
                'Decision Tree':  DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'AdaBoost': AdaBoostRegressor(random_state=42),
                'Gradient Boost': GradientBoostingRegressor(random_state=42),
                'Xgboost': XGBRegressor(random_state=42)
            }

            params = {
                'Linear Regression': {},
                'Lasso Regression': {
                    'alpha': list(range(1,100))
                },
                'Ridge Regression': {
                    'alpha': list(range(1,100))
                },
                'KNeighbors Regression':{
                    'n_neighbors': list(range(1,100))
                },
                'Decision Tree':{
                    'max_depth':list(range(1,10)),
                    'criterion': ['squared_error', 'absolute_error']
                },
                'Random Forest': {
                    'n_estimators': list(range(1,10)),
                    'max_depth': list(range(1,10)),
                    'criterion': ['squared_error', 'absolute_error']
                },
                'AdaBoost':{
                    'n_estimators': list(range(1,10)),
                    'learning_rate': [0.1,0.5,1]
                },
                'Gradient Boost':{
                    'n_estimators': list(range(1,10)),
                    'learning_rate': [0.1,0.5,1]
                },
                'Xgboost':{
                    'n_estimators': list(range(1,10)),
                    'learning_rate': [0.1,0.5,1],
                    'gamma': [0,0.1,0.5,1]
                }
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            #print(model_report)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
        
            logging.info(f'Best_model_name: {best_model_name}')


            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            logging.error(f'Error Occured: {e}')
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
    print('Everything is ok')




