import pickle, os, sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logging.info('Dumped the object into pickle file')
    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        report_data = []

        for i in range(len(list(models))):

            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(model, param, cv=5, scoring='r2')
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_model_score = r2_score(y_train, y_pred_train)
            test_model_score = r2_score(y_test, y_pred_test)
            cross_score = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            result = {
                'Model': model,
                'Train_Score': train_model_score,
                'Test_Score': test_model_score,
                'Cross_Val_Score': cross_score.mean()
            }

            report_data.append(result)

            report[list(models.keys())[i]] = test_model_score

        report_df = pd.DataFrame(report_data)
        report_df.to_csv('artifacts/model_results.csv', index=False)  
        return report
        
    except Exception as e:
        logging.error(f'Error occured: {e}')
        raise CustomException(e, sys)
        