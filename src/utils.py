import os 
import sys 
import pickle 
import warnings
from sqlalchemy import create_engine
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging


@dataclass
class ConnectDBConfig(): 
        host = 'localhost'
        user = 'root'
        password = 'Sagnik123#'
        database = 'flights_data'
        table_name = 'flights'
        dataset_path:str = os.path.join('dataset', 'flights_data.csv')

class ConnectDB():     
    def __init__(self):
         self.connect_db_config = ConnectDBConfig()  
      
    def retrieve_data(self):
        try:
            logging.info('Initiating Database Connection')
            engine = create_engine(f'mysql+mysqlconnector://{self.connect_db_config.user}:{self.connect_db_config.password}@{self.connect_db_config.host}/{self.connect_db_config.database}')
            query = f"SELECT * From {self.connect_db_config.table_name}"
            df = pd.read_sql(query, engine)
            os.makedirs(os.path.dirname(self.connect_db_config.dataset_path),exist_ok=True)
            df.to_csv(self.connect_db_config.dataset_path,index=False)
            logging.info('Copy of Dataset stored in dataset folder as a csv file')
        except Exception as e: 
            raise CustomException(e,sys)
        finally:
                engine.dispose()
                logging.info('Database connection closed')
        
def save_function(file_path, obj): 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open (file_path, "wb") as file_obj: 
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)         

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
# Train models
            model.fit(X_train, y_train)
# Test data
            y_test_pred = model.predict(X_test)
            #R2 Score 
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)

# Function to load a particular object 
def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)


def remove_outliers(df, columns, threshold):
   
    #Making a copy of the dataframe first
    df_cleaned = df.copy()
    
    for col in columns:
        # Calculating the quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        
        # Calculating the IQR
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR            
        upper_bound = Q3 + threshold * IQR
        

        df_cleaned = df_cleaned[(df_cleaned[col] > lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned