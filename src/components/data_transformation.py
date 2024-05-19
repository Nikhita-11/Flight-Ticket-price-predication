import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function
from src.utils import remove_outliers
from src.utils import data_transform


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Defining the categorical columns and the numerical columns
            # stops and class columns contain ordinal data 
            categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
            ordinal_cols = ['stops', 'class']
            onehot_cols = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city']
            numerical_cols = ['duration', 'days_left']

            #Defining the categories for ordinal columns
            stops_categories = ['zero', 'one', 'two_or_more']
            class_categories = ['Economy', 'Business']
            
            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Ordinal Pipeline
            ordinal_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[stops_categories, class_categories])),
                ('scaler',StandardScaler())
                ]

            )

            # Onehot pipeline
            onehot_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('ordinal_pipe', ordinal_pipeline, ordinal_cols),
            ('onehot_pipe', onehot_pipeline, onehot_cols)
            ])
            
            logging.info('Pipeline Completed')
            return preprocessor
            

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            #removing outliers from distance and calculated_duration 
            train_df = remove_outliers(train_df, columns=['duration', 'days_left', 'price'], threshold=1.5)
            test_df = remove_outliers(test_df, columns=['duration', 'days_left', 'price'], threshold=1.5)

            logging.info('EDA inside data transformation complete')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            target_column_name = 'price'
            drop_columns = [target_column_name,'flight','Unnamed: 0']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Transforming using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)