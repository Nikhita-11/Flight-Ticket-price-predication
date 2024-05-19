import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, airline:str,
                     source_city:str,
                     destination_city:str, 
                     departure_time:str,
                     arrival_time:str,
                     stops:str, 
                     Class:str, 
                     duration:float, 
                     days_left:int): 
             self.airline = airline
             self.source_city = source_city
             self.destination_city = destination_city
             self.departure_time = departure_time 
             self.arrival_time = arrival_time
             self.stops = stops
             self.Class = Class 
             self.duration = duration
             self.days_left = days_left 

        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = { 
                       'airline': [self.airline], 
                       'source_city': [self.source_city], 
                       'destination_city': [self.destination_city],
                       'departure_time':[self.departure_time],
                       'arrival_time':[self.arrival_time], 
                       'stops': [self.stops], 
                       'class': [self.Class], 
                       'duration': [self.duration],
                       'days_left': [self.days_left]
                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")

                  return df
             
             except Exception as e:
                  logging.info("Error occured in get function in prediction_pipeline")
                  raise CustomException(e,sys)       