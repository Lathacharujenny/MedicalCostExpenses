import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
         
         model_path=os.path.join("artifacts","model.pkl")
         preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
         print("Before Loading")
         model=load_object(file_path=model_path)
         print('The model is : ', model)
         preprocessor=load_object(file_path=preprocessor_path)
         print("After Loading")
         data_scaled=preprocessor.transform(features)
         print('The scaled data is : ', data_scaled.T)
         preds=model.predict(data_scaled)
         return preds
    
class CustomData:
    def __init__(self,
                 brand: str,
                 battery_capacity: int,
                 screen_size: float,
                 touchscreen: str,
                 processor: int,
                 ram: int,
                 internal_storage: int,
                 operating_system: str,
                 wi_fi: str,
                 bluetooth: str,
                 gps: str,
                 number_of_sims: int,
                 threeg: str,
                 fourg_lte: str,
                 resolution_width: int,
                 resolution_height: int,
                 rear_camera: int,
                 front_camera: int
                 ):
        self.brand = brand
        self.battery_capacity = battery_capacity
        self.screen_size = screen_size
        self.touchscreen = touchscreen
        self.processor = processor
        self.ram = ram
        self.internal_storage = internal_storage
        self.operating_system = operating_system
        self.wi_fi = wi_fi
        self.bluetooth = bluetooth
        self.gps = gps
        self.number_of_sims = number_of_sims
        self.threeg = threeg
        self.fourg_lte = fourg_lte
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.rear_camera = rear_camera
        self.front_camera = front_camera

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Brand' : [self.brand],
                'Battery_capacity(mAh)' : [self.battery_capacity],
                'Screen_size(inches)' : [self.screen_size],
                'Touchscreen' : [self.touchscreen],
                'Processor' : [self.processor],
                'RAM(GB)' : [self.ram],
                'Internal_storage(GB)' : [self.internal_storage],
                'Operating system' : [self.operating_system],
                'Wi-Fi' : [self.wi_fi],
                'Bluetooth' : [self.bluetooth],
                'GPS' : [self.gps],
                'Number of SIMs' : [self.number_of_sims],
                '3G' : [self.threeg],
                '4G/ LTE' : [self.fourg_lte],
                'Resolution_width(px)': [self.resolution_width],
                'Resolution_height(px)' : [self.resolution_height],
                'Rear_Camera(MP)' : [self.rear_camera],
                 'Front_Camera(MP)' : [self.front_camera]
                  }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
             raise CustomException(e, sys)

        