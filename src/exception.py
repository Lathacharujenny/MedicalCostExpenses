import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    """
    Function to print detailed error message.
    """
    _,_,exc_tab = error_detail.exc_info()
    file_name = exc_tab.tb_frame.f_code.co_filename
    error_message = f"Error: {str(error)}\nFile: {file_name}\nLine: {exc_tab.tb_lineno}"
    print(error_message)

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
    if __name__ == '__main__':
        try:
            a=1/0
        except Exception as e:
            logging.error("An error occurred")  # Log the error using the logger module
            raise CustomException(e,sys) # type: ignore
    
    