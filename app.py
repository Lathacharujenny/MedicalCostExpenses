
from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():

    if request.method=='GET':
        return render_template('home.html', results=None)
    else:
        data = CustomData(
               brand = request.form.get('brand').lower(),
               battery_capacity = int(request.form.get('battery_capacity')),
               screen_size = float(request.form.get('screen_size')),
               processor = int(request.form.get('processor')),
               ram = int(request.form.get('ram')),
               internal_storage = int(request.form.get('internal_storage')),
               operating_system = request.form.get('operating_system').lower(),
               number_of_sims = int(request.form.get('number_of_sims')),
               resolution_width = int(request.form.get('resolution_width')),
               resolution_height = int(request.form.get('resolution_height')),
               rear_camera = int(request.form.get('rear_camera')),
               front_camera = int(request.form.get('front_camera'))
        )

        pred_df = data.get_data_as_data_frame()
        print('The data to be predicted: ', pred_df.T)

        predict_pipeline = PredictPipeline()
        print('Got the predict_pipeline')
        log_results = predict_pipeline.predict(pred_df)
        print('Log results: ', log_results)
        results = np.expm1(log_results)
        print('The result is : ', results)
        return render_template('home.html', results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")
    # app.run(host="0.0.0.0", debug=True)  use this when to do debug 
   