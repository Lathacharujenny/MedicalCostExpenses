from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    results = None
    if request.method=='GET':
        return render_template('home.html', results=results)
    else:
        data = CustomData(
               brand = request.form.get('brand').lower(),
               battery_capacity = int(request.form.get('battery_capacity')),
               screen_size = float(request.form.get('screen_size')),
               touchscreen = request.form.get('touchscreen'),
               processor = int(request.form.get('processor')),
               ram = int(request.form.get('ram')),
               internal_storage = int(request.form.get('internal_storage')),
               operating_system = request.form.get('operating_system').lower(),
               wi_fi = request.form.get('wi_fi'),
               bluetooth = request.form.get('bluetooth'),
               gps = request.form.get('gps'),
               number_of_sims = int(request.form.get('number_of_sims')),
               threeg = request.form.get('threeg'),
               fourg_lte = request.form.get('fourg_lte'),
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
    app.run(host="0.0.0.0", debug=True) 
   