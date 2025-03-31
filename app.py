from flask import Flask, render_template, request
import os 
import numpy as np 
import pandas as pd 
from SurgeSense.pipeline.prediction import PredictionPipeline


app=Flask(__name__)


@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/train',methods=['GET'])
def training():
    os.system('python main.py')
    return 'Training Successful!'

@app.route('/predict',methods=['GET','POST'])
def index():
    if request.method=='POST':
        try:
            distance=float(request.form['distance'])
            surge_multiplier=float(request.form['surge_multiplier'])
            temp=float(request.form['temp'])  
            clouds=float(request.form['clouds'])
            pressure=float(request.form['pressure']) 
            rain=float(request.form['rain'])
            humidity=float(request.form['humidity'])
            wind=float(request.form['wind'])
            # day=float(request.form['day'])
            hour=float(request.form['hour'])
            # month=float(request.form['month'])
            cab_type=str(request.form['cab_type'])
            destination=str(request.form['destination'])
            source=str(request.form['source'])
            name=str(request.form['name'])
        
            data=[
                distance,cab_type,destination,source,surge_multiplier,name,temp,clouds,pressure,rain,humidity,wind,hour
            ]
            columns = ['distance','cab_type','destination','source','surge_multiplier','name',
           'temp','clouds','pressure','rain','humidity','wind','hour']
            df=pd.DataFrame([data],columns=columns)
            obj=PredictionPipeline()
            predict=obj.predict(df)
            return render_template('results.html',prediction=str(round(predict[0],2)))
        
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
        
    else:
        render_template('index_html')



if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)
