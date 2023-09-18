import pickle
from flask import Flask , jsonify , render_template,request

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


#Import Lassocv and StandardScaler Pickle FiLe :

lasso = pickle.load(open('models/lassocv.pkl' ,'rb') )
scaler = pickle.load(open('models/scaler.pkl' ,'rb') )

#Create Homepage : 
@app.route("/")
def homepage():
    return render_template('index.html')

#Create Prediction Page:
@app.route("/result" , methods = ['POST' , 'GET'])

def predict():
    if request.method == "POST":
         Temperature=float(request.form.get('Temperature'))
         RH = float(request.form.get('RH'))
         Ws = float(request.form.get('Ws'))
         Rain = float(request.form.get('Rain'))
         FFMC = float(request.form.get('FFMC'))
         DMC = float(request.form.get('DMC'))
         ISI = float(request.form.get('ISI'))
         Classes = float(request.form.get('Classes'))
         Region = float(request.form.get('Region'))

         new_data_scaled=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
         result=lasso.predict(new_data_scaled)

         return render_template('home.html',result=result[0])
        
    else:
        return render_template("home.html")


if __name__ == '__main__':
    app.run(host = "0.0.0.0")
