# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:58:11 2022

@author: HP
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Create flask app

app = Flask(__name__)

#Load the pickle model
model = pickle.load(open("regression.pkl", "rb")) 

@app.route("/")
def Home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='Prediksi Total Produksi Padi : {} Ton'.format(output))

if __name__ == "__main__":
    app.run(debug=True)