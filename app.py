# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:58:11 2022

@author: HP
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app

app = Flask(__name__)
app.debug = True

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/pertanian")
def panen():
    return render_template("pertanian.html")

@app.route("/pendidikan")
def sekolah():
    return render_template("pendidikan.html")

@app.route("/umkm")
def umkm():
    return render_template("umkm.html")

@app.route("/database")
def database():
    return render_template("pertanian/database.html")

@app.route("/rumahbibit")
def rumahbibit():
    return render_template("pertanian/rumahbibit.html")   

@app.route("/mikroEfektif")
def mikroEfektif():
    return render_template("pertanian/mikroEfektif.html")   

@app.route("/rambutan")
def rambutan():
    return render_template("pertanian/rambutan.html")  

@app.route("/budidaya")
def budidaya():
    return render_template("pertanian/budidaya.html")  

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/profil")
def profil():
    return render_template("profil.html")

@app.route("/page")
def page():
    return render_template("404.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    return render_template('pertanian/database.html', prediction_text='Prediksi Total Produksi Padi : {} Ton'.format(output))

if __name__ == "__main__":
    app.run()
