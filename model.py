# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:33:39 2022

@author: HP
"""

#Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNN_Reg
import pickle

#Load dataset
df = pd.read_csv('sample.csv')

#print(df)

#variabel independen
X=df['luasPanen(Ha)'].values.reshape(-1,1)
#variable dependen
y=df['produksi(Ton)']

#Split the dataset
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 15)

#Model
model = KNN_Reg(n_neighbors = 3)

#Fit the model
model.fit(X_train,y_train)

#Evaluation Model
score = model.score(X_train, y_train)  
print("Training score: %.2f " % score)

#Pickle file for model
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open("model.pkl", "rb")) 


