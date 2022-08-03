# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 21:33:39 2022

@author: HP
"""

#Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#Load dataset
df = pd.read_excel('dataset.xlsx')

print(df)

#variabel independen
X=df['luasPanen(Ha)'].values.reshape(-1,1)
#variable dependen
y=df['produksi(Ton)']

#Split the dataset
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 15)

#Model
lr = LinearRegression()

#Fit the model
lr.fit(X_train,y_train)

#Evaluation Model
score = lr.score(X_train, y_train)  
print("Training score: %.2f " % score)

#Pickle file for model
pickle.dump(lr, open('regression.pkl','wb'))

model = pickle.load(open("regression.pkl", "rb")) 


