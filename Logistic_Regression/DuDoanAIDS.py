import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('AIDS_Classification.csv').values

data_train, data_test = train_test_split(data, test_size=0.3,shuffle=False)

x_train, y_train = data_train[:,:22], data_train[:,22]
x_test, y_test = data_test[:,:22], data_test[:,22]

logis = LogisticRegression()
logis.fit(x_train,y_train)
y_pred = logis.predict(x_test)
w0 = logis.intercept_ 
w1 = logis.coef_[0]
print("MSE = ", mean_squared_error(y_test, y_pred))
print("MAE = ", mean_absolute_error(y_test, y_pred))
print(logis.predict([[948,2,48,89.8128,0,0,0,100,0,0,0,0,0,0,1,0,0,0,422,477,566,324]]))