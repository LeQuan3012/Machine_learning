import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def MSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))

def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))

data = pd.read_csv('Milknew.csv').values

data_train, data_test = train_test_split(data, test_size=0.3,shuffle=False)

x_train, y_train = data_train[:,:7], data_train[:,7]
x_test, y_test = data_test[:,:7], data_test[:,7]

lrg = LinearRegression()

lrg.fit(x_train, y_train)

y_pred = lrg.predict(x_test)

plt.plot(y_test, 'r')
plt.plot(y_pred, 'b')

print("Hệ số w0:", lrg.intercept_)
print("Hệ số w1:", lrg.coef_)


print("MSE = ", MSE(y_test, y_pred))
print("MAE = ", MAE(y_test, y_pred))

test = lrg.predict([[8.5,70,10,10,10,10,2046]])
print(test) 