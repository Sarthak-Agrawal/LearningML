import pandas as pd
import quandl
import numpy as np
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'usreNu1ssxsnR2H8CZco'
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_%'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low'] * 100.00
df['Change_%'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.00

df = df[['Adj. Close', 'HL_%', 'Change_%', 'Adj. Volume']]
# print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
# getting length of 1% of dataset
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.tail())
# Removing NaN values
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
Y = np.array(df['label'])
X = preprocessing.scale(X)

print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

classifier = LinearRegression(n_jobs=-1)
# Train
classifier.fit(X_train, Y_train)
# Test
accuracy = classifier.score(X_test, Y_test)
print("Linear Regression Accuracy:")
print(accuracy)

classifier = svm.SVR(kernel='poly')
# Train
classifier.fit(X_train, Y_train)
# Test
accuracy = classifier.score(X_test, Y_test)
print("Support Vector Regression Accuracy:")
print(accuracy)
