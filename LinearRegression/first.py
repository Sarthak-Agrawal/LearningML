import pandas as pd
import quandl
import numpy as np
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plot
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = 'usreNu1ssxsnR2H8CZco'
df = quandl.get('WIKI/GOOGL')
print(df.tail())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_%'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.00
df['Change_%'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00

df = df[['Adj. Close', 'HL_%', 'Change_%', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
# getting length of 1% of dataset
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
print(df.tail())
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

# X_lately is goinf to be the predicted labels features
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Removing NaN values
df.dropna(inplace=True)
Y = np.array(df['label'])
print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

# classifier = LinearRegression(n_jobs=-1)
# # Train
# classifier.fit(X_train, Y_train)
# # Test
# accuracy = classifier.score(X_test, Y_test)
# print("Linear Regression Accuracy:")
# print(accuracy)

pickle_in = open('linearRegression.pickle','rb')
classifier = pickle.load(pickle_in)


# prediction
forecast_set = classifier.predict(X_lately)
print(forecast_set)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    # print(next_date)
    next_unix += one_day
    # This line sets all the columns not a number except for the forecast column which is represented via [i]
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

plot.switch_backend('TkAgg')
# print(type(df['Adj. Close']))
df['Adj. Close'].plot()
df['Forecast'].plot()
plot.legend(loc=4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()

with open('linearRegression.pickle', 'wb') as f:
    pickle.dump(classifier, f)




# print("Support Vector Regression Accuracy:")
# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     classifier = svm.SVR(kernel=k)
#     # Train
#     classifier.fit(X_train, Y_train)
#     # Test
#     accuracy = classifier.score(X_test, Y_test)
#     print(k, ":",  accuracy)
