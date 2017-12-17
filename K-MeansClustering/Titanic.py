import matplotlib.pyplot as plot
from matplotlib import style
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

style.use('ggplot')
plot.switch_backend('TkAgg')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
'''
The dataframe *should* be read as numbers where there are numbers. For some reason, Pandas will seemingly randomly read 
some rows in columns as strings, despite even the strings being actual number. Makes no sense to me, so I just convert 
to numeric to be totally certain.
'''
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
# print(df.head())


def handleNonNumericalData(df):
    columns = df.columns.values
    for column in columns:
        textDigitVals = {}

        def convertToInt(val):
            return textDigitVals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            columnContents = df[column].values.tolist()
            uniqueElements = set(columnContents)
            x = 0
            for unique in uniqueElements:
                if unique not in textDigitVals:
                    textDigitVals[unique] = x
                    x += 1

            # The map() fn(x) applies a given to function to each item of an iterable and returns a list of the results.
            df[column] = list(map(convertToInt, df[column]))

    return df

df = handleNonNumericalData(df)
# print(df.head())
df.drop(['ticket'], 1, inplace=True)
x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

classifier = KMeans(n_clusters=2)
classifier.fit(x)

correct = 0
for i in range(len(x)):
    predictMe = np.array(x[i].astype(float))
    predictMe = predictMe.reshape(-1, len(predictMe))
    prediction = classifier.predict(predictMe)
    if y[i] == prediction:
        correct = correct + 1

print(max(correct/len(x), 1 - correct/len(x)))
