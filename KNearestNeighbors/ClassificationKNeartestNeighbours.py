import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])

XTrain, XTest, YTrain, YTest = model_selection.train_test_split(X, y, test_size=0.2)

classifier = neighbors.KNeighborsClassifier()
classifier.fit(XTrain, YTrain)
accuracy = classifier.score(XTest, YTest)

print(accuracy)

# exampleMeasures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
# exampleMeasures = exampleMeasures.reshape(1, -1)
exampleMeasures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
# Number of samples is 2 here.
exampleMeasures = exampleMeasures.reshape(len(exampleMeasures), -1)

prediction = classifier.predict(exampleMeasures)
print(prediction)

