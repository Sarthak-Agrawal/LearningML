import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plot
from matplotlib import style
from collections import Counter

from pip._vendor import colorama

style.use('fivethirtyeight')
plot.switch_backend('TkAgg')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
newFeatures = [5, 7]

for i in dataset:
    for j in dataset[i]:
        plot.scatter(j[0], j[1], s=50, color=i)

def kNearestNeighbors(data, predict, k=3):
    if len(data) >= k:
        # This means that the number of groups to which comparison is to be done is more than k
        warnings.warn("K is set to a value less than total voting groups!")

    distances = []
    for group in data:
        for features in data[group]:
            euclideanDistance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            distances.append([euclideanDistance, group])

    votes = []
    distances = sorted(distances)
    for i in distances[:k]:
        # adding the groups in votes
        votes.append(i[1])

    print(Counter(votes).most_common(1))
    voteResult = Counter(votes).most_common(1)[0][0]
    return voteResult


result = kNearestNeighbors(dataset, newFeatures, k=3)
print(result)

plot.scatter(newFeatures[0], newFeatures[1], s=30)
plot.show()
