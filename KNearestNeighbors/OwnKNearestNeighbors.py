import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


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

    # print(Counter(votes).most_common(1))
    voteResult = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return voteResult, confidence


accuracies = []

for i in range(25):
    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # Converting data to float(some data was coming in form of string)
    fullData = df.astype(float).values.tolist()
    random.shuffle(fullData)

    testSize = 0.2
    trainSet = {
        2: [],
        4: []
    }
    testSet = {
        2: [],
        4: []
    }
    trainData = fullData[:-int(testSize*len(fullData))]
    testData = fullData[-int(testSize*len(fullData)):]

    for i in trainData:
        trainSet[i[-1]].append(i[:-1])

    for i in testData:
        testSet[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in testSet:
        for data in testSet[group]:
            vote, confidence = kNearestNeighbors(trainSet, data, k=5)
            if group == vote:
                correct += 1
            # else:
                # print(confidence)
            total += 1

    print("Accuracy : ", correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
