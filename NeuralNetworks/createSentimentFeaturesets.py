from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
numberOfLines = 10000000


def createLexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:numberOfLines]:
                allWords = word_tokenize(l.lower())
                lexicon += list(allWords)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Counter will return a dictionary of words with their number of occurrence
    wordCounts = Counter(lexicon)

    # Removing super common words and rare words from the lexicon
    l2 = []
    for w in wordCounts:
        if 50 < wordCounts[w] < 1000:
            l2.append(w)

    return l2


def sampleHandling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as file:
        contents = file.readlines()
        for l in contents[:numberOfLines]:
            currentWords = word_tokenize(l.lower())
            currentWords = [lemmatizer.lemmatize(i) for i in currentWords]
            features = np.zeros(len(lexicon))
            for word in currentWords:
                if word.lower() in lexicon:
                    index = lexicon.index(word.lower())
                    features[index] += 1

            featureset.append([list(features), classification])

    return featureset


def createFeaturesetsAndLabels(pos, neg, testSize=0.1):
    lexicon = createLexicon(pos, neg)
    features = []
    features += sampleHandling('data/pos.txt', lexicon, [1, 0])
    features += sampleHandling('data/neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testingSize = int(testSize * len(features))

    trainX = list(features[:, 0][:-testingSize])
    trainY = list(features[:, 1][:-testingSize])
    testX = list(features[:, 0][-testingSize:])
    testY = list(features[:, 1][-testingSize:])

    return trainX, trainY, testX, testY


if __name__ == '__main__':
    trainX, trainY,testX, testY = createFeaturesetsAndLabels(
        'data/pos.txt', 'data/neg.txt'
    )
    with open('data/sentimentSet.pickle', 'wb') as f:
        pickle.dump([trainX, trainY, testX, testY], f)
