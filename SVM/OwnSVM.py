import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np

style.use('ggplot')
plot.switch_backend('TkAgg')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plot.figure()
            self.axis = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        # data is the training set
        self.data = data
        # {|w|: [w,b] }
        opt_dict = {}
        transforms = [[1, 1],
                      [1, -1],
                      [-1, 1],
                      [-1, -1]]

        # yi is the class
        self.max_feature_value = -999999
        self.min_feature_value = 999999
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    if feature > self.max_feature_value:
                        self.max_feature_value = feature
                    if feature < self.min_feature_value:
                        self.min_feature_value = feature

        stepSizes = [self.max_feature_value*0.1, self.max_feature_value * 0.01, self.max_feature_value*0.001]

        # extremely expensive and not as important as w
        bRangeMultiple = 5
        bMultiple = 5

        # first element in vector w
        latestOptimum = self.max_feature_value*10

        for step in stepSizes:
            w = np.array([latestOptimum, latestOptimum])
            # because convex problem
            optimized = False
            while not optimized:
                pass

    def predict(self, features):
        # sign( x.w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        return classification


dataDict = {-1: np.array([[1, 7],
                          [2, 8],
                          [3, 8]]),

            1: np.array([[5, 1],
                         [6, -1],
                         [7, 3]])}


