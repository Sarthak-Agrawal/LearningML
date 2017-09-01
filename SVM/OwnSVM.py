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
        optDict = {}
        transforms = [[1, 1],
                      [1, -1],
                      [-1, 1],
                      [-1, -1]]

        # finding minmum and maximum feature
        self.max_feature_value = -999999
        self.min_feature_value = 999999
        for yi in self.data:
            # yi is the class i.e. 1 or -1 class here
            for featureset in self.data[yi]:
                # featureset is the point
                for feature in featureset:
                    # feature is each axis coordinate (here first x then y)
                    if feature > self.max_feature_value:
                        self.max_feature_value = feature
                    if feature < self.min_feature_value:
                        self.min_feature_value = feature

        # support vectors yi(xi.w+b) = 1

        stepSizes = [self.max_feature_value*0.1, self.max_feature_value * 0.01, self.max_feature_value*0.001]

        # extremely expensive and not as important as w
        bRangeMultiple = 5
        bMultiple = 5

        # first element in vector w
        latestOptimum = self.max_feature_value*10

        for step in stepSizes:
            w = np.array([latestOptimum, latestOptimum])
            # The optimized var will be true when we have checked all steps down to the base of the convex shape.
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value*bRangeMultiple), self.max_feature_value*bRangeMultiple,
                                   step*bMultiple):
                    for transformation in transforms:
                        wt = w*transformation
                        foundOption = True
                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(wt, xi) + b) >= 1:
                                    # This is the case when instances of both classes lie on the same side of the
                                    # hyperplane
                                    foundOption = False
                                    break

                        if foundOption:
                            optDict[np.linalg.norm(wt)] = [wt, b]

                # When iterated through all the bs
                if w[0] < 0:
                    optimized = True
                    print("Optimized a step.")
                else:
                    w = w - step

                modWs = sorted([n for n in optDict])
                # min |w|
                optimumChoice = optDict[modWs[0]]
                self.w = optimumChoice[0]
                self.b = optimumChoice[1]
                latestOptimum = optimumChoice[0][0] + step*2

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


