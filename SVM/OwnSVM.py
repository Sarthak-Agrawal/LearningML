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

        for yi in self.data:
            for xi in self.data[yi]:
                print(xi, ": ", (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # classification = sign( x.w + b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.axis.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        for i in dataDict:
            for x in dataDict[i]:
                self.axis.scatter(x[0], x[1], s=100, color=self.colors[i])

        # hyperplane=x.w + b
        # v = x.w + b
        # psv = 1, nsv = -1, decisionBoundary = 0
        def hyperplane(x, w, b, v):
            # Given the x point determine the y point
            # x,y is an unknown point on the hyperplane
            # x_v and w_v are the vector
            # x_v= [x,y]
            # x_v.w_v+b =1 for postive sv
            # x.w[0] + y.w[1] + b =1
            # y = -x.w[0] - b + 1 / w[1]
            # So we get our y co-ordinate to plot it.
            return (-w[0]*x - b + v)/w[i]

        # dataranges for graphs
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hypXMin = datarange[0]
        hypXMax = datarange[1]

        # positive support vector hyperplane
        # x.w + b = 1
        psv1 = hyperplane(hypXMin, self.w, self.b, 1)
        psv2 = hyperplane(hypXMax, self.w, self.b, 1)
        self.axis.plot([hypXMin, hypXMax], [psv1, psv2], 'k')

        # negative support vector hyperplane
        # x.w + b = -1
        nsv1 = hyperplane(hypXMin, self.w, self.b, -1)
        nsv2 = hyperplane(hypXMax, self.w, self.b, -1)
        self.axis.plot([hypXMin, hypXMax], [nsv1, nsv2], 'k')

        # decision boundary hyperplane
        # x.w + b = 0
        db1 = hyperplane(hypXMin, self.w, self.b, 0)
        db2 = hyperplane(hypXMax, self.w, self.b, 0)
        self.axis.plot([hypXMin, hypXMax], [db1, db2], 'y--')

        plot.show()



dataDict = {-1: np.array([[1, 7],
                          [2, 8],
                          [3, 8]]),

            1: np.array([[5, 1],
                         [6, -1],
                         [7, 3]])}

svm = SupportVectorMachine()
svm.fit(data=dataDict)

predict = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]

for point in predict:
    svm.predict(point)

svm.visualize()
