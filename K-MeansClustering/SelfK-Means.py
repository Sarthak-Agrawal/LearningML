import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np

style.use('ggplot')
plot.switch_backend('TkAgg')

x = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

colors = 10*["g", "r", "c", "b", "k"]

# plot.scatter(x[:, 0], x[:, 1], s=30)
# plot.show()


class SelfKMeans:
    def __init__(self, k=2, tolerance=0.0001, maxIterations=300):
        self.k = k
        self.tolerance = tolerance
        self.maxIterations = maxIterations

    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            # centroids are in the form of {0:centroid1 , 1:centroid2}
            self.centroids[i] = data[i]

        for i in range(self.maxIterations):
            self.classifications = {}

            # created k classification lists
            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = []
                for centroid in self.centroids:
                    distances.append(np.linalg.norm(featureset-self.centroids[centroid]))
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # creates previous Centroids in the same manner as self.centroids
            prevCentroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # if any centroid changes more than tolerance, then optimize more ultil loop expires
            optimized = True

            for c in self.centroids:
                # self.centroids[c] is a list containing the x and y coordinates
                if self.tolerance < np.sum((self.centroids[c]-prevCentroids[c])/prevCentroids[c] * 100):
                    print((self.centroids[c]-prevCentroids[c])/prevCentroids[c] * 100)
                    optimized = False
                    break

            if optimized:
                break

    def predict(self, feature):
        distances = []
        for centroid in self.centroids:
            distances.append(np.linalg.norm(feature - self.centroids[centroid]))
        classification = distances.index(min(distances))
        return classification

classifier = SelfKMeans()
classifier.fit(x)

for centroid in classifier.centroids:
    plot.scatter(classifier.centroids[centroid][0], classifier.centroids[centroid][1], color="black", s=40, marker='x')

for classification in classifier.classifications:
    color = colors[classification]
    for featureset in classifier.classifications[classification]:
        plot.scatter(featureset[0], featureset[1], s=30, color=color, marker='o')

unknowns = np.array([[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])

for point in unknowns:
    classification = classifier.predict(point)
    plot.scatter(point[0], point[1], s=40, color=colors[classification], marker='*')

plot.show()
