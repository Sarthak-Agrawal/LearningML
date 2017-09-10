import matplotlib.pyplot as plot
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')
plot.switch_backend('TkAgg')

x = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plot.scatter(x[:, 0], x[:, 1], s=150)
# plot.show()

classifier = KMeans(n_clusters=2)
classifier.fit(x)

centroids = classifier.cluster_centers_
# label will be either 0 or 1
labels = classifier.labels_

colors = 10 * ["g.", "r.", "c.", "y.", "k."]

for i in range(len(x)):
    plot.plot(x[i][0], x[i][1], colors[labels[i]], markersize=20)

plot.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=150, linewidths=5)
plot.show()
