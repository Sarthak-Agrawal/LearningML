from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import random

style.use('ggplot')
plot.switch_backend('TkAgg')


def createDataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val = val + step
        elif correlation and correlation == 'neg':
            val = val - step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def slope(xs, ys):
    m = ((mean(xs)*mean(ys))-mean(xs*ys))/((mean(xs)**2) - mean(xs*xs))
    return m


def yIntercept(xs, ys, m):
    b = mean(ys) - m*mean(xs)
    return b


def squaredError(ysOriginal, ysLine):
    return sum((ysOriginal-ysLine)*(ysOriginal-ysLine))


def coeffOfDetermination(ysOriginal, ysLine):
    yMean = [mean(ysOriginal) for y in ysOriginal]
    squaredErrorRegression = squaredError(ysOriginal, ysLine)
    squaredErrorMean = squaredError(ysOriginal, yMean)
    return 1 - (squaredErrorRegression / squaredErrorMean)

# xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

# xs, ys = createDataset(40, 80, 2, correlation='pos')
# xs, ys = createDataset(40, 10, 2, correlation='pos')
xs, ys = createDataset(40, 10, 2, correlation=False)

m = slope(xs, ys)
b = yIntercept(xs, ys, m)
# print(m)
# print(b)

regressionLine = [m*x + b for x in xs]

rSquare = coeffOfDetermination(ys, regressionLine)

print(rSquare)

predictX = 8
predictY = m*predictX + b
# print(predictY)
plot.scatter(xs, ys, color="Blue", label="Data")
plot.plot(xs, regressionLine, label="Best-fit line")
plot.scatter(predictX, predictY, label="Prediction", color="Green")
plot.legend(loc=4)
plot.show()
