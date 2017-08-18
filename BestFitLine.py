from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style

style.use('ggplot')
plot.switch_backend('TkAgg')


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

xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

m = slope(xs, ys)
b = yIntercept(xs, ys, m)
print(m)
print(b)

regressionLine = [m*x + b for x in xs]

rSquare = coeffOfDetermination(ys, regressionLine)

print(rSquare)

# predictX = 7
# predictY = m*predictX + b
# print(predictY)
# plot.scatter(xs, ys, color="Blue", label="Data")
# plot.plot(xs, regressionLine, label="Best-fit line")
# plot.scatter(predictX, predictY, label="Prediction", color="Green")
# plot.legend(loc=4)
# plot.show()
