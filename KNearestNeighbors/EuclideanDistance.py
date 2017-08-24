from math import sqrt

p1 = [1, 3]
p2 = [2, 5]

dis = 0
for i in range(0, len(p1)):
    dis = dis + ((p1[i] - p2[i])**2)
dis = sqrt(dis)

print(dis)
