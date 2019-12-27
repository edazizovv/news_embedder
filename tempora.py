import numpy
from matplotlib import pyplot


x0 = 1
x1 = 2
X = [x0, x1]

p = 2
N = 100

fi = [0.2, -0.7]
c = 0.1

for j in range(N):
    xk = c + X[j] * fi[0] + X[j + 1] * fi[1] + numpy.random.normal()
    X.append(xk)
    
del N, c, fi, j, p, x0, x1, xk

tt = list(range(len(X)))

pyplot.plot(tt, X)
