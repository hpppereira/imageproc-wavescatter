import numpy as np

x = np.random.rand(20)
y = np.random.rand(20)

# metodo 1

xm = np.mean(x)
ym = np.mean(y)

dist = np.sqrt((x - xm)**2 + (y - ym)**2)

var1 = np.mean(dist**2)

# metodo 2

dist_x = ((x - xm)**2).mean()
dist_y = ((y - ym)**2).mean()

dist = np.sqrt((dist_x + dist_y))

var2 = np.mean(dist**2)


print (var1, var2)

