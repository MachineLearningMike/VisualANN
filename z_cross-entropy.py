import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(1, 1, 1, projection='3d')

x = np.arange(.01, .99, .01).reshape(-1, 1)
y = np.arange(.01, .99, .01).reshape(-1, 1)
X, Y = np.meshgrid(x, y)
# XY = np.stack((X, Y), axis=-1)
# Z =  np.power(X-Y, 2)
Z = - Y * np.log(X) - (1.-Y) * np.log(1.-X)
Z1 = (1.-Y)/(1.-X) - Y/X 

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                       linewidth=0, antialiased=False, cmap=cm.PRGn) #gray)
# surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,
#                        linewidth=0, antialiased=False, cmap=cm.PRGn) #gray)

# ax.contourf(X, Y, Z, zdir='z')
# ax.contourf(X, Y, Z, zdir='x')
# ax.contourf(X, Y, Z, zdir='y')

ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.show()