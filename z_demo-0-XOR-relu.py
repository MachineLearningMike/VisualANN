import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation
import matplotlib.lines as lines

import Activations
import Layers
import Model
dense0 = Layers.Dense(2, input_dim=2, activation=Activations.tanh)
dense1 = Layers.Dense(1, activation=Activations.identity)
dense2 = Layers.Dense(1, activation=Activations.identity)
nn = Model.Sequential([dense0, dense1])

limL = -1.
limH = 2.
nContours = 7
step = 0.4
min = int(dense0.activation.eval(limL) - 1.)
max = int(dense0.activation.eval(limH) + 1.)
clabels0 = np.arange(min, max + step, step)
clabels1 = np.arange(min * 2, max * 2 + step, step) 

step = 0.2
x = np.arange(limL, limH+step, step).reshape(-1, 1)
y = np.arange(limL, limH+step, step).reshape(-1, 1)
X, Y = np.meshgrid(x, y)
XY = np.stack((X, Y), axis=-1)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9), sharex=True, sharey=True)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.25) #, wspace=0.05, hspace=0.001)

def draw_contour(ax, X, Y, Z, title):
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot([limL, limH], [0, 0], '-', color='r', lw=1)
    ax.plot([0, 0], [limL, limH], '-', color='r', lw=1)
    
    ax.plot([[0, 0], [1., 1.]], 'o', color='y', lw=0.2)
    ax.plot([[1., 0], [0., 1.]], 'x', color='y', lw=0.2)

    outputs = Z

    origin = 'lower'
    CS = ax.contourf(X, Y, outputs, nContours, cmap=plt.cm.bone, origin=origin)
    CS2 = ax.contour(X, Y, outputs, nContours, colors=('k'), origin=origin, linewidths=.2)
    CS3 = ax.contour(X, Y, outputs, 1, colors=('y'), origin=origin, linewidths=.5)    
    plt.clabel(CS2, fmt='%1.1f', colors='y', fontsize=6, inline=True)

Z1 = np.maximum(0., Y - X)
Z2 = np.maximum(0., X - Y)
Z3 = 2 * Z1 + 2 * Z2
Z4 = np.maximum(0., Z3 - 1.)

draw_contour(axes[0,0], X, Y, Z1, "Z1 = Relu( Y - X )")
draw_contour(axes[1,0], X, Y, Z2, "Z2 = Relu( X - Y )")
draw_contour(axes[0,1], X, Y, Z3, "Z3 = 2 * Z1 + 2 * Z2")
draw_contour(axes[1,1], X, Y, Z4, "Z4 = Relu( Z3 - 1.)")

plt.show()