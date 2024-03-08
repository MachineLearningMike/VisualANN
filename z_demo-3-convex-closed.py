import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation
import matplotlib.lines as lines

import Activations
import Layers
import Model
dense0 = Layers.Dense(3, input_dim=2, activation=Activations.tanh)
dense1 = Layers.Dense(1, activation=Activations.identity)
dense2 = Layers.Dense(1, activation=Activations.identity)
nn = Model.Sequential([dense0, dense1])

lim = 6.
step = 0.2
min = int(dense0.activation.eval(-lim) - 1.) - step
max = int(dense0.activation.eval(lim) + 1.) + step
clabels0 = np.arange(min, max, step)
clabels1 = np.arange(min * 2, max * 2, step) 

x = np.arange(-lim, lim, 0.2).reshape(-1, 1)
y = np.arange(-lim, lim, 0.2).reshape(-1, 1)
X, Y = np.meshgrid(x, y)
XY = np.stack((X, Y), axis=-1)

fig, ax = plt.subplots()

def data_gen():
    for cnt in itertools.count():
        t0 = 2 * np.pi * cnt / 2880
        t1 = t0 + np.pi/2 * 1.8
        t2 = t1 + np.pi/2 * 1.8
        t3 = t2 + np.pi/2 * 1.5
        yield np.cos(t0), np.sin(t0), \
        np.cos(t1), np.sin(t1), \
        np.cos(t2), np.sin(t2), \
        np.cos(t3), np.sin(t3)


def run(data):
    # update the data
    a00, a01, a10, a11, a20, a21, a30, a31 = data
    nn.SetWeights([
        np.array(((a00, a01, -1), (a10, a11, -1), (a20, a21, -1))), 
        np.array(((1., 1., 1., 0.),))
        ])
    Z = nn.Feedforwards(XY).squeeze()
    ax.cla()
    ax.plot([-lim, lim], [0, 0], color='r', lw=0.5)
    ax.plot([0, 0], [-lim, lim], color='r', lw=0.5)
    # ax.grid(True, which='major')

    outputs = nn.GetOutputs()

    CS00 = ax.contour(X, Y, outputs[0][0,::], clabels0)
    # ax.clabel(CS00, inline=True, fontsize=6, fmt="%1.1f")
    CS01 = ax.contour(X, Y, outputs[0][1,::], clabels0)
    # ax.clabel(CS01, inline=True, fontsize=6, fmt="%1.1f")
    CS10 = ax.contour(X, Y, outputs[1][0,::], clabels1)
    ax.clabel(CS10, inline=True, fontsize=6, fmt="%1.1f")

    ax.plot([0, a00], [0, a01], color='b', lw=0.8)
    ax.plot([0, a10], [0, a11], color='b', lw=0.8)

    ax.set_title("z = {:s} ( {:.2f} * x0 + {:.2f} * x1,  {:.2f} * x0 + {:.2f} * x1)"
    .format(dense1.activation.name, a00, a01, a10, a11))

    return

ani = animation.FuncAnimation(fig, run, data_gen, interval=0)
# ani = animation.FuncAnimation(fig, run, 2, interval=0)

plt.show()