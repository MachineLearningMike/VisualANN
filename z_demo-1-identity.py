import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation
import matplotlib.lines as lines

import Activations
import Layers
import Model
dense0 = Layers.Dense(2, input_dim=2, activation=Activations.identity)
dense1 = Layers.Dense(1, activation=Activations.identity)
dense2 = Layers.Dense(1, activation=Activations.identity)
nn = Model.Sequential([dense0, dense1])

lim = 3.
step = 0.5
min = int(dense0.activation.eval(-lim) - 1.) - step
max = int(dense0.activation.eval(lim) + 1.) + step
clabels0 = np.arange(min, max, step)
clabels1 = np.arange(min * 2, max * 2, step) 

x = np.arange(-lim, lim, 0.2).reshape(-1, 1)
y = np.arange(-lim, lim, 0.2).reshape(-1, 1)
X, Y = np.meshgrid(x, y)
XY = np.stack((X, Y), axis=-1)

fig, ax = plt.subplots(figsize=(16,9), sharex=True, sharey=True)
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9), sharex=True, sharey=True)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, wspace=0.1, hspace=0.25) #, wspace=0.05, hspace=0.001)

counts = [0] * 200 + list(range(2880))

def data_gen():
    for cnt in counts: # itertools.count():
        t0 = 2 * np.pi * cnt / 2880
        t1 = t0 + np.pi/2
        t2 = t1 + np.pi/2
        t3 = t2 + np.pi/2
        yield np.cos(t0), np.sin(t0), \
        np.cos(t1), np.sin(t1), \
        np.cos(t2), np.sin(t2), \
        np.cos(t3), np.sin(t3)


def run(data):
    # update the data
    a00, a01, a10, a11, _, _, _, _1 = data
    a02 = 0; a12 = 0
    nn.SetWeights([
        np.array(((a00, a01, a02), (a10, a11, a12))), 
        np.array(((1., 1., 0.),))
        ])
    Z = nn.Feedforwards(XY).squeeze()
    ax.cla()
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.plot([-lim, lim], [0, 0], color='r', lw=0.5)
    ax.plot([0, 0], [-lim, lim], color='r', lw=0.5)
    # ax.grid(True, which='major')

    outputs = nn.GetOutputs()

    xcolor = 'g'
    ycolor = 'b'
    contcolor = 'black'
    CS00 = ax.contour(X, Y, outputs[0][0,::], clabels0, linewidths=.5, linestyles='dashed', colors=xcolor)
    # ax.clabel(CS00, inline=True, fontsize=6, fmt="%1.1f")
    CS01 = ax.contour(X, Y, outputs[0][1,::], clabels0, linewidths=.5, linestyles='dashed', colors=ycolor)
    # ax.clabel(CS01, inline=True, fontsize=6, fmt="%1.1f")
    CS10 = ax.contour(X, Y, outputs[1][0,::], clabels1, linewidths=.2, colors=contcolor)
    ax.clabel(CS10, inline=True, fontsize=6, fmt="%1.1f")

    ax.plot([0, a00], [0, a01], color=xcolor, lw=0.8)
    ax.plot([0, a10], [0, a11], color=ycolor, lw=0.8)

    ax.set_title("z = {:s}( {:.2f} * x0 + {:.2f} * x1 + {:.1f} ) + {:s}( {:.2f} * x0 + {:.2f} * x1 + {:.1f})"
    .format(dense0.activation.name, a00, a01, a02, dense0.activation.name, a10, a11, a12))

    return

ani = animation.FuncAnimation(fig, run, data_gen, interval=0)
# ani = animation.FuncAnimation(fig, run, 2, interval=0)

plt.show()