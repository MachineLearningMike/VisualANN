import numpy as np
 
class Activation:
    name = 'activation'
    def eval(x):
        return None
    def gradient(x):
        return None

class identity(Activation):
    name = 'identity'
    def eval(x):
        return x
    def gradient(x):
        return np.ones_like(x)

class relu(Activation):
    name = 'relu'
    def eval(x):
        return np.maximum(np.zeros_like(x), x)
    def gradient(x):
        ZL = np.zeros_like(x)
        return np.maximum(ZL, x > ZL)

class sigmoid(Activation):
    name = 'sigmoid'
    def eval(x):
        OL = np.ones_like(x)
        return OL / (OL + np.exp(-x))
    def gradient(x):
        eval = sigmoid.eval(x)
        return eval * (np.ones_like(x) - eval)

class tanh(Activation):
    name = 'tanh'
    def eval(x):
        exp_plus = np.exp(x)
        exp_minus = np.exp(-x)
        return (exp_plus - exp_minus) / (exp_plus + exp_minus)
    def gradient(x):
        eval = tanh.eval(x)
        return (1. - pow(eval, 2))

class tanh_identity(Activation):
    alpha = 0.

    def eval(x):
        return tanh.eval(x) * tanh_identity.alpha + (1.-tanh_identity.alpha) * identity.eval(x)
    def gradient(x):
        return tanh.tangent(x) * tanh_identity.alpha + (1.-tanh_identity.alpha) * identity.tangent(x)

# tests -----------------------------------------
# import matplotlib.pyplot as plt

# fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)

# def show_activation(ax, activation, xlim=2., ylim=2.):
#     x = np.arange(-xlim - .1, xlim + .1, 0.01)
#     eval = activation.eval(x)
#     gradient = activation.gradient(x)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.plot([-xlim, xlim], [0, 0], color='k', lw=0.5)
#     ax.plot([0, 0], [-ylim, ylim], color='k', lw=0.5)
#     ax.plot(x, eval, lw=1)
#     ax.plot(x, gradient, lw=1)
#     ax.set_title(activation.name)

# show_activation(ax1, identity)
# show_activation(ax2, relu)
# show_activation(ax3, sigmoid, xlim=4.)
# show_activation(ax4, tanh)

# plt.subplots_adjust(hspace=0.4, wspace=0.4)

# plt.show()