import numpy as np

from Activations import identity
from Neuron import neuron

rng = np.random.default_rng()

class Layer:
    hasWeights = None

    def __init__(self, dim, input_dim, activation=identity) -> None:
        self.dim = dim
        self.neurons, self.input_dim = self.setNeurons(dim, input_dim)         
        self.activation = activation

    def setNeurons(self, dim, input_dim) -> None:
        self.neurons = [neuron(input_dim=0, weights=rng.normal(0,1,input_dim), 
        bias=(rng.normal(0,1,1)[0])) for _ in range(dim)]    # must be a list but a tuple generator  
        self.input_dim = input_dim
        return self.neurons, self.input_dim
        # self.neurons = [neuron(...) for i in range(dim)]
        # for i in range(dim):
        #     self.neurons.append(neuron(...))

    def GetWeights(self):
        # nr.GetWeights(): shape = (input_dim+1,)
        totalWeights = np.stack([nr.GetWeights() for nr in self.neurons], axis=0) # must be a list but a tuple generator
        return totalWeights

    def SetWeights(self, totalWeights) -> None:
        # assert totalWeights.shape == (self.dim, self.input_dim)
        for i in range(self.dim):
            self.neurons[i].SetWeights(totalWeights[i])

    def GetGradients(self):
        # nr.GetGradients(): shape = BS + (nr.input_dim+1,)
        wFullGradients = np.stack(
            [nr.GetGradients() for nr in self.neurons], 
            axis=-2) # must be a list but a tuple generator
        return wFullGradients

    # def setWFullGradients(self, wFullGradients) -> None:
    #     assert wFullGradients.shape == (self.dim, self.input_dim)
    #     for i in range(self.dim):
    #         self.neurons[i].setWFullGradients(wFullGradients[i])

    def GetOutputs(self):
        outputs = np.stack(
            [nr.output for nr in self.neurons], # np.output.shape = BS
            axis=-1) # must be a list but a tuple generator
        return outputs

    def Feedforwards(self, x):
        return None

    def Backpropagate(self, gradient):
        return None

class Dense(Layer):
    hasWeights = True

    def __init__(self, dim, input_dim=None, activation = identity) -> None:
        super(Dense, self).__init__(dim, input_dim, activation)

    def Feedforwards(self, x):
        """
        Feeds x forwards through the neurons.
        x.shape: BS + (input_dim,)
        return.shape: BS + (dim,)
        """
        assert self.input_dim == x.shape[-1]
        outputs = np.stack(
            [nr.Feedforwards(x, self.activation) for nr in self.neurons]
            , axis=-1)
        return outputs

    def Backpropagate(self, gradient):
        """
        Propagates gradient backwards for the neurons.
        gradient.shape: BS + (dim,)
        return.shape: BS + (input_dim,)
        """
        assert self.dim == gradient.shape[-1]
        outputs = np.sum(
            np.stack(
                [self.neurons[i].Backpropagate(gradient[...,i], self.activation) 
                for i in range(self.dim)], 
                axis=0),
            axis=0
            # whatever axis number is allocated to the self.dim dimension, 
            # we must (reduce) sum up on the self.dim dimension and keep the self.input_dim dimension.
        )
        return outputs


class Softmax:
    hasWeights = False

    def __init__(self) -> None:
        pass

    def GetOutputs(self):
        return self.outputs

    def Feedforwards(self, x):
        # add checking x.shape[-1]
        self.x = x  # BS + (input_dim,)
        self.exp_x = np.power( np.e, x )    # BS + (input_dim,)
        self.exp_sum = np.sum( self.exp_x, axis=-1 )    # BS
        self.outputs = np.stack( 
            [ self.exp_x[...,i] / self.exp_sum for i in range(x.shape[-1]) ], 
            axis=-1 )
        return self.outputs   # BS + (input_dim,)

    def Backpropagate(self, gradient):
        # add checking ce.shape[-1]
        assert gradient.shape == self.x.shape
        grad = self.gradient()
        grad = np.stack(
            [grad[i, :, :] @ gradient[i] for i in range(self.x.shape[-1])], 
            axis=-1) # BS
        return grad

    def gradient(self):
        square_exp_sum = np.power(self.exp_sum, 2)  # BS
        grad = np.stack([
            np.stack([- self.exp_x[..., k] * self.exp_x[..., i] \
                / square_exp_sum for k in range(self.x.shape[-1])], 
                axis=-1)
            for i in range(self.x.shape[-1])], 
            axis=-1)  # BS + (input_dim,) + (input_dim,) = BS + (input_dim, input_dim)
        for k in range(self.x.shape[-1]):
            # It took damned TWO full days to find and replace buggy "self.x" with "self.exp_x".
            grad[..., k, k] = self.exp_x[..., k] * ( self.exp_sum - self.exp_x[..., k] ) \
                / square_exp_sum
        return grad # BS + (input_dim, input_dim).   d (prob_i) / d (x_j)



# tests --------------------------------------------

# print("OOOOOOOO  layer1 = Dense(1, input_dim=2)")
# layer1 = Dense(1, input_dim=2)

# i = 0
# for nr in list(layer1.neurons):
#     nr.Initialize(0, (i*2, i*2+1), np.array((i)))
#     print("nr.weights: ", nr.weights, "nr.bias:", nr.bias)
#     i += 1

# print('1: ', layer1.Feedforwards(np.array((1,2))))   # BS == (), shape: BS + (dim,) = (1,)
# print('2: ', layer1.Backpropagate(np.array((1,))))   # BS == (), shape: BS + (2,) = (2,)
# print('3: ', layer1.Feedforwards(np.array((2,3))))   # BS == (), shape == BS + (1,)
# print('4: ', layer1.Backpropagate(np.array((2,))))  # BS == (), shape == BS + (2,) = (2,)

# print('5(1,3,1): ', layer1.Feedforwards(np.array(((1,2),(2,3),(1,2)))))      # BS == (3,), shape == (3,) + (1,) 
# print('6(2,4,2): ', layer1.Backpropagate(np.array(((1,),(2,),(1,)))))  # BS == (3,), shape == (3, 2)
# # print('7: ', layer1.Feedforwards(np.array((((1,2),(2,3)),((2,3),(1,2))))))   # BS == (2,2), shape == (2,2) + (1,)
# # print('8: ', layer1.Backpropagate(np.array(((((1,),(2,)),((2,),(3,)))))))   # BS == (2,2), shape == (2,2) + (2,) 


# print("OOOOOOOO  layer1 = Dense(3, input_dim=2)")
# layer1 = Dense(3, input_dim=2)

# # print(len(layer1.neurons))
# i = 0
# for nr in list(layer1.neurons):
#     nr.Initialize(0, (i*2, i*2+1), np.array((i)))
#     print("nr.weights: ", nr.weights, "nr.bias:", nr.bias)
#     i += 1

# print('1: ', layer1.Feedforwards(np.array((1,2))))   # BS == (), shape == () + (3,)
# print('2: ', layer1.Backpropagate(np.array((1,2,3))))   # BS == (), shape == () + (2,)
# print('3: ', layer1.Feedforwards(np.array((2,3))))   # BS == (), shape == () + (3,)
# print('4: ', layer1.Backpropagate(np.array((2,3,4))))   # BS == (), shape == () + (2,)

# print('5(1,3,1): ', layer1.Feedforwards(np.array(((1,2),(2,3),(1,2)))))  # BS == (3,), shape == (3,) + (3,)
# print('6(2,4,2): ', layer1.Backpropagate(np.array(((1,2,3),(2,3,4),(1,2,3)))))  # BS == (3,), shape == (3,) + (2)
# print('7: ', layer1.Feedforwards(np.array((((1,2),(2,3)),((2,3),(1,2))))))   # BS == (2,2), shape == (2,2,3)
# print('8: ', layer1.Backpropagate(np.array((((1,2,3),(2,3,4)),((2,3,4),(1,2,3)))))) # BS == (2,3), shape == (2,2) + (2,)

#---------------------------------------------------------------
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import itertools
# import matplotlib.animation as animation
# import matplotlib.lines as lines

# import Activations
# nn = Dense(2, input_dim=2, activation=Activations.tanh)
# nn.SetWeights(np.array(((1, 2, 0.),)*2))
# print(nn.GetWeights())

# lim = 3.
# x = np.arange(-lim, lim, 0.2).reshape(-1, 1)
# y = np.arange(-lim, lim, 0.2).reshape(-1, 1)
# X, Y = np.meshgrid(x, y)
# XY = np.stack((X, Y), axis=-1)

# fig, ax = plt.subplots()

# def data_gen():
#     for cnt in itertools.count():
#         t = 2 * np.pi * cnt / 2880
#         yield np.cos(t), np.sin(t), np.cos(t+np.pi/2), np.sin(t+np.pi/2)

# clabels = np.arange(-1., 1., 0.1)

# def run(data):
#     # update the data
#     a00, a01, a10, a11 = data
#     nn.SetWeights(np.array(((a00, a01, 0.),(a10, a11, 0.))))
#     Z = nn.Feedforwards(XY)
#     ax.cla()
#     ax.plot([-lim, lim], [0, 0], color='r', lw=0.5)
#     ax.plot([0, 0], [-lim, lim], color='r', lw=0.5)
#     # ax.grid(True, which='major')

#     CS0 = ax.contour(X, Y, Z[:,:,0], clabels)
#     ax.clabel(CS0, inline=True, fontsize=6)
#     CS1 = ax.contour(X, Y, Z[:,:,1], clabels)
#     ax.clabel(CS1, inline=True, fontsize=6)

#     ax.plot([0, a00], [0, a01], color='r', lw=0.8)
#     ax.plot([0, a10], [0, a11], color='r', lw=0.8)

#     ax.set_title("z0 = {:s} ({:.2f} * x0 + {:.2f} * x1) , z1 = {:s} ({:.2f} * x0 + {:.2f} * x1)"
#     .format(nn.activation.name, a00, a01, nn.activation.name, a10, a11))

#     return

# ani = animation.FuncAnimation(fig, run, data_gen, interval=0)
# plt.show()