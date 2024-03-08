import numpy as np

class neuron:
    def __init__(self, input_dim = 0, weights = 0, bias = 0) -> None:
        self.Initialize(input_dim, weights, bias)

    def Initialize(self, input_dim, weights, bias) -> None:
        assert type(int(bias)) == type(3)
        self.bias = np.array((bias,), dtype=float)
        # assert np.array(bias).shape == () and np.array((bias,)).shape == (1,)
        self.input_dim, self.weights =  neuron.create1DWeights(input_dim, weights)

    def print(self) -> None:
        print(self.dim, self.weights, self.bias)

    def Feedforwards(self, x, activation):
        """
        Feeds x forwards through the neuron.
        x.shpae: BS + (self.input_dim,)
        eg: if BS == () then x.shape == BS + (self.input_dim,) == (self.input_dim,)
        return.shape: BS + (1,)
        """
        self.x = x    # shape: BS + (dim,)
        self.sum = x @ self.weights + self.bias.squeeze() # super(BS, ()) = BS
        self.output = activation.eval(self.sum) # BS
        return self.output  # shape: BS
    
    def Backpropagate(self, nrOutGrad, activation) -> None:
        """
        Propagate gradients backwards through the neuron.
        nrOutGrad: d(Loss)/d(output), shape: BS
        eg: if BS == () then nrOutGrad.shape == ()
        return.shape: BS + (self.input_dim,)
        """
        sumFullGrad = activation.gradient(self.sum) * nrOutGrad    # d(Loss)/d(self.sum). BS
        self.bFullGrad = sumFullGrad[..., np.newaxis]              # d(Loss)/d(self.bias). BS
        repeated = np.stack((sumFullGrad,) * self.input_dim, axis=-1) # BS + (input_dim,)
        self.wFullGrad = repeated * self.x       # d(Loss)/d(self.weights).  BS + (input_dim,)
        self.fullGrad = repeated * self.weights  # d(Loss)/d(self.x).  BS + (dim,)
        return self.fullGrad   # shape: BS + (input_dim,)
    
    def GetWeights(self):
        return np.concatenate((self.weights, self.bias), axis=-1)   # Find for a more efficiend code

    def SetWeights(self, totalWeights) -> None:
        assert totalWeights.shape == (self.input_dim + 1,)
        self.weights = totalWeights[:-1]
        self.bias = np.array((totalWeights[-1],))

    def GetGradients(self):
        return np.concatenate((self.wFullGrad, self.bFullGrad), axis=-1) # BS + (input_shape+1,)

    # def setWFullGradients(self, wFullGradients) -> None:
    #     self.wFullGrad = wFullGradients[:-1]
    #     self.bFullGrad = np.array((wFullGradients[-1],))

    def regularize(irregular):
        if np.array(irregular).shape == () and np.array((irregular,)).shape == (1,): # if irregular is a scalar.
            irregular = np.array((irregular,))
        return irregular

    def create1DWeights(input_dim, weights):
        assert type(int(input_dim)) == type(3)
        input_dim = int(input_dim)
        test = np.array(weights, dtype=float)
        dimensions = len(test.shape)
        if dimensions == 0: # weights is like 3 or np.array(1), of shape ()
            weights = np.array((weights,), dtype=float)
        elif dimensions == 1:   # weights is like array([1,2,3]), of shape (3)
            weights = np.array(weights, dtype=float)
        elif dimensions >= 2:   # weights is like np.array([[1,2],[3,4]]), of shape (2,2).
            raise Exception("Invalid initialization of neuron")
        assert len(weights.shape) == 1 and weights.shape[0] > 0

        if input_dim > 0:
            if weights.shape[0] > input_dim:
                weights = weights[: input_dim]
            elif weights.shape[0] < input_dim:
                if weights.shape[0] > 1:
                    raise Exception("Fewer weights than input_dim")
                else:
                    weights = np.array((weights[0],) * input_dim)
        else:
            input_dim = weights.shape[0]
            weights = np.array(weights, dtype=float)

        return input_dim, weights

# tests --------------------------------------------

# import Activations

# nr = neuron(weights = 1, bias = -1) # imput_dim =+ 1
# print("nr.input_dim: ", nr.input_dim, ", nr.weights: ", nr.weights, ", nr.bias:", nr.bias)

# x = np.array((1.,)) # BS == ()
# print('1: ', nr.Feedforwards(x, Activations.identity))   # shape == BS
# print('2: ', nr.Backpropagate( np.array((-1)), Activations.identity)) # shape == (1,)

# print('grad', nr.GetGradients())

# x = np.array((2.,)) # BS == ()
# print('3:', nr.Feedforwards(x, Activations.identity))    # shape == BS
# print('4: ', nr.Backpropagate( np.array(-2), Activations.identity)) # shape == (1,)

# print('grad', nr.GetGradients())

# x = np.array(((1,),(2,),(1,)))  # (3,1) = (3,) + (1,), so BS == (3,)
# print('5(1,3,1): ', nr.Feedforwards(x, Activations.identity))    # shape = BS
# print('6(2,3,2): ', nr.Backpropagate( np.array((-1,-2,-1)), Activations.identity))  # shape == (3,1)

# print('grad', nr.GetGradients())

# x = np.array((((1,),(2,)),((2,),(1,)))) # (2,2,1) = (2,2) + (1,), so BS == (2,2)
# print('7: ', nr.Feedforwards(x, Activations.identity))   # shape == BS == (2,2)
# print('8: ', nr.Backpropagate( np.array(((1,-2), (1,-2))), Activations.identity))   # shape == (2,2,1)

# print('grad', nr.GetGradients())

# print('11', nr.GetWeights())
# nr.SetWeights(np.array((1.,2.)))
# print('12', nr.GetWeights())


# #--------------------------------------------------------------------------------------

# print("OOOOOO neuron(dim=2, weights = (1., -2.), bias = np.array((-1.,)))")
# nr = neuron(input_dim=2, weights = (1., 0.), bias = np.array((-1.)))
# print("nr.weights: ", nr.weights, "nr.bias:", nr.bias)

# x = np.array((1,2)) # (2,) = BS + (2,), so BS == ()
# print('1: ', nr.Feedforwards(x, Activations.identity))   # shape == BS ==()
# print('2: ', nr.Backpropagate( np.array(-1), Activations.identity))  # shape == (2,)

# print('grad', nr.GetGradients())

# x = np.array((2,3)) # (2,) = BS + (2,), so BS == ()
# print('3:', nr.Feedforwards(x, Activations.identity))    # shape == BS == ()
# print('4: ', nr.Backpropagate( np.array(-2), Activations.identity))  # shape == (2,)

# print('grad', nr.GetGradients())

# x = np.array(((1,2),(2,3),(1,2)))   # (3,2) = BS + (2,), so BS == (3,)
# print('5(1,3,1): ', nr.Feedforwards(x, Activations.identity))    # shape == BS == (3,)
# print('6(2,3,2): ', nr.Backpropagate( np.array((-1,-2,-1)), Activations.identity))  # shape == (3,2) 

# print('grad', nr.GetGradients())

# x = np.array((((1,2),(2,3)),((2,3),(1,2)))) # (2,2,2) = BS + (2,), so BS == (2,2)
# print('7: ', nr.Feedforwards(x, Activations.identity))   # shape == (2,2)
# print('8: ', nr.Backpropagate( np.array(((1,-2), (1,-2))), Activations.identity))   # shape == (2,2,2)

# print('grad', nr.GetGradients())

# x = np.array(((((1,2),(2,3)),((1,2),(2,3))), (((1,2),(2,3)),((1,2),(2,3))))) # BS == (2,2,2)
# print('9: ', nr.Feedforwards(x, Activations.identity))   # shape == (2,2,2)
# print('10: ', nr.Backpropagate( np.array((((1,-2), (1,-2)),((1,-2), (1,-2)))), Activations.identity))   # shape == (2,2,2,2)

# print('11', nr.GetWeights())
# nr.SetWeights(np.array((1.,2.,3.)))
# print('12', nr.GetWeights())

#---------------------------------------------------------------

# import matplotlib.pyplot as plt
# import itertools
# import matplotlib.animation as animation

# import Activations
# nr = neuron(2)

# lim = 5.
# x = np.arange(-lim, lim, 0.2)#.reshape(-1, 1)
# y = np.arange(-lim, lim, 0.2)#.reshape(-1, 1)
# X, Y = np.meshgrid(x, y)
# XY = np.stack((X, Y), axis=-1)

# fig, ax = plt.subplots()

# def data_gen():
#     for cnt in itertools.count():
#         t = 2 * np.pi * cnt / 360
#         yield np.cos(t), np.sin(t)

# activation = Activations.tanh
# Z = nr.Feedforwards(XY, activation=activation)#.squeeze()

# def run(data):
#     # update data
#     a1, a2 = data
#     nr.SetWeights(np.array((a1, a2, 0.)))
#     Z = nr.Feedforwards(XY, activation=activation)#.squeeze()
#     ax.cla()
#     ax.plot([-lim, lim], [0, 0], color='r', lw=0.5)
#     ax.plot([0, 0], [-lim, lim], color='r', lw=0.5)

#     ax.grid(True, which='major')
#     CS = ax.contour(X, Y, Z, 16)
#     ax.clabel(CS, inline=True, fontsize=6)

#     ax.plot([0, a1], [0, a2], color='b', lw=0.8)

#     ax.set_title("z = {:s} ( {:.2f} * x1 + {:.2f} * x2 )"
#     .format(activation.name, a1, a2))
#     return CS

# ani = animation.FuncAnimation(fig, run, data_gen, interval=0)
# plt.show()