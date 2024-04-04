import numpy as np
import Layers
from Optimizer import Optimizer

class Sequential:

    def __init__(self) -> None:
        self.layers = []
        self.wHistory = []
    
    def __init__(self, layers) -> None:
        self.wHistory = []
        self.layers = layers
        if (len(layers) > 0):
            assert self.layers[0].neurons is not None
            for i in range(1, len(layers)):
                if self.layers[i].neurons is not None:
                    self.layers[i].setNeurons(self.layers[i].dim, self.layers[i-1].dim)

    def GetWeights(self):
        # must be a list, and not a tuple generator
        totalWeights = [layer.GetWeights() for layer in self.layers if type(layer).hasWeights]
        return totalWeights

    def SetWeights(self, totalWeights) -> None:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if type(layer).hasWeights:
                layer.SetWeights(totalWeights[i])

    def SetWeightsFlatten(self, totalWeightsFlatten) -> None:
         for i in range(len(self.layers)):
            layer = self.layers[i]
            if type(layer).hasWeights:
                nWeights = layer.dim * (layer.input_dim+1)
                totalWeights = totalWeightsFlatten[: nWeights]
                totalWeightsFlatten = totalWeightsFlatten[nWeights:]
                layer.SetWeights(totalWeights.reshape(layer.dim, layer.input_dim+1))

    def GetGradients(self):
        # must be a list, and not a tuple generator
        wFullGradients = [layer.GetGradients() for layer in self.layers if type(layer).hasWeights]
        return wFullGradients

    def GetOutputs(self):
        return [layer.GetOutputs() for layer in self.layers]

    def Feedforwards(self, x):
        """
        Feeds inputs forwards through the layers.
        input.shape: BS + (first layer.input_dim,)
        output.shape: BS + (last layer.dim,)
        """
        for layer in self.layers:
            x = layer.Feedforwards(x)
        return x

    def Backpropagate(self, gradient):
        """
        Propagates inputs backwards through the layers.
        input.shape: BS + (last layers.dim,)
        output.shape: BS + (first layers.input_dim,)
        """
        for i in range(len(self.layers)-1, -1, -1):
            gradient = self.layers[i].Backpropagate(gradient)
        return gradient

    def Compile(self, loss, optimizer): # Add metrics later
        self.loss = loss
        self.optimizer = optimizer
        # self.metrics = metrics
        return

    def fit(self, X, Y, batchsize=64, epochs=10, wHistory=False):
        for epoch in range(epochs):
            X, Y = permute(X, Y)
            for bStart in range(0, Y.shape[0], batchsize):
                minibatch_x, minibatch_y = \
                    X[bStart : bStart + batchsize, :], Y[bStart : bStart + batchsize, :]
                self.step_with_minibatch(minibatch_x, minibatch_y, epoch)
            if wHistory:
                self.wHistory.append(self.GetWeights())
        return

    def step_with_minibatch(self, minibatch_x, minibatch_y, epoch) -> None:
        W = self.GetWeights()
        pred = self.Feedforwards(minibatch_x)
        loss = self.loss.loss(pred, minibatch_y)
        delta = self.loss.gradient() # BS + (out_dim,)
        self.Backpropagate(delta)
        G = self.GetGradients()
        W_flatten, G_flatten = np.array(()), np.array(())
        for (w, g) in zip(W,G):
            # Sum up the gradients throughout the batch
            for i in range(len(g.shape)-len(w.shape)):
                g = np.sum(g, axis=0)   # reduce_sum on Batch dimentions
            # Flatten and concatenate
            W_flatten = np.concatenate( (W_flatten, w.flatten()), axis=-1)
            G_flatten = np.concatenate( (G_flatten, g.flatten()), axis=-1)
        # Apply optimizer
        W_new = self.optimizer.step(W_flatten, G_flatten, epoch)
        # Set W_new back to Model.
        self.SetWeightsFlatten(W_new)
        return
         
def reshapeForMCE(Y, pred_dim):
    onehotY = np.array([], dtype=float)
    if len(Y.shape) > 0 and Y.shape[0] > 0:
        assert 0 <= np.min(Y)
        assert np.max(Y) < pred_dim
        Z = np.zeros((Y.shape[0], pred_dim), dtype=float)
        for i in range(Z.shape[0]):
            Z[i][Y[i]] = 1.
        onehotY = np.array(Z)
    return onehotY

def permute(X, Y):
    rng = np.random.default_rng()
    permutation = rng.permutation(np.arange(Y.shape[0]))
    X = X[permutation, :]
    Y = Y[permutation, :]
    return X, Y

# tests --------------------------------------------
# from Activations import *

# print("OOOOOOOO     model = Sequential([...............")
# model = Sequential([
#     Layers.Dense(1, input_dim=1, activation=relu), 
#     Layers.Dense(2, activation=sigmoid), 
#     Layers.Dense(9, activation=tanh),
#     Layers.Dense(1, activation=sigmoid)
#     ])

# # dense1 = Dense(1, input_dim=1, activation=identity)
# # model = Sequential([dense1])

# totalWeights = [
#     np.array(((1., -1.),) * 1, dtype=float),
#     np.array(((1, 2),) * 2, dtype=float),
#     np.array(((1, -2, 1),) * 9, dtype=float),
#     np.array(((1,2,3,4,5,6,7,8,9, 0),) * 1, dtype=float),
#     ]
# model.SetWeights(totalWeights)

# print('0', model.GetWeights())

 
# print('1: ', model.Feedforwards(np.array((2.,))))    # BS == (), shape == BS + (1,)
# print('2: ', model.Backpropagate(np.array((-2,))))  # BS == (), shape == BS + (1,)

# print('3: ', model.Feedforwards(np.array((3.,))))    # BS == (), shape == BS + (1,)
# print('4: ', model.Backpropagate(np.array((1,))))   # BS == (), shape == BS + (1,)

# print('5(1,3,1): ', model.Feedforwards(np.array(((2.,),(3,),(2,))))) # BS == (3,), shape == BS + (1,)
# print('6(2,4,2): ', model.Backpropagate(np.array(((-2,), (1,), (-2,))))) # BS == (3,), shape == BS + (1,)

# print('7: ', model.Feedforwards(np.array((((2.,),(3,)),((2.,),(3,))))))  # BS == (2,2), shape == BS + (1)
# print('8: ', model.Backpropagate(np.array((((-2,), (1,)),((-2,), (1,))))))  # BS == (2,2), shape == (2,2,1)


# dense1 = Layers.Dense(1, input_dim=2, activation=identity)
# model = Sequential([dense1])

# totalWeights = np.stack( (np.array((1., -1., 2.)),) * 1, axis=0)
# dense1.SetWeights(totalWeights)

# model = Sequential([dense1])
# x = np.array((1,2))
# print('1: ', model.Feedforwards(x))  # BS == (), shape == BS + (1,)
# print('2: ', model.Backpropagate( np.array((1,))))  # BS == (), shape == () + (2,)

# x = np.array((2,3))
# print('3:', model.Feedforwards(x))   # BS == (), shape == BS + (1,)
# print('4: ', model.Backpropagate( np.array((-2,))))

# x = np.array(((1,2),(2,3),(1,2)))
# print('5(1,3,1): ', model.Feedforwards(x))   # BS == (3,), shape == BS + (1,) 
# print('6(2,3,2): ', model.Backpropagate( np.array(((1,), (-2,), (1,))) ))  # BS == (3,), shape == BS + (2,)


# x = np.array((((1,2),(2,3)),((2,3),(1,2))))
# print('7: ', model.Feedforwards(x))  # BS == (2,2), shape == BS + (1,)
# print('8: ', model.Backpropagate( np.array((((1,), (-2,)), ((1,), (-2,))))))  # BS == (2,2), shape == BS + (2,)

# x = np.array(((((1,2),(2,3)),((1,2),(2,3))), (((1,2),(2,3)),((1,2),(2,3)))))
# print('9: ', model.Feedforwards(x))  # BS == (2,2,2), shape == (2,2,2,1)
# print('10: ', model.Backpropagate( np.array(((((1,),(-2,)), ((1,), (-2,))),(((1,),(-2,)), ((1,),(-2,)))))))  # BS == (2,2,2), shape == BS + (2,)


# import Activations
# model = Sequential([
#     Layers.Dense(1, input_dim=1, activation=Activations.relu), 
#     Layers.Dense(2, activation=Activations.sigmoid), 
#     Layers.Dense(9, activation=Activations.tanh),
#     Layers.Dense(1, activation=Activations.sigmoid),
#     Layers.Softmax()
#     ])

# print(model.layers)