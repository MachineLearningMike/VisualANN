import numpy as np


import Activations
import Layers
import Model
import Losses
import Optimizer

from GUI import GUI

#----------------------- define task --------------------------------------
# loss, dim, pred_dim = Losses.MeanSquaredError(), 6, 1
# loss, dim, pred_dim = Losses.BinaryCrossEntropy(), 6, 1
loss, dim, pred_dim = Losses.MultiCrossEntropy(), 6, 5# 30, 15

assert loss.name != 'MSE' or pred_dim == 1
assert loss.name != 'BCE' or pred_dim == 1
assert loss.name != 'MCE' or pred_dim > 1

#----------------------- create model -------------------------------------
dense0 = Layers.Dense(dim, input_dim=2, activation=Activations.sigmoid)
dense1 = Layers.Dense(dim, activation=Activations.tanh)
dense2 = Layers.Dense(pred_dim, activation=Activations.identity)
model = Model.Sequential([
    dense0, 
    # dense1, 
    dense2
    ])
model.SetWeights([
    np.array([(np.cos(2*k*np.pi/dim), np.sin(2*k*np.pi/dim), 0.) for k in range(dim)]),
    # np.array([((3.,) * dim + (0.,)) for i in range(dim)]),
    np.array([((3.,) * dim + (0.,)) for i in range(pred_dim)])
])
weightsShapes = [weights.shape for weights in model.GetWeights()]

#----------------------- define optimizer ----------------------------------
learningRate = 5e-5
optimizer = Optimizer.DummyOptimizer(learningRate=learningRate)

#----------------------- compile model -------------------------------------
model.Compile(loss, optimizer)

#----------------------- define train data ---------------------------------

#========= Train
def permute(X, Y):
    rng = np.random.default_rng()
    permutation = rng.permutation(np.arange(Y.shape[0]))
    X = X[permutation, :]
    Y = Y[permutation, :]
    return X, Y

def collectXY():
    X, Y = GUI.collectData()
    if model.loss.name == 'MCE':
        Y = Model.reshapeForMCE(Y, pred_dim)
    assert X.shape[0] == Y.shape[0]
    return X, Y

def CompileTitle():
    activation = model.layers[0].activation
    X, Y = collectXY()
    XYShape = [X.shape, Y.shape]
    return (weightsShapes, XYShape, activation.name, loss.name, optimizer.learningRate)

run = 0
def Train(epochs, batchsize):
    X, Y = collectXY()
    global run
    print("{:d}. ".format(run), end='')
    run += 1
    if X.shape[0] > 0:
        model.Fit(X, Y, batchsize, epochs)
        GUI.refreshMainAxes(X, nContours=15, weights=model.GetWeights())
        print("{:d} epochs".format(epochs))
        return True
    else:
        return False

#----------------------- inferface to delegate to GUI ---------------------------------

def Remove():
    global run
    run = 0
    return

def LrUp():
    optimizer.learningRate *= 2
    return

def LrDn():
    optimizer.learningRate /= 2
    return

#----------------------- launch GUI ---------------------------------

GUI.init(model, loss.name, pred_dim)
GUI.refreshMainAxes(None, nContours=15, weights=model.GetWeights())
GUI.activateDataCreator()
GUI.yieldControlToUI(Train, Remove, LrUp, LrDn, CompileTitle)
title = CompileTitle()
GUI.showTitle(title, "")
GUI.showPlot()
