import numpy as np

import Activations
import Layers
import Model
import Losses
import Optimizer

from GUI import GUI

#----------------------- define task --------------------------------------
# loss, dim, pred_dim = Losses.MeanSquaredError(), 6, 1
# loss, dim, pred_dim = Losses.BinaryCrossentropy(), 6, 1
loss, dim, pred_dim = Losses.CategoricalCrossentropy(), 15, 3 #30, 15 # 6, 5# 30, 15


assert loss.__name__ != 'mean_squared_error' or pred_dim == 1
assert loss.__name__ != 'binary_crossentropy' or pred_dim == 1
assert loss.__name__ != 'categorical_crossentropy' or pred_dim > 1


dense0 = Layers.Dense(dim, input_dim=2, activation=Activations.sigmoid)
# dense1 = Layers.Dense(dim, activation=Activations.tanh)
dense2 = Layers.Dense(pred_dim, activation=Activations.identity)
model = Model.Sequential([ dense0, dense2 ])
model.SetWeights([
    np.array([(np.cos(2*k*np.pi/dim), np.sin(2*k*np.pi/dim), 0.) for k in range(dim)]),
    # np.array([((3.,) * dim + (0.,)) for i in range(dim)]),
    np.array([((3.,) * dim + (0.,)) for i in range(pred_dim)])
])
weightsShapes = [weights.shape for weights in model.GetWeights()]
learningRate = 5e-5
optimizer = Optimizer.DummyOptimizer(learningRate=learningRate)
model.Compile(loss, optimizer)

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import activations
# dense0 = layers.Dense(dim, input_shape=(2,), activation='sigmoid')
# dense1 = layers.Dense(dim, activation='tanh')
# dense2 = layers.Dense(pred_dim, activation='linear')
# model = tf.keras.Sequential([dense0, dense2])
# optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
# loss = tf.keras.losses.get(loss.__name__)
# print(loss.__name__)
# model.compile(optimizer=optimizer, loss=loss)

#========= Train
def permute(X, Y):
    rng = np.random.default_rng()
    permutation = rng.permutation(np.arange(Y.shape[0]))
    X = X[permutation, :]
    Y = Y[permutation, :]
    return X, Y

def collectXY():
    X, Y = GUI.collectData()
    if model.loss.__name__ == 'categorical_crossentropy':
        Y = Model.reshapeForMCE(Y, pred_dim)
    assert X.shape[0] == Y.shape[0]
    return X, Y

def CompileTitle():
    activation = model.layers[0].activation
    X, Y = collectXY()
    XYShape = [X.shape, Y.shape]
    return (weightsShapes, XYShape, activation.__name__, loss.__name__, optimizer.lr.numpy() if not np.isscalar(optimizer.lr) else optimizer.lr)

run = 0
def Train(epochs, batchsize):
    X, Y = collectXY()
    global run
    print("{:d}. ".format(run), end='')
    run += 1
    if X.shape[0] > 0:
        model.fit(X, Y, batchsize, epochs)
        GUI.refreshMainAxes(X, nContours=15)
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
    if not np.isscalar(optimizer.lr):
        optimizer.lr.assign(optimizer.lr.numpy() * 2)
    else:
        optimizer.lr *= 2
    return

def LrDn():
    if not np.isscalar(optimizer.lr):
        optimizer.lr.assign(optimizer.lr.numpy() / 2)
    else:
        optimizer.lr /= 2
    return

#----------------------- launch GUI ---------------------------------

GUI.init(model, loss.__name__, pred_dim)
# GUI.refreshMainAxes(None, nContours=15, weights=model.GetWeights())
GUI.activateDataCreator()
GUI.yieldControlToUI(Train, Remove, LrUp, LrDn, CompileTitle)
title = CompileTitle()
GUI.showTitle(title, "")
GUI.showPlot()
