
import numpy as np
import Activations
import Layers

class Loss:
    def __init__(self, name) -> None:
        pass

    def loss(self, pred, true):
        return None

    def gradient(self):
        return None

class MeanSquaredError(Loss):
    def __init__(self, name='MSE') -> None:
        self.name = name

    def loss(self, pred, true):
        self.pred = pred    # BS + (pred_dim,)
        self.true = true    # BS + (pred_dim,)
        loss = np.power(true - pred, 2) / 2  # BS + (pred_dim,)

        # Reduce batch axes.
        for _ in range(len(loss.shape)-1):
            loss = np.sum(loss, axis=0)
        return loss # (pred_dim)
        
    def gradient(self):
        return self.pred - self.true    # BS + (pred_dim,)

class BinaryCrossEntropy(Loss):
    # It would be Softmax if it were MultiCrossEntropy.
    # BinaryCrossEntropy has a single dimension pred.
    sigmoid = Activations.sigmoid

    def __init__(self, with_logits=True, name='BCE') -> None:
        self.with_logits = with_logits
        self.name = name

    def loss(self, pred, true):
        self.pred = pred    # BS + (pred_dim,)
        self.probability = BinaryCrossEntropy.sigmoid.eval(pred) if self.with_logits else pred
        self.true = true    # BS + (pred_dim,)
        loss = - true * np.log(self.probability) - (1.-true) * np.log(1.-self.probability)
        
        # Reduce batch axes.
        for _ in range(len(loss.shape)-1):
            loss = np.sum(loss, axis=0)
        return loss

    def gradient(self):
        # we believe sigmoid cannot evaluate to be 0 nor 1.
        gradPred = BinaryCrossEntropy.sigmoid.gradient(self.pred) if self.with_logits else np.ones_like(self.pred, dtype=self.pred.dtype) # BS + (out_dim = 1,)
        gradProbability = (1.-self.true)/(1.-self.probability) - self.true/(self.probability) # BS + (out_dim = 1,)
        return gradPred * gradProbability # BS + (pred_dim,)

class MultiCrossEntropy(Loss):
    softmax = Layers.Softmax()

    def __init__(self, with_logits=True,  name='MCE') -> None:
        self.with_logits = with_logits
        self.name = name

    def loss(self, pred, true):
        self.pred = pred    # BS + (out_dim,)
        self.probability = MultiCrossEntropy.softmax.Feedforwards(pred) if self.with_logits else pred    # BS + (out_dim,)
        self.true = true    # Bs + (out_dim,)
        loss = - true * np.log(self.probability) - (1.-true) * np.log(1.-self.probability)

        # Reduce batch axes.
        for _ in range(len(loss.shape)-1):
            loss = np.sum(loss, axis=0)
        return loss # (pred_dim,)

    def gradient(self):
        if self.with_logits:
            gradPred = MultiCrossEntropy.softmax.gradient() # BS + (out_dim, out_dim)
        else:
            # I want more efficient code.
            gradPred = np.zeros_like(self.pred.shape + (self.pred.shape[-1],), dtype=self.pred.dtype) # BS + (out_dim, out_dim)
            for i in range(self.pred.shape[-1]):
                gradPred[..., i, i] = 1.
        gradProbability = (1.-self.true)/(1.-self.probability) - self.true/(self.probability) # BS + (out_dim,)
        # print('pred', self.pred.shape); print(self.pred)
        # print('true', self.true.shape); print(self.true)
        # print('gradPred', gradPred.shape); print(gradPred)
        # print('gradPro', gradProbability.shape); print(gradProbability)
        grad = np.stack([gradPred[i, :, :] @ gradProbability[i] for i in range(self.pred.shape[0])], axis=0)   # assuming BS = (n,)
        return grad # BS + (pred_dim,)