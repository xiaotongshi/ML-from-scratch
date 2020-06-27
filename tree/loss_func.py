import numpy as np
from sklearn.metrics import accuracy_score

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0
        
class LeastSquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, actual, predicted):
        return actual - predicted

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class ExponentialLoss(Loss):
    def gradient(self, actual, predicted):
        return actual * np.exp(-actual * predicted)

    def hess(self, actual, predicted):
        expits = np.exp(predicted)
        return np.exp * (1 - expits)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

# class LogisticLoss():
#     def __init__(self):
#         sigmoid = Sigmoid()
#         self.log_func = sigmoid
#         self.log_grad = sigmoid.gradient

#     def loss(self, y, y_pred):
#         y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
#         p = self.log_func(y_pred)
#         return y * np.log(p) + (1 - y) * np.log(1 - p)

#     # gradient w.r.t y_pred
#     def gradient(self, y, y_pred):
#         p = self.log_func(y_pred)
#         return -(y - p)

#     # w.r.t y_pred
#     def hess(self, y, y_pred):
#         p = self.log_func(y_pred)
#         return p * (1 - p)