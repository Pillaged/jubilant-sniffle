import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import Backprop


X, y = spiral_data(100, 3)
#print(X, y)

m, n = X.shape
data_dev = X[0:1000].T
Y_dev = y
X_dev = 1

#Shape (300,3)
def OneHot():
    num_answ = 3
    answers = np.array(y)
    one_hot = np.eye(num_answ)[answers]
    return one_hot

class Layer():
    def __init__(self, inputs_ct, neurons_ct) -> None:
        self.weights = 0.1 * np.random.randn(inputs_ct, neurons_ct)
        self.biases = np.zeros((1, neurons_ct))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

class Activ_ReLU():
    omega = "ReLU"
    def forward(self, inputs, biases):
        self.output = np.maximum(0, inputs) + biases


class Activ_SoftMax():
    omega = "SoftMax"
    def forward(self, inputs, biases):
        exp_vals = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) 
        norm_vals = exp_vals / np.sum(exp_vals, axis = 1, keepdims=True)
        self.output = norm_vals + biases

def Find_Omega_Derv(omega, weights, err):
    return Backprop.FindDeltaOmega(omega, weights, err)

