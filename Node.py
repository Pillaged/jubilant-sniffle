import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import Backprop


X, y = spiral_data(100, 3)
t = zip(X, y)
print(t[0])

class Layer():
    def __init__(self, inputs_ct, neurons_ct) -> None:
        self.weights = 0.1 * np.random.randn(inputs_ct, neurons_ct)
        self.biases = np.zeros((1, neurons_ct))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

class Activ_ReLU():
    omega = "ReLU"
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) + self.biases


class Activ_SoftMax():
    omega = "SoftMax"
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) 
        norm_vals = exp_vals / np.sum(exp_vals, axis = 1, keepdims=True)
        self.output = norm_vals + self.biases

