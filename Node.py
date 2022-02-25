import numpy as np
import nnfs
from nnfs.datasets import spiral_data


X, y = spiral_data(100, 3)
print((X), (y))


class Layer():
    def __init__(self, inputs_ct, neurons_ct) -> None:
        self.weights = 0.11 * np.random.randn(inputs_ct, neurons_ct)
        self.biases = np.zeros((1, neurons_ct))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)

class Activ_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activ_SoftMax():
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True)) 
        norm_vals = exp_vals / np.sum(exp_vals, axis = 1, keepdims=True)
        self.output = norm_vals

#This block defines the layers
dense1 = Layer(2,3)
activation1 = Activ_ReLU()

dense2 = Layer(3,4)
activation2 = Activ_ReLU()

dense3 = Layer(4,3)
activation3 = Activ_SoftMax

#This block runs the layers and their activation functions.
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)



print(len(activation2.output))