from turtle import shape
import Node
import numpy as np
import Backprop
import nnfs
from nnfs.datasets import spiral_data

#X, y = spiral_data(100, 3)
# np.random.shuffle(X)


# This block defines the layers and their shapes.
# dense1 is matrix of shape ([ [1], [2], [3] ], [ [1], [2], [3] ])
dense1 = Node.Layer(2, 3)
activation1 = Node.Activ_ReLU()

dense2 = Node.Layer(3, 4)
activation2 = Node.Activ_ReLU()

dense3 = Node.Layer(4, 3)
activation3 = Node.Activ_SoftMax()

# This block runs the layers and their activation functions.
# Also establishes       a relationship between the layers.
dense1.forward(Node.X)
activation1.forward(dense1.output, dense1.biases)

dense2.forward(activation1.output)
activation2.forward(dense2.output, dense2.biases)

dense3.forward(activation2.output)
activation3.forward(dense3.output, dense3.biases)

# Following block is for training the weights and biases.
erra = Backprop.linearErr(Node.OneHot(), activation3.output)
sum_error = np.sum(erra, axis=(1,0))


# General format: dSSR/dPredicted * dPredicted/dy * dy/dx * dx/dw = Sum(-2(Obsv - Pred)) * w * omega deriv * Input or 1

a = (Backprop.derivSSRdW(erra)) * dense3.weights
b = Backprop.derivSSRdW(erra)
c = Backprop.FindDeltaOmega("SoftMax", dense3.weights, sum_error)
d = activation3.output
for i in d:
    #print((i))
    print( b * c * a* i)
'''
dlt_wt1 = (Backprop.derivSSRdW(sum_error)) * dense3.weights * \
Backprop.FindDeltaOmega("SoftMax", dense3.weights, sum_error) \
* Node.X





dlt_b1 = Backprop.derivSSRdW(erra) * dense1.weights * Node.Find_Omega_Derv(erra) * 1

dlt_wt2 = Backprop.derivSSRdW(Backprop.linearErr()) * dense2.weights * Node.Find_Omega_Derv(erra) * Node.X
print(dlt_b1)'''