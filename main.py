import Node
import numpy as np
import Backprop
import nnfs
from nnfs.datasets import spiral_data

#X, y = spiral_data(100, 3)
#np.random.shuffle(X)




class main():
    def __init__(self) -> None:
        #This block defines the layers and their shapes.
        #dense1 is matrix of shape ([ [1], [2], [3] ], [ [1], [2], [3] ])
        dense1 = Node.Layer(2,3)
        activation1 = Node.Activ_ReLU()

        dense2 = Node.Layer(3,4)
        activation2 = Node.Activ_ReLU()

        dense3 = Node.Layer(4,3)
        activation3 = Node.Activ_SoftMax

        #This block runs the layers and their activation functions.
        #Also establishes the relationship between the layers.
        dense1.forward(Node.X)
        activation1.forward(dense1.output)

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        dense3.forward(activation2.output)
        activation3.forward(dense3.output)



        print(activation3.output)