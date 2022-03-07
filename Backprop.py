import numpy as np
import Node

alpha = 0.12
     
class Backprop():
    def FindDeltaOmega(self, function, weights_mtx, error_mtx):

        if function == "ReLU":
            return np.maximum(0, weights_mtx)
        elif function == "SoftMax":
            pass
        pass
    def linearErr(self, expected, input):
        self.err = 0.5 * np.power((expected-input), 2) #Error function essentially using r^2 values
    def derivSSRdW(self, err):
        a = np.sqrt(err)
        self.err = a
        return -2*a
