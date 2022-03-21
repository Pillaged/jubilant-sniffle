import numpy as np
import Node
import math


err = 0
alpha = 0.12
    
def FindDeltaOmega(function, weights_mtx, error_mtx):

    if function == "ReLU":
        return np.maximum(0, weights_mtx)
    elif function == "SoftMax":
        return np.exp(weights_mtx)/(1 + np.exp(weights_mtx))
    return "No type"

def linearErr(expected, tested):
    err = 0.5 * np.power((expected-tested), 2) #Error function essentially using r^2 values
    return err

def derivSSRdW(err):
    a = np.sqrt(err)
    err = -2*a
    return -2*a


