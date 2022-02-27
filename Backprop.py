from typing_extensions import runtime
import numpy as np
import Node

def Error(self, answers_mtx, runtime_mtx):
    return (0.5 * np.power((answers_mtx - runtime_mtx), 2))

def ReLU_deriv(Z):
    return Z > 0

class Backprop():

    def one_hot(expected_mtx):
        one_hot_Y = np.zeros(expected_mtx.size, expected_mtx.max + 1)
        one_hot_Y[np.arange(expected_mtx.size), expected_mtx] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
    
    def backward_prop(A1, A2, func1, func2, W1, W2, X, Y):
        one_hot_Y = Backprop.one_hot(Y)
        d_func2 = A2 - one_hot_Y
        d_W2 = 1 / Node.m * d_func2.dot(A1.T)
        d_b2 = 1 / Node.m * np.sum(d_func2)
        d_func1 = W2.T.dot(d_func2) * ReLU_deriv(func1)
        d_W1 = 1 / Node.m * d_func1.dot(X.T)
        d_b1 = 1 / Node.m * np.sum(d_func1)
        return d_W1, d_b1, d_W2, d_b2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1    
        W2 = W2 - alpha * dW2  
        b2 = b2 - alpha * db2    
        return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2