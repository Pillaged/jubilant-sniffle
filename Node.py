import numpy as np
import nnfs
from nnfs.datasets import vertical_data
import Backprop

np.random.seed(0)

# Base Layer Class
class Node:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = (0.01* np.random.randn(n_inputs, n_neurons))
        self.biases = np.zeros((1,n_neurons))

    def forward_pass(self,inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        #Gradient on parameters.
        self.dweights = np.dot(self.input, dvalues)
        self.dbiases = np.sum(self.biases, keepdims=True)

        #Gradient on values!!
        self.dinputs = np.dot(self.weights, dvalues)

class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
        self.inputs= inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0]=0

class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output= single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output)- np.dot(single_output,single_output.T)

# Overarching Loss class
class Loss:
    def calc(self, output, y):
        sample_losses = self.forward(output,y)
        loss_data = np.mean(sample_losses)
        return loss_data

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        batch_samples_ct = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)

        # Categorical Label makes array for a single value which declares True or False
        if len(y_true.shape)==1:
            correct_confidences = y_pred_clipped[range(batch_samples_ct), y_true]
        
        # Mask values so only one-hot positive probabilities shown.
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        # Losses
        negative_log_probability = -np.log(correct_confidences)
        return negative_log_probability

# Create data
X, y = vertical_data(samples=100, classes=3)

# Create all layers with weights, and activation functions to have data passed forward
dense1 = Node(2,3)
activation1 = Activation_ReLU()
dense2 = Node(3,3)
activation2 = Activation_Softmax()

# Create Loss function
loss_func = Loss_CategoricalCrossEntropy()

# Data passed forward
dense1.forward_pass(X)
activation1.forward(dense1.output)

dense2.forward_pass(activation1.output)
activation2.forward(dense2.output)

loss = loss_func.calc(activation2.output, y)
print('loss is: ', loss)

prediction_indices = np.argmax(activation2.output,axis=1)
accuracy = np.mean(prediction_indices == y)
print("acc: ", accuracy)

#Best Weights tracker
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for i in range(1000):
    dense1.weights = 0.05* np.random.randn(2,3)
    dense1.biases = 0.05* np.random.randn(1,3)
    dense2.weights = 0.05* np.random.randn(3,3)
    dense2.biases = 0.05* np.random.randn(1,3)

    dense1.forward_pass(X)
    activation1.forward(dense1.output)
    dense2.forward_pass(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_func.calc(activation2.output, y)
    
    prediction_indices = np.argmax(activation2.output,axis=1)
    accuracy = np.mean(prediction_indices == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', i,
        'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss


