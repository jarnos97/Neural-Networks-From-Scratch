#%% Imports
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Seed
nnfs.init()


#%% Dense layer
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # sample (n, n,) numbers from Gaussian distribution with center=0, variance=1, and multiply to make smaller
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = 0  # Initialize variable

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationRelu:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        # get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


 #%% Execution
X, y = spiral_data(samples=100, classes=3)  # 300 samples of 2-dimensional data

# Create dense layer with 2 input features and 3 output values
dense1 = LayerDense(n_inputs=2, n_neurons=3)

# Create RELU for dense layers
activation1 = ActivationRelu()

# Create second dense layer with 3 input features and 3 output values
dense2 = LayerDense(n_inputs=3, n_neurons=3)

# Create Softmax for output
activation2 = ActivationSoftmax()

# Apply forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])


