import numpy as np
from numpy import ndarray


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int, l1_weight: float = 0, l1_bias: float = 0,
                 l2_weight: float = 0, l2_bias: float = 0) -> None:
        # sample (n, n,) numbers from Gaussian distribution with center=0, variance=1, and multiply to make smaller
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Initialize variables
        self.output = self.inputs = self.dweights = self.dbiases = self.dinputs = None
        # Set regularization strength
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 weight
        if self.l1_weight > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.dweights += self.l1_weight * d_l1

        # L1 bias
        if self.l1_bias > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.dbiases += self.l1_bias * d_l1

        # L2 weights
        if self.l2_weight > 0:
            self.dweights += 2 * self.l2_weight * self.weights

        # L2 biases
        if self.l2_bias > 0:
            self.dbiases += 2 * self.l2_bias * self.biases

        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class LayerDropout:
    def __init__(self, rate):
        # Store rate --> inverted as we need as success rate
        self.rate = 1 - rate
        self.inputs = self.output = self.binary_mask = self.dinputs = None

    def forward(self, inputs, training):
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class LayerInput:
    def __init__(self):
        self.output = None

    def forward(self, inputs, training):
        self.output = inputs


class LayerGCNConv:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Initialize variables
        self.output = self.a = self.h = self.dweights = self.dbiases = self.dinputs = None

    def forward(self, a: ndarray, h: ndarray) -> ndarray:
        # Need to compute AHW + B
        self.a = a
        self.h = h
        self.output = np.linalg.multi_dot([a, h, self.weights]) + self.biases

    def backward(self, dvalues) -> ndarray:
        # Need to compute three gradients: input, weights, and biases
        self.dweights = np.dot(self.a, self.h)
        self.dweights = np.dot(self.dweights.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(self.a, dvalues)
        self.dinputs = np.dot(self.dinputs, self.weights.T)
    # TODO: adapt to work with new model
