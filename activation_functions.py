import numpy as np


class ActivationRelu:
    def __init__(self):
        # Initialize variables
        self.output = self.inputs = self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Make a copy of the values
        self.dinputs = dvalues.copy()
        # Zero gradient wher input values were negative
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def __init__(self):
        self.output = self.dinputs = self.inputs = None

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # get un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and graidents
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradients and add it to array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class ActivationSigmoid:
    def __init__(self):
        self.inputs = self.output = self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class ActivationLinear:
    def __init__(self):
        self.inputs = self.output = self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()
