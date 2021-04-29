#%% Imports
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Seed
nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # sample (n, n,) numbers from Gaussian distribution with center=0, variance=1, and multiply to make smaller
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Initialize variables
        self.output = self.inputs = self.dweights = self.dbiases = self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationRelu:
    def __init__(self):
        # Initialize variables
        self.output = self.inputs = self.dinputs  = None

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


class Loss:  # Common loss class
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        self.dinputs = None

    @staticmethod
    def forward(y_pred, y_true):  # forward pass
        samples = len(y_pred)  # number of samples in a batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip data to prevent division by 0
        if len(y_true.shape) == 1:  # if labels are categorical (e.g. [1],[4])
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # if labels are one-hot encoded (e.g. [1, 0, 0])
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of label in every sample
        labels = len(dvalues[0])
        # if labels are sparse, turn into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate the gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class ActivationSoftmaxLossCategoricalCrossentropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()
        self.output = self.dinputs = None

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        # calculate and return loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If lables are one-hot encoded convert to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy dvalues
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gadient
        self.dinputs = self.dinputs / samples


class OptimizerSGD():
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights  # dweights = gradients
        layer.biases += -self.learning_rate * layer.dbiases


# Setting values and initializing classes
X, y = spiral_data(samples=100, classes=3)  # 300 samples of 2-dimensional data

# Create dense layer with 2 input features and 3 output values
dense1 = LayerDense(n_inputs=2, n_neurons=64)

# Create RELU for dense layers
activation1 = ActivationRelu()

# Create second dense layer with 3 input features and 3 output values
dense2 = LayerDense(n_inputs=64, n_neurons=3)

# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer object
optimizer = OptimizerSGD()

# Execute
for epoch in range(10001):
    # Apply forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    # Perform forward pass through the activation/loss function (returns loss)
    loss = loss_activation.forward(dense2.output, y)

    # Calculating accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:  # convert if one-hot encoded
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print results every 100 epochs
    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}"
              )

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update layer parameters
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

