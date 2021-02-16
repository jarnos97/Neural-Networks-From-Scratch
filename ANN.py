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


class Loss:  # Common loss class
    def calculate(self, output, y):
        # calculate sample losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):  # forward pass
        samples = len(y_pred)  # number of samples in a batch
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip data to prevent division by 0
        if len(y_true.shape) == 1:  # if labels are categorical (e.g. [1],[4])
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:  # if labels are one-hot encoded (e.g. [1, 0, 0])
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



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

# Create loss function
loss_function = LossCategoricalCrossEntropy()

# Apply forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# calulcate loss
loss = loss_function.calculate(activation2.output, y=y)
print('loss:', loss)

# Calculating accuracy
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:  # convert if one-hot encoded
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
