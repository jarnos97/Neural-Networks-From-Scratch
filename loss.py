import numpy as np
from activation_functions import ActivationSoftmax


class Loss:  # Common loss class
    @staticmethod
    def regularization_loss(layer):
        # Default = 0
        regularization_loss = 0

        # L1 weights regularization
        if layer.l1_weight > 0:
            regularization_loss += layer.l1_weight * np.sum(np.abs(layer.weights))

        # L1 bias regularization
        if layer.l1_bias > 0:
            regularization_loss += layer.l1_bias * np.sum(np.abs(layer.biases))

        # L2 weights regularization
        if layer.l2_weight > 0:
            regularization_loss += layer.l2_weight * np.sum(layer.weights * layer.weights)

        # L2 bias regularization
        if layer.l2_bias > 0:
            regularization_loss += layer.l2_bias * np.sum(layer.biases * layer.biases)

        return regularization_loss

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


class ActivationSoftmaxLossCategoricalCrossentropy:
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
