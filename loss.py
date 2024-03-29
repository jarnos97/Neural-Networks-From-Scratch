import numpy as np
from activation_functions import ActivationSoftmax


class Loss:  # Common loss class
    def __init__(self):
        self.trainable_layers = self.accumulated_sum = self.accumulated_count = None

    def regularization_loss(self):
        # Default = 0
        regularization_loss = 0

        for layer in self.trainable_layers:
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

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        # calculate sample losses
        sample_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss, return it
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularzation=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularzation:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
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
        self.dinputs = None

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


class LossBinaryCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.dinputs = None

    @staticmethod
    def forward(y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)
        return sample_losses

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossMeanSquaredError(Loss):
    def __init__(self):
        super().__init__()
        self.dinputs = None

    @staticmethod
    def forward(y_pred, y_true):
        return np.mean((y_true - y_pred)**2, axis=-1)

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = -2 * (y_true - dvalues) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples


class LossMeanAbsoluteError(Loss):
    def __init__(self):
        super().__init__()
        self.dinputs = None

    @staticmethod
    def forward(y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples
