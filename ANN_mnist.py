# Imports
import os

import numpy as np
import nnfs
from model import Model
from layers import LayerDense, LayerDropout
from activation_functions import ActivationRelu, ActivationSoftmax
from loss import LossCategoricalCrossEntropy
from optimizers import OptimizerAdam
from metrics import AccuracyCategorical
from helperfunctions.mnist_dataset import create_data_mnist


# Seed
nnfs.init()

# Create dataset
X, y, X_test, y_test = create_data_mnist(path='data/mnist')
print(f"Finished loading data")

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Init model
model = Model()

# Add layers
model.add(LayerDense(X.shape[1], 128))
model.add(ActivationRelu())
# model.add(LayerDropout(0.1))
model.add(LayerDense(128, 128))
model.add(ActivationRelu())
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())

# Set loss optimizer and accuracy
model.set(
    loss=LossCategoricalCrossEntropy(),
    optimizer=OptimizerAdam(decay=1e-3),
    accuracy=AccuracyCategorical()
)

# Finalize model
model.finalize()

# Train model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

# Retrieve and print parameters
parameters = model.get_parameters()
print(parameters)

# Evaluate
model.evaluate(X, y)
model.evaluate(X_test, y_test)


# TODO: If we want to try to execute on GPU, need to convert arrays to tensors
# TODO: Make a sequential-like function!
# TODO: make bias optional
# TODO: add different ways to initialize the weights
# TODO: Make a base-class for our layers/activations/etc? Which require a forward and backward. Only if useful.
# TODO: make a wrapper/decorator or similar for logging. Instead of printing outputs.
# TODO: use TQDM package for progress bar
