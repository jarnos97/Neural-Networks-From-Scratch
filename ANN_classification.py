# Imports
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from model import Model
from layers import LayerDense, LayerDropout
from activation_functions import ActivationRelu, ActivationSoftmax
from loss import LossCategoricalCrossEntropy
from optimizers import OptimizerAdam
from metrics import AccuracyCategorical

# Seed
nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Init model
model = Model()

# Add layers
model.add(LayerDense(2, 512, l2_weight=5e-4, l2_bias=5e-4))
model.add(ActivationRelu())
model.add(LayerDropout(0.1))
model.add(LayerDense(512, 3))
model.add(ActivationSoftmax())

# Set loss optimizer and accuracy
model.set(
    loss=LossCategoricalCrossEntropy(),
    optimizer=OptimizerAdam(learning_rate=0.05, decay=5e-5),
    accuracy=AccuracyCategorical()
)

# Finalize model
model.finalize()

# Train model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
# model.train(X, y, validation_data=(X_test, y_test), epochs=400, print_every=100)

# TODO: If we want to try to execute on GPU, need to convert arrays to tensors
# TODO: Make a sequential-like function!
# TODO: make bias optional
# TODO: add different ways to initialize the weights
# TODO: Make a base-class for our layers/activations/etc? Which require a forward and backward. Only if useful.
# TODO: make a wrapper/decorator or similar for logging. Instead of printing outputs.
# TODO: use TQDM package for progress bar
