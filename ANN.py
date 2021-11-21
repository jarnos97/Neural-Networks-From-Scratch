#%% Imports
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layers import LayerDense
from activation_functions import ActivationRelu
from loss import ActivationSoftmaxLossCategoricalCrossentropy
from optimizers import OptimizerAdagrad, OptimizerRMSProp, OptimizerSGD, OptimizerAdam

# Seed
nnfs.init()

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
optimizer = OptimizerAdam(learning_rate=0.05, decay=5e-7)

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
              f"loss: {loss:.3f} " +
              f"lr: {optimizer.current_learning_rate}"
              )

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update layer parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# TODO: what happens when we use tensors instead of numpy? For speed
# TODO: Make a sequential-like function. To facilitate model creation.
