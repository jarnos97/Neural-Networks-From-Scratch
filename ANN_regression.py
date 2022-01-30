import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
from layers import LayerDense, LayerDropout
from activation_functions import ActivationRelu, ActivationSigmoid, ActivationSoftmax, ActivationLinear
from loss import LossMeanSquaredError
from optimizers import OptimizerAdam
import numpy as np

# Seed
nnfs.init()

X, y = sine_data()

# Dense layer with 1 input feature
dense1 = LayerDense(1, 64)

# Relu activation layer
activation1 = ActivationRelu()

# Second dense layer with 1 output value
dense2 = LayerDense(64, 64)

# Relu activation layer
activation2 = ActivationRelu()

# Second dense layer with 1 output value
dense3 = LayerDense(64, 1)

# Linear activation layer
activation3 = ActivationLinear()

# Loss function
loss_function = LossMeanSquaredError()

# Optimizer
optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)

# Define accuracy precision. This is the allowed deviance from the true y-value we will count as correctly classified
accuracy_precision = np.std(y) / 250

# train model
for epoch in range(10001):
    # Foward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    # Calculate data loss
    data_loss = loss_function.calculate(activation3.output, y)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + \
        loss_function.regularization_loss(dense3)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

    # Print results every 100 epochs
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, '+
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()



