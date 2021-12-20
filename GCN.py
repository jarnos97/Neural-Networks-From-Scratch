from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from layers import LayerDense, LayerDropout, LayerGCNConv
from activation_functions import ActivationRelu, ActivationSigmoid
from loss import ActivationSoftmaxLossCategoricalCrossentropy, LossBinaryCrossEntropy, LossCategoricalCrossEntropy
from optimizers import OptimizerAdagrad, OptimizerRMSProp, OptimizerSGD, OptimizerAdam
import numpy as np
from torch_geometric.utils import add_remaining_self_loops

dataset = Planetoid(root='data', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

# Create adjacency matrix
adjacency = np.zeros((2708, 2708))
for index, row_index in enumerate(data.edge_index[0]):
    col_index = data.edge_index[1][index]
    adjacency[row_index][col_index] = 1


# Add self-loops
np.fill_diagonal(adjacency, 1)


def normalizeAdjacency(m):
    # Check that the matrix is square
    assert m.shape[0] == m.shape[1]
    # Compute the degree vector
    d = np.sum(m, axis=1)
    # Invert the square root of the degree
    d = 1 / np.sqrt(d)
    # And build the square root inverse degree matrix
    d_diag = np.diag(d)
    # Return the Normalized Adjacency
    return d_diag @ m @ d_diag


# Degree-normalize the adjacency
adjacency = normalizeAdjacency(adjacency)
adjacency = adjacency.round(2)

# Convert tensors to numpy arrays
features = np.array(data.x)
y_true = np.array(data.y)

# Create layers
conv1 = LayerGCNConv(1433, 16)
activation1 = ActivationRelu()
dropout = LayerDropout(0.5)
conv2 = LayerGCNConv(16, 7)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Create optimizer
optimizer = OptimizerAdam(learning_rate=0.001, decay=5e-4)

# Train the model
for epoch in range(2000):
    # Forward pass
    conv1.forward(a=adjacency, h=features)
    activation1.forward(conv1.output)
    dropout.forward(activation1.output)
    conv2.forward(a=adjacency, h=dropout.output)

    # Perform forward pass through the activation/loss function (returns loss)
    loss = loss_activation.forward(conv2.output, y_true)

    # Calculating accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_true.shape) == 2:  # convert if one-hot encoded
        y = np.argmax(y_true, axis=1)
    accuracy = np.mean(predictions == y_true)

    # Print results every 100 epochs
    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"lr: {optimizer.current_learning_rate}"
              )

    # Backward pass
    loss_activation.backward(loss_activation.output, y_true)
    conv2.backward(loss_activation.dinputs)
    dropout.backward(conv2.dinputs)
    activation1.backward(dropout.dinputs)
    conv1.backward(activation1.dinputs)

    # Update layer parameters
    optimizer.pre_update_params()
    optimizer.update_params(conv1)
    optimizer.update_params(conv2)
    optimizer.post_update_params()

# NOTE: It works!
