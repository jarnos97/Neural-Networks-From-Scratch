from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from layers import LayerDense, LayerDropout, LayerGCNConv
from activation_functions import ActivationRelu, ActivationSigmoid, ActivationSoftmax
from loss import ActivationSoftmaxLossCategoricalCrossentropy, LossBinaryCrossEntropy, LossCategoricalCrossEntropy
from optimizers import OptimizerAdagrad, OptimizerRMSProp, OptimizerSGD, OptimizerAdam
import numpy as np
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm

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
activation2 = ActivationSoftmax()

# Create optimizer
optimizer = OptimizerAdam(learning_rate=0.001, decay=5e-4)

# Create loss function
loss_function = LossCategoricalCrossEntropy()

# Train the model
for epoch in tqdm(range(2000)):
    # Forward pass
    conv1.forward(a=adjacency, h=features)
    activation1.forward(conv1.output)
    dropout.forward(activation1.output)
    conv2.forward(a=adjacency, h=dropout.output)
    activation2.forward(conv2.output)

    # Caculate loss
    loss = loss_function.calculate(conv2.output[data.train_mask], y_true[data.train_mask])

    # Calculating accuracy
    predictions = np.argmax(activation2.output, axis=1)
    train_accuracy = np.mean(predictions[data.train_mask] == y_true[data.train_mask])
    val_accuracy = np.mean(predictions[data.val_mask] == y_true[data.val_mask])

    # Print results every 100 epochs
    if not epoch % 100:
        print(f"Epoch: {epoch}, " +
              f"Loss: {loss:.3f}, " +
              f"Train acc: {train_accuracy:.3f}, " +
              f"Val acc: {val_accuracy:.3f} " +
              f"Lr: {optimizer.current_learning_rate}"
              )

    # Backward pass
    loss_function.backward(activation2.output, y_true)
    activation2.backward(loss_function.dinputs)
    conv2.backward(activation2.dinputs)
    dropout.backward(conv2.dinputs)
    activation1.backward(dropout.dinputs)
    conv1.backward(activation1.dinputs)

    # Update layer parameters
    optimizer.pre_update_params()
    optimizer.update_params(conv1)
    optimizer.update_params(conv2)
    optimizer.post_update_params()


# Validate the model
# Forward pass
conv1.forward(a=adjacency, h=features)
activation1.forward(conv1.output)
conv2.forward(a=adjacency, h=activation1.output)
activation2.forward(conv2.output)

# Caculate loss
test_loss = loss_function.calculate(conv2.output[data.test_mask], y_true[data.test_mask])

# Calculate accuracy
predictions = np.argmax(activation2.output, axis=1)
test_accuracy = np.mean(predictions[data.test_mask] == y_true[data.test_mask])

print(f"Test loss: {test_loss:.3f} Test accuracy: {test_accuracy:.3f}")


# # Calculate the data loss
# loss = loss_function.calculate(activation2.output, y_test)
#
# # Calculate accuracy from output of activation2 and targets
# # Part in the brackets returns a binary mask - array consisting of
# # True/False values, multiplying it by 1 changes it into array
# # of 1s and 0s
# predictions = (activation2.output > 0.5) * 1
# accuracy = np.mean(predictions == y_test)
#
# print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
# # NOTE: It works!
