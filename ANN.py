import numpy as np

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]  # batch-size = 3 (input size 4)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  # Defines the random range of weights, can be altered
        self.biases = np.zeros((1, n_neurons))  # Initialize each bias as zero
        self.output = None
        pass

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = LayerDense(n_inputs=4, n_neurons=5)
layer2 = LayerDense(n_inputs=5, n_neurons=2)

layer1.forward(inputs=X)
print(layer1.output)
layer2.forward(inputs=layer1.output)
print(layer2.output)