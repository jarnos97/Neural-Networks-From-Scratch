from layers import LayerInput
from activation_functions import ActivationSoftmax
from loss import LossCategoricalCrossEntropy, ActivationSoftmaxLossCategoricalCrossentropy


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = self.loss = self.optimizer = self.accuracy = self.input_layer = \
            self.trainable_layers = self.output_layer_activation = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        # Create and set the input layer
        self.input_layer = LayerInput()

        # Count layers
        layer_count = len(self.layers)

        # Define trainable layers
        self.trainable_layers = []

        # Loop over layers
        for i in range(layer_count):
            # If first layer, prevous layer is the input
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All non-first and non-last layers
            elif i < (layer_count - 1):
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # If last layer, next object is lsos
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                # Save the models output (from the last layer)
                self.output_layer_activation = self.layers[i]

            # If layer has weights, its a trainable layer.
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy. Create combined object
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossEntropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy
        self.accuracy.init(y)

        # Train loop
        for epoch in range(1, epochs+1):
            # Forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Make predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Backward pass
            self.backward(output, y)

            # Update parameters
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Summary
            if not epoch % print_every:
                print(f"epoch: {epoch}, " +
                      f"acc: {accuracy:.3f}, " +
                      f"loss: {loss:.3f}," +
                      f"data_loss: {data_loss:.3f}, " +
                      f"reg_loss: {regularization_loss:.3f}, " +
                      f"lr: {self.optimizer.current_learning_rate:.3f}\n")

        # If there is validation data
        if validation_data is not None:
            X_val, y_val = validation_data

            # Forward pass
            output = self.forward(X_val, training=False)

            # Calculate loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Summary
            print(f"validation accuracy {accuracy:.3f}, " +
                  f"validation loss: {loss:.3f}")

    def forward(self, X, training: bool):
        # Call forward method on the input layer
        self.input_layer.forward(X, training)

        # Call forward method on layers
        for layer in self.layers:
            layer.forward(layer.prev.output, training=training)

        # Layer is now the last object in the list. We can return its output
        return layer.output

    def backward(self, output, y):
        # IF softmax classifier
        if self.softmax_classifier_output is not None:
            # Call backward on the combined object. This will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Set dinputs for the last layer, since we do not call backward on it
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward for all layers but the last in reverse order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # ELSE: call backward method on the loss
        self.loss.backward(output, y)

        # Call backward for all layers in reversed order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
