from layers import LayerInput
from activation_functions import ActivationSoftmax
from loss import LossCategoricalCrossEntropy, ActivationSoftmaxLossCategoricalCrossentropy
import pickle


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = self.loss = self.optimizer = self.accuracy = self.input_layer = \
            self.trainable_layers = self.output_layer_activation = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy. Create combined object
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, LossCategoricalCrossEntropy):
            self.softmax_classifier_output = ActivationSoftmaxLossCategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy
        self.accuracy.init(y)

        # Default value if no batch size
        train_steps = 1

        # if no validation data setd defaul steps for validation
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. Add one to include remaining data
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                # Dividing rounds down. Add one to include remaining data
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Train loop
        for epoch in range(1, epochs+1):
            print(f"Epoch: {epoch}")

            # Reset accumulated vallues in loss and accuracy
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                # If batxhsize not set train on full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]  # NOTE: will only work for 2-d data (?)
                    batch_y = y[step*batch_size:(step+1)*batch_size]  # NOTE: will only work for 2-d data (?)

                # Forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Make predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backward pass
                self.backward(output, batch_y)

                # Update parameters
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Summary
                if not step % print_every or step == train_steps - 1:
                    print(f"epoch: {step}, " +
                          f"acc: {accuracy:.3f}, " +
                          f"loss: {loss:.3f}, " +
                          f"data_loss: {data_loss:.3f}, " +
                          f"reg_loss: {regularization_loss:.3f}, " +
                          f"lr: {self.optimizer.current_learning_rate:.3f}\n")

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularzation=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} ' +
                  f'data_loss: {epoch_data_loss:.3f} ' +
                  f'reg_loss: {epoch_regularization_loss:.3f }' +
                  f'lr: {self.optimizer.current_learning_rate}\n')

            # If there is validation data
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

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

    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if no batch size
        validation_steps = 1

        # Calculate num steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. Add one to include remaining data
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If no batch size train on full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            # Forward pass
            output = self.forward(batch_X, training=False)

            # Calculate loss
            loss = self.loss.calculate(output, batch_y)

            # Get predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Summary
        print(f"validation accuracy {validation_accuracy:.3f}, " +
              f"validation loss: {validation_loss:.3f}")

    def get_parameters(self):
        parameters = []
        # Loop over trainable layers
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        # Iterate over parameters and layers and update each layer
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path: str):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))