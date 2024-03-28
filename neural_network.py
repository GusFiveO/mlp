import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from utils import random_uniform_generator, sigmoid, softmax


class Layer:
    def __init__(self, input_shape, lenght, activation, weights_initializer=None):
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.lenght = lenght
        self.input_shape = input_shape
        self.__init_weights(input_shape, weights_initializer)
        self.activations = np.zeros(shape=(1, lenght))
        self.biases = np.zeros(shape=(1, lenght))

    def __init_weights(self, input_shape, initializer):
        if initializer == "uniform":
            rand = random_uniform_generator(self.lenght * input_shape)
            self.weights = rand.reshape(input_shape, self.lenght)

    def __repr__(self) -> str:
        return (
            f"activation: {self.activation}\nactivations: {self.activations}\nlenght: {self.lenght}\ninput_shape: {self.input_shape}\nbiases: {self.biases}\ninitializer: {self.weights_initializer}\n"
            + (
                f"weights: {self.weights.shape}\nweights content: {self.weights}"
                if self.weights_initializer is not None
                else ""
            )
        )

    def forward(self, previous_activations):
        if self.activation == "sigmoid":
            self.activations = sigmoid(
                previous_activations.dot(self.weights) + self.biases
            )
            return self.activations
        elif self.activation == "softmax":
            self.activations = previous_activations.dot(self.weights) + self.biases
            self.activations = softmax(self.activations)
            return self.activations
        elif self.activation is None:
            return previous_activations
        return


class NeuralNetwork:

    def __init__(self, epochs, learning_rate, layer_shapes_list):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_shapes_list = layer_shapes_list
        self.layers = []

    def __init_layers(
        self, input_shape, output_shape, shapes_list, activation, initializer
    ):
        self.layers.append(Layer(1, input_shape, None))
        for i, layer_shape in enumerate(shapes_list):
            self.layers.append(Layer(input_shape, layer_shape, activation, initializer))
            input_shape = layer_shape
        self.layers.append(Layer(input_shape, output_shape, "softmax", initializer))

    def __normalize_features(self, features):
        features = (features - features.min()) / (features.max() - features.min())
        return features

    def __repr__(self) -> str:
        repr_string = f"epochs: {self.epochs}\nlearning_rate: {self.learning_rate}\n----------LAYERS-----------\n"
        for i, layer in enumerate(self.layers):
            repr_string += f"\nlayer n{i}:\n" + repr(layer) + "\n"
        return repr_string

    def forward_propagation(self, input_values):
        print("input values", input_values)
        tmp_activations = input_values
        for layer in self.layers:
            tmp_activations = layer.forward(tmp_activations)
            print("tmp_activation shape: ", tmp_activations.shape)
        print("output:", tmp_activations)
        return

    def fit(self, features, targets):
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        initializer = "uniform"
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        features = self.__normalize_features(features)
        self.forward_propagation(features)
