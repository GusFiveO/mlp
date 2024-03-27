import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from utils import random_uniform_generator


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
            self.weights = rand.reshape(self.lenght, input_shape)

    def __repr__(self) -> str:
        return (
            f"activation: {self.activation}\nactivations: {self.activations}\nlenght: {self.lenght}\ninput_shape: {self.input_shape}\nbiases: {self.biases}\ninitializer: {self.weights_initializer}\n"
            + (
                f"weights: {self.weights.shape}\nweights content: {self.weights}"
                if self.weights_initializer is not None
                else ""
            )
        )


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
        features = (features - features.max() / features.min()) * 2 - 1
        return features

    def __repr__(self) -> str:
        repr_string = f"epochs: {self.epochs}\nlearning_rate: {self.learning_rate}\n----------LAYERS-----------\n"
        for i, layer in enumerate(self.layers):
            repr_string += f"\nlayer n{i}:\n" + repr(layer) + "\n"
        return repr_string

    def forward_propagation(self, features, targets):
        for index, row in features.iterrows():
            print(index)
            print(np.array(row.values))

    def fit(self, features, targets):
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        initializer = "uniform"
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        self.forward_propagation(features, targets)
