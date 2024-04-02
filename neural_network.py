import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("TkAgg")

from utils import random_uniform_generator, sigmoid, softmax


class Layer:
    def __init__(
        self,
        input_shape,
        lenght,
        activation,
        weights_initializer=None,
        is_last=False,
        index=None,
    ):
        self.index = index
        self.is_last = is_last
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.lenght = lenght
        self.input_shape = input_shape
        self.__init_weights(input_shape, weights_initializer)
        self.activations = np.zeros(shape=(1, lenght))
        self.biases = np.zeros(shape=(lenght, 1))

    def __init_weights(self, input_shape, initializer):
        if initializer == "uniform":
            rand = -1 + (1 + 1) * random_uniform_generator(
                self.lenght * input_shape,
                seed=(self.index + 1) * self.input_shape * self.lenght * 100,
            )
            self.weights = rand.reshape(self.lenght, input_shape)

    def __repr__(self) -> str:
        return (
            f"activation: {self.activation}\nactivations: {self.activations}\nshape: {self.activations.shape}\nbiases: {self.biases}\ninitializer: {self.weights_initializer}\n"
            + (
                f"weights: {self.weights.shape}\nweights content: {self.weights}"
                if self.weights_initializer is not None
                else ""
            )
        )

    def get_activations(self):
        return self.activations

    def forward(self, previous_activations):
        if self.activation == "sigmoid":
            self.activations = sigmoid(
                self.weights.dot(previous_activations) + self.biases
            )
            return self.activations
        elif self.activation == "softmax":
            activations = self.weights.dot(previous_activations) + self.biases
            self.activations = softmax(activations)
            return self.activations
        elif (
            self.activation is None
        ):  # if is the input layer so it will store the features
            self.activations = previous_activations
            return previous_activations
        return

    def backward(self, dz, previous_activations, targets):
        targets_lenght = targets.shape[0]
        if dz is None:
            dz = self.activations - targets
        dw = (1 / targets_lenght) * dz.dot(previous_activations.T)
        db = (1 / targets_lenght) * np.sum(dz, axis=1, keepdims=True)
        next_dz = (
            self.weights.T.dot(dz) * previous_activations * (1 - previous_activations)
        )
        return next_dz, dw, db

    def update(self, dw, db, learning_rate):
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db


class NeuralNetwork:

    def __init__(self, epochs, learning_rate, layer_shapes_list):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_shapes_list = layer_shapes_list
        self.layers = []

    def __init_layers(
        self, input_shape, output_shape, shapes_list, activation, initializer
    ):
        self.layers.append(Layer(1, input_shape, None, index=1))
        for i, layer_shape in enumerate(shapes_list):
            self.layers.append(
                Layer(input_shape, layer_shape, activation, initializer, index=i)
            )
            input_shape = layer_shape
        self.layers.append(
            Layer(
                input_shape, output_shape, "softmax", initializer, is_last=True, index=i
            )
        )

    def __normalize_features(self, features):
        features = (features - features.min()) / (features.max() - features.min())
        return features.to_numpy().T

    def __prepare_targets(self, targets):
        targets = pd.get_dummies(targets, dtype=float)
        return targets.to_numpy().T

    def __repr__(self) -> str:
        repr_string = f"epochs: {self.epochs}\nlearning_rate: {self.learning_rate}\n----------LAYERS-----------\n"
        for i, layer in enumerate(self.layers):
            repr_string += f"\nlayer n{i}:\n" + repr(layer) + "\n"
        return repr_string

    def forward_propagation(self, input_values):
        tmp_activations = input_values
        for layer in self.layers:
            tmp_activations = layer.forward(tmp_activations)
        return tmp_activations

    def backward_propagation(self, targets, output):
        tmp_dz = None
        reversed_layers = list(reversed(self.layers))

        for idx, layer in enumerate(reversed_layers[:-1]):
            tmp_dz, dw, db = layer.backward(
                tmp_dz,
                reversed_layers[idx + 1].get_activations(),
                targets,
            )
            layer.update(dw, db, self.learning_rate)
        return

    def __compute_log_loss(self, pred, true):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return - np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))

    def __compute_accuracy(self, pred, true):
        pred_binary = np.round(pred).astype(int)
        correct_prediction = np.sum(pred_binary == true)
        total_sample = len(true)
        return correct_prediction / total_sample

    def fit(self, features, targets):
        log_loss_history = []
        accuracy_history = []
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        initializer = "uniform"
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        features = self.__normalize_features(features)
        targets = self.__prepare_targets(targets)
        for _ in range(self.epochs):
            output = self.forward_propagation(features)
            self.backward_propagation(targets, output)
            log_loss_history.append(self.__compute_log_loss(output[0], targets[0]))
            accuracy_history.append(self.__compute_accuracy(output[0], targets[0]))
            print("---------------")

        return (output, log_loss_history, accuracy_history)

    def predict(self, data):
        data = self.__normalize_features(data)
        return self.forward_propagation(data)
