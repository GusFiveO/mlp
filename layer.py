import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from utils import (
    random_uniform_generator,
    sigmoid,
    softmax,
    xavier_uniform_generator,
    xavier_uniform_initializer,
)


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
        self.acc = None

    def __init_weights(self, input_shape, initializer):
        if initializer == "Uniform":
            rand = random_uniform_generator(
                -1,
                1,
                self.lenght * input_shape,
                seed=(self.index + 1) * self.input_shape * self.lenght * 100,
            )
            self.weights = rand.reshape(self.lenght, input_shape)
        elif initializer == "XavierUniform":
            weights = xavier_uniform_initializer(
                self.input_shape,
                self.lenght,
                seed=(self.index + 1) * self.input_shape * self.lenght * 100,
            )
            self.weights = weights
        else:
            self.weights = None

    def __repr__(self) -> str:
        return (
            f"layer n{self.index}\n"
            + f"activation: {self.activation}\nactivations: {self.activations}\nshape: {self.activations.shape}\nbiases: {self.biases}\ninitializer: {self.weights_initializer}\n"
            + (
                f"weights: {self.weights.shape}\nweights content: {self.weights}"
                if self.weights_initializer is not None
                else ""
            )
        )

    def reset_acc(self):
        self.acc = None

    def get_activations(self):
        return self.activations

    def forward(self, previous_activations, training=True, momentum=None):
        new_activations = None
        weights = self.weights
        biases = self.biases
        if momentum and weights is not None:
            if self.acc is None:
                self.acc = {"weights": 0, "biases": 0}
                self.acc["weights"] = 0
                self.acc["biases"] = 0

            self.acc["weights"] *= momentum
            self.acc["biases"] *= momentum
            weights -= self.acc["weights"]
            biases -= self.acc["biases"]
        if self.activation == "sigmoid":
            new_activations = sigmoid(weights.dot(previous_activations) + biases)
        elif self.activation == "softmax":
            tmp = weights.dot(previous_activations) + biases
            new_activations = softmax(tmp)
        elif (
            self.activation is None
        ):  # if is the input layer so it will store the features
            new_activations = previous_activations
        if training is True:
            self.activations = new_activations
        return new_activations

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
        if self.acc is not None:
            self.weights += self.acc["weights"] - learning_rate * dw
            self.biases += self.acc["biases"] - learning_rate * db
            self.acc["weights"] = self.acc["weights"] - learning_rate * dw
            self.acc["biases"] = self.acc["biases"] - learning_rate * db
        else:
            self.weights -= learning_rate * dw
            self.biases -= learning_rate * db
