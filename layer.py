import numpy as np

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
        # print("dz and is shape\n", dz, "\n", dz.shape, "\n", type(dz[0]))
        db = (1 / targets_lenght) * np.sum(dz, axis=1, keepdims=True)
        next_dz = (
            self.weights.T.dot(dz) * previous_activations * (1 - previous_activations)
        )
        return next_dz, dw, db

    def update(self, dw, db, learning_rate):
        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
