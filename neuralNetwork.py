import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from layer import Layer


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
        targets = pd.get_dummies(targets, dtype=int)
        return targets.to_numpy().T

    def __repr__(self) -> str:
        repr_string = f"epochs: {self.epochs}\nlearning_rate: {self.learning_rate}\n----------LAYERS-----------\n"
        for i, layer in enumerate(self.layers):
            repr_string += f"\nlayer n{i}:\n" + repr(layer) + "\n"
        return repr_string

    def __compute_binary_cross_entropy(self, pred, true):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1 - epsilon)
        return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))

    def __compute_accuracy(self, pred, true):
        pred_binary = np.round(pred).astype(int)
        correct_prediction = np.sum(pred_binary == true)
        total_sample = len(true)
        return correct_prediction / total_sample

    def __shuffle_data(self, features, targets):
        shuffled_indices = list(range(features.shape[1]))
        rand.Random(3).shuffle(shuffled_indices)
        shuffled_features = np.ndarray(features.shape)
        shuffled_targets = np.ndarray(targets.shape)
        for idx, shuffled_idx in enumerate(shuffled_indices):
            shuffled_features[:, idx] = features[:, shuffled_idx]
            shuffled_targets[:, idx] = targets[:, shuffled_idx]
        shuffled_features = shuffled_features
        shuffled_targets = shuffled_targets
        return shuffled_features, shuffled_targets

    def __prepare_data(self, features, targets):
        features = self.__normalize_features(features)
        targets = self.__prepare_targets(targets)
        features, targets = self.__shuffle_data(features, targets)
        return features, targets

    def forward_propagation(self, input_values, training=True, momentum=None):
        tmp_activations = input_values
        for layer in self.layers:
            tmp_activations = layer.forward(
                tmp_activations, training, momentum=momentum
            )
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

    def sgd(self, train_features, train_targets, batch_size=1, momentum=None):
        rand_index = rand.randint(0, train_features.shape[1] - 1)
        output = self.forward_propagation(
            train_features[:, rand_index : rand_index + batch_size], momentum=momentum
        )
        full_output = self.forward_propagation(
            train_features,
            training=False,
        )
        self.backward_propagation(
            train_targets[:, rand_index : rand_index + batch_size],
            output,
        )
        log_loss = self.__compute_binary_cross_entropy(
            full_output[0],
            train_targets[0],
        )
        accuracy = self.__compute_accuracy(
            full_output[0],
            train_targets[0],
        )
        return output, log_loss, accuracy

    def gd(self, train_features, train_targets, momentum=None):
        output = self.forward_propagation(train_features, momentum=momentum)
        self.backward_propagation(train_targets, output)
        log_loss = self.__compute_binary_cross_entropy(output[0], train_targets[0])
        accuracy = self.__compute_accuracy(output[0], train_targets[0])
        return output, log_loss, accuracy

    def fit(self, features, targets, initializer, batch_size=None, momentum=None):
        accuracy_history = {"train": [], "valid": []}
        log_loss_history = {"train": [], "valid": []}
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        features, targets = self.__prepare_data(features, targets)
        split_index = int(features.shape[1] * 0.80)
        train_features = features[:, :split_index]
        valid_features = features[:, split_index:]
        train_targets = targets[:, :split_index]
        valid_targets = targets[:, split_index:]
        for _ in range(self.epochs):
            if batch_size is not None:
                output, log_loss, accuracy = self.sgd(
                    train_features, train_targets, batch_size, momentum=momentum
                )
            else:
                output, log_loss, accuracy = self.gd(
                    train_features, train_targets, momentum=momentum
                )
            log_loss_history["train"].append(log_loss)
            accuracy_history["train"].append(accuracy)
            valid_output = self.forward_propagation(valid_features)
            log_loss_history["valid"].append(
                self.__compute_binary_cross_entropy(valid_output[0], valid_targets[0])
            )
            accuracy_history["valid"].append(
                self.__compute_accuracy(valid_output[0], valid_targets[0])
            )
            # break

        return (output, log_loss_history, accuracy_history)

    def predict(self, data):
        data = self.__normalize_features(data)
        return self.forward_propagation(data)
