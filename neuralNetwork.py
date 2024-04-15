import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.use("TkAgg")

from layer import Layer


class NeuralNetwork:

    def __init__(self, epochs, learning_rate, layer_shapes_list):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layer_shapes_list = layer_shapes_list
        self.layers = []
        self.saved_layers = None
        self.patience = 0
        self.best_loss = None
        self.best_epoch = None

    def __init_layers(
        self, input_shape, output_shape, shapes_list, activation, initializer
    ):
        i = 0
        self.layers.append(Layer(1, input_shape, None, index=i))
        for layer_shape in shapes_list:
            i += 1
            self.layers.append(
                Layer(input_shape, layer_shape, activation, initializer, index=i)
            )
            input_shape = layer_shape
        self.layers.append(
            Layer(
                input_shape,
                output_shape,
                "softmax",
                initializer,
                index=i + 1,
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
            repr_string += repr(layer) + "\n"
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

    @classmethod
    def split(cls, features, targets, train_percent=80):
        features_split_index = int(len(features) * train_percent / 100)
        targets_split_index = int(len(targets) * train_percent / 100)
        train_targets = targets.iloc[:targets_split_index]
        train_features = features.iloc[:features_split_index]
        valid_targets = targets.iloc[targets_split_index:]
        valid_features = features.iloc[features_split_index:]
        return train_features, train_targets, (valid_features, valid_targets)

    @classmethod
    def stratified_train_test_split(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        df = pd.concat([X, y], axis=1)
        target_column = y.name

        class_counts = y.value_counts()
        test_counts = (class_counts * test_size).astype(int)

        train_indices = []
        test_indices = []

        for class_value in class_counts.index:
            class_indices = df[df[target_column] == class_value].index.to_list()
            # np.random.seed(random_state)
            np.random.shuffle(class_indices)

            test_idx = class_indices[: test_counts[class_value]]
            train_idx = class_indices[test_counts[class_value] :]

            train_indices.extend(train_idx)
            test_indices.extend(test_idx)

        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]

        return X_train, y_train, (X_test, y_test)

    def __prepare_data(self, train_features, train_targets, validation_data):
        train_features = self.__normalize_features(train_features)
        train_targets = self.__prepare_targets(train_targets)
        if validation_data is not None:
            valid_features, valid_targets = validation_data
            valid_features = self.__normalize_features(valid_features)
            valid_targets = self.__prepare_targets(valid_targets)
            return train_features, train_targets, valid_features, valid_targets
        return train_features, train_targets, None, None

    def __save_layers(self):
        self.saved_layers = self.layers

    def __reset_saved_layers(self):
        self.saved_layers = None

    def __reset_acc(self):
        for layer in self.layers:
            layer.reset_acc()

    def __print_metrics(self, epoch, train_metrics, valid_metrics, bold=False):
        if bold:
            print("\033[1m")
            print(
                f"epoch {epoch}/{self.epochs} - valid_loss: {valid_metrics[0]:.4f} - valid_acc: {valid_metrics[1]:.4f} - train_loss: {train_metrics[0]:.4f} - train_acc: {train_metrics[1]:.4f} ",
            )
            print("\033[0m")
        else:
            print(
                f"epoch {epoch}/{self.epochs} - valid_loss: {valid_metrics[0]:.4f} - valid_acc: {valid_metrics[1]:.4f} - train_loss: {train_metrics[0]:.4f} - train_acc: {train_metrics[1]:.4f} ",
            )

    def save(self, path):
        infos_list = []
        for layer in self.layers:
            infos_list.append(layer.get_infos())
        with open(path + "saved_model.pkl", "wb") as fd:
            pickle.dump(infos_list, fd)
            print("model succesfuly saved !")

    def load(self, path):
        with open(path, "rb") as fp:
            layers_info = pickle.load(fp)
            new_layers = []
            for layer_info in layers_info:
                new_layer = Layer(
                    layer_info["input_shape"],
                    layer_info["lenght"],
                    None,
                    layer_info=layer_info,
                )
                new_layers.append(new_layer)
            self.layers = new_layers
        return

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

    def early_stop(self, valid_loss, epoch, max_patience=20):
        if self.best_loss is None or valid_loss < self.best_loss:
            self.__save_layers()
            self.best_loss = valid_loss
            self.best_epoch = epoch
            if self.patience != 0:
                self.patience = 0
                self.__reset_saved_layers()
        elif valid_loss > self.best_loss:
            self.patience += 1
            if self.patience >= max_patience:
                return True
        return False

    def fit(
        self,
        features,
        targets,
        initializer,
        validation_data=None,
        batch_size=None,
        momentum=None,
    ):
        self.best_loss = None
        self.best_epoch = None
        accuracy_history = {"train": [], "valid": []}
        log_loss_history = {"train": [], "valid": []}
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        train_features, train_targets, valid_features, valid_targets = (
            self.__prepare_data(features, targets, validation_data)
        )
        for epoch in range(self.epochs):
            if batch_size is not None:
                output, train_loss, train_accuracy = self.sgd(
                    train_features, train_targets, batch_size, momentum=momentum
                )
            else:
                output, train_loss, train_accuracy = self.gd(
                    train_features, train_targets, momentum=momentum
                )
            log_loss_history["train"].append(train_loss)
            accuracy_history["train"].append(train_accuracy)
            if validation_data is not None:
                valid_output = self.forward_propagation(valid_features)
                valid_loss = self.__compute_binary_cross_entropy(
                    valid_output[0], valid_targets[0]
                )
                log_loss_history["valid"].append(valid_loss)
                valid_accuracy = self.__compute_accuracy(
                    valid_output[0], valid_targets[0]
                )
                accuracy_history["valid"].append(valid_accuracy)
            self.__print_metrics(
                epoch, (train_loss, train_accuracy), (valid_loss, valid_accuracy)
            )
            if self.early_stop(valid_loss, epoch, max_patience=100) is True:
                break

        self.__print_metrics(
            self.best_epoch,
            (
                log_loss_history["train"][self.best_epoch],
                accuracy_history["train"][self.best_epoch],
            ),
            (
                log_loss_history["valid"][self.best_epoch],
                accuracy_history["valid"][self.best_epoch],
            ),
            bold=True,
        )
        if self.saved_layers is not None:
            self.layers = self.saved_layers
            self.__reset_saved_layers()
        self.__reset_acc()
        return (output, log_loss_history, accuracy_history, self.best_epoch)

    def predict(self, data):
        data = self.__normalize_features(data)
        return self.forward_propagation(data)
