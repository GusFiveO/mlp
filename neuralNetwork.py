import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import copy

from utils import (
    compute_accuracy,
    compute_binary_cross_entropy,
    compute_f1,
    compute_precision,
    compute_recall,
    confusion_matrix,
    init_histories,
    prepare_data,
)

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

    def __repr__(self) -> str:
        repr_string = f"epochs: {self.epochs}\nlearning_rate: {self.learning_rate}\n----------LAYERS-----------\n"
        for i, layer in enumerate(self.layers):
            repr_string += repr(layer) + "\n"
        return repr_string

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

    def __save_layers(self):
        saved_layers = []
        for layer in self.layers:
            new_layer = copy.deepcopy(layer)
            saved_layers.append(new_layer)
        self.saved_layers = saved_layers

    def __reset_saved_layers(self):
        self.saved_layers = None

    def __reset_acc(self):
        for layer in self.layers:
            layer.reset_acc()

    def __print_metrics(self, epoch, train_metrics, valid_metrics, bold=False):
        if bold:
            print("\033[1m")
            print(
                f"epoch {epoch}/{self.epochs} - valid_loss: {valid_metrics[0]:.4f} - valid_acc: {valid_metrics[1]:.4f} - valid_prec: {valid_metrics[2]:.4f} - train_loss: {train_metrics[0]:.4f} - train_acc: {train_metrics[1]:.4f} - train_prec: {train_metrics[2]:.4f}",
            )
            print("\033[0m")
        else:
            print(
                f"epoch {epoch}/{self.epochs} - valid_loss: {valid_metrics[0]:.4f} - valid_acc: {valid_metrics[1]:.4f} - valid_prec: {valid_metrics[2]:.4f} - train_loss: {train_metrics[0]:.4f} - train_acc: {train_metrics[1]:.4f} - train_prec: {train_metrics[2]:.4f}",
            )

    def early_stop(self, valid_loss, epoch, max_patience=20):
        if self.best_loss is None or valid_loss < self.best_loss:
            self.__save_layers()
            self.patience = 0
            self.best_loss = valid_loss
            self.best_epoch = epoch
        elif valid_loss > self.best_loss:
            self.patience += 1
            if self.patience >= max_patience:
                self.layers = self.saved_layers
                self.__reset_saved_layers()
                return True
        return False

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

    # def sgd(self, train_features, train_targets, batch_size=1, momentum=None):
    #     rand_index = rand.randint(0, train_features.shape[1] - 1)
    #     output = self.forward_propagation(
    #         train_features[:, rand_index : rand_index + batch_size], momentum=momentum
    #     )
    #     full_output = self.forward_propagation(
    #         train_features,
    #         training=False,
    #     )
    #     self.backward_propagation(
    #         train_targets[:, rand_index : rand_index + batch_size],
    #         output,
    #     )
    #     log_loss = compute_binary_cross_entropy(
    #         full_output[0],
    #         train_targets[0],
    #     )
    #     accuracy = compute_accuracy(
    #         full_output[0],
    #         train_targets[0],
    #     )
    #     precision = compute_precision(output[0], train_targets[0])
    #     recall = compute_recall(output[0], train_targets[0])
    #     f1 = compute_f1(output[0], train_targets[0])
    #     return output, log_loss, accuracy, precision, recall, f1

    def gd(self, train_features, train_targets, batch_size=None, momentum=None):
        if batch_size is not None:
            rand_index = rand.randint(0, train_features.shape[1] - 1)
            print(rand_index, batch_size)
            batch_output = self.forward_propagation(
                train_features[:, rand_index : rand_index + batch_size],
                momentum=momentum,
            )
            self.backward_propagation(
                train_targets[:, rand_index : rand_index + batch_size],
                batch_output,
            )
            output = self.forward_propagation(
                train_features, momentum=momentum, training=False
            )

        else:
            output = self.forward_propagation(train_features, momentum=momentum)
            self.backward_propagation(train_targets, output)
        log_loss = compute_binary_cross_entropy(output[0], train_targets[0])
        accuracy = compute_accuracy(output[0], train_targets[0])
        precision = compute_precision(output[0], train_targets[0])
        recall = compute_recall(output[0], train_targets[0])
        f1 = compute_f1(output[0], train_targets[0])
        return output, log_loss, accuracy, precision, recall, f1

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
        (
            accuracy_history,
            log_loss_history,
            precision_history,
            recall_history,
            f1_history,
        ) = init_histories()
        input_shape = features.shape[1]
        output_shape = targets.unique().shape[0]
        self.__init_layers(
            input_shape, output_shape, self.layer_shapes_list, "sigmoid", initializer
        )
        train_features, train_targets, valid_features, valid_targets = prepare_data(
            features, targets, validation_data
        )
        for epoch in range(self.epochs):
            # if batch_size is not None:
            #     (
            #         output,
            #         train_loss,
            #         train_accuracy,
            #         train_precision,
            #         train_recall,
            #         train_f1,
            #     ) = self.sgd(
            #         train_features, train_targets, batch_size, momentum=momentum
            #     )
            # else:
            #     (
            #         output,
            #         train_loss,
            #         train_accuracy,
            #         train_precision,
            #         train_recall,
            #         train_f1,
            #     ) = self.gd(train_features, train_targets, momentum=momentum)
            (
                output,
                train_loss,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
            ) = self.gd(
                train_features, train_targets, batch_size=batch_size, momentum=momentum
            )
            log_loss_history["train"].append(train_loss)
            accuracy_history["train"].append(train_accuracy)
            precision_history["train"].append(train_precision)
            recall_history["train"].append(train_recall)
            f1_history["train"].append(train_f1)
            if validation_data is not None:
                valid_output = self.forward_propagation(valid_features)
                valid_loss = compute_binary_cross_entropy(
                    valid_output[0], valid_targets[0]
                )
                log_loss_history["valid"].append(valid_loss)
                valid_accuracy = compute_accuracy(valid_output[0], valid_targets[0])
                accuracy_history["valid"].append(valid_accuracy)
                valid_precision = compute_precision(valid_output[0], valid_targets[0])
                precision_history["valid"].append(valid_precision)
                valid_recall = compute_recall(valid_output[0], valid_targets[0])
                recall_history["valid"].append(valid_recall)
                valid_f1 = compute_f1(valid_output[0], valid_targets[0])
                f1_history["valid"].append(valid_f1)
            self.__print_metrics(
                epoch,
                (train_loss, train_accuracy, train_precision),
                (valid_loss, valid_accuracy, valid_precision),
            )
            if self.early_stop(valid_loss, epoch, max_patience=100) is True:
                break

        self.__print_metrics(
            self.best_epoch,
            (
                log_loss_history["train"][self.best_epoch],
                accuracy_history["train"][self.best_epoch],
                precision_history["train"][self.best_epoch],
            ),
            (
                log_loss_history["valid"][self.best_epoch],
                accuracy_history["valid"][self.best_epoch],
                precision_history["valid"][self.best_epoch],
            ),
            bold=True,
        )
        self.__reset_acc()
        output = np.rint(self.forward_propagation(valid_features)[0]).astype(int)
        cm = confusion_matrix(output, valid_targets[0])
        return (
            output,
            log_loss_history,
            accuracy_history,
            precision_history,
            recall_history,
            f1_history,
            cm,
            self.best_epoch,
        )

    def predict(self, data):
        data = self.__normalize_features(data)
        return self.forward_propagation(data)
