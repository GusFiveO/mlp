#! /usr/bin/env python3

from neuralNetwork import NeuralNetwork
from utils import load_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def train(df: pd.DataFrame):
    targets = pd.DataFrame(df.pop("2"))
    df = df.drop(["1"], axis=1)

    epochs = 500
    # epochs = 150
    # epochs = 2000
    epochs = 2500
    model = NeuralNetwork(epochs, 0.02, [10, 10])  # for gd
    # model = NeuralNetwork(epochs, 0.001, [8, 8])  # for gd

    # model = NeuralNetwork(epochs, 0.1, [15])
    output, log_loss_history, accuracy_history, best_epochs = model.fit(
        df,
        targets,
        initializer="XavierUniform",
        # initializer="Uniform",
        # batch_size=8,
        momentum=0.9,
    )

    result_epochs = len(log_loss_history["train"])
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(range(0, result_epochs), log_loss_history["train"], label="train loss")
    axs[0].plot(range(0, result_epochs), log_loss_history["valid"], label="valid loss")
    axs[1].plot(range(0, result_epochs), accuracy_history["train"], label="train acc")
    axs[1].plot(range(0, result_epochs), accuracy_history["valid"], label="valid acc")
    if best_epochs is not None:
        axs[0].scatter(
            best_epochs,
            log_loss_history["valid"][best_epochs],
            c="red",
            s=100,
            marker="|",
        )
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Binary Cross Entropy")
    axs[1].set_title("Accuracy")
    print(log_loss_history["valid"][-1])
    print(accuracy_history["valid"][-1])
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./data_mlp.csv")
    if df is None:
        exit()
    train(df)
