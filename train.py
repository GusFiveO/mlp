#! /usr/bin/env python3

from neuralNetwork import NeuralNetwork
from utils import load_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def split(df: pd.DataFrame):
    print(df.shape[0])
    benign_count = df[df["2"] == "B"].shape[0]
    print("begnin_count: ", benign_count)


def train(df: pd.DataFrame):
    targets = df.pop("2")
    df = df.drop(["1"], axis=1)

    epochs = 300
    # model = NeuralNetwork(epochs, 0.001, [12, 12, 12, 2])
    model = NeuralNetwork(epochs, 0.001, [12, 12])
    # model = NeuralNetwork(epochs, 0.000314, [24, 24, 24])
    output, log_loss_history, accuracy_history = model.fit(df, targets)
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(range(0, epochs), log_loss_history["train"], label="train loss")
    axs[0].plot(range(0, epochs), log_loss_history["valid"], label="valid loss")
    axs[1].plot(range(0, epochs), accuracy_history["train"], label="train acc")
    axs[1].plot(range(0, epochs), accuracy_history["valid"], label="valid acc")
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title("Binary Cross Entropy")
    axs[1].set_title("Accuracy")
    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv("./data_mlp.csv", nrows=22)
    df = pd.read_csv("./data_mlp.csv")
    # df = pd.read_csv("./data_test.csv")
    if df is None:
        exit()
    split(df)
    train(df)
