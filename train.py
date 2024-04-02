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
    # print(df)
    df = df.drop(["1"], axis=1)

    # print("targets:")
    # print(targets)
    # print()
    epochs = 200
    # model = NeuralNetwork(epochs, 0.001, [12, 12, 12])
    model = NeuralNetwork(epochs, 0.001, [24, 24, 24])
    # model = NeuralNetwork(epochs, 0.0001, [2, 2, 2])
    # model = NeuralNetwork(10, 0.1, [3, 2])
    # print(
    #     "---------------------\n",
    #     model.fit(df[:400], targets[:400]),
    #     "\n------------------------",
    # )
    output, log_loss_history, accuracy_history = model.fit(df[:400], targets[:400])
    _, axs = plt.subplots(2)
    axs[0].plot(range(0, epochs), log_loss_history)
    axs[1].plot(range(0, epochs), accuracy_history)
    plt.show()
    # print(model)
    # print(model.predict(df[400:]))


if __name__ == "__main__":
    # df = pd.read_csv("./data_mlp.csv", nrows=22)
    df = pd.read_csv("./data_mlp.csv")
    # df = pd.read_csv("./data_test.csv")
    if df is None:
        exit()
    split(df)
    train(df)
