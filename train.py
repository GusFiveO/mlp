#! /usr/bin/env python3

from neural_network import NeuralNetwork
from utils import load_csv
import pandas as pd
import numpy as np


def train(df: pd.DataFrame):
    targets = df.pop("2")
    # print(df)
    df = df.drop(["1"], axis=1)

    # print("targets:")
    # print(targets)
    # print()
    model = NeuralNetwork(1000, 0.00001, [24, 24, 24])
    # model = NeuralNetwork(10, 0.1, [3, 2])
    print(
        "---------------------\n",
        model.fit(df[:400], targets[:400]),
        "\n------------------------",
    )
    print(model)
    print(model.predict(df[400:]))


if __name__ == "__main__":
    # df = pd.read_csv("./data_mlp.csv", nrows=22)
    df = pd.read_csv("./data_mlp.csv")
    # df = pd.read_csv("./data_test.csv")
    if df is None:
        exit()
    train(df)
