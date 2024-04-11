#! /usr/bin/env python3

from neuralNetwork import NeuralNetwork
from utils import load_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


def train(df: pd.DataFrame):
    targets = pd.DataFrame(df.pop("diagnosis"))
    df = df.drop(["id"], axis=1)

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
    df = pd.read_csv("./data/data.csv", header=None)
    if df is None:
        exit()
    columns_titles = [
        "id",
        "diagnosis",
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    df.columns = columns_titles
    print(df)
    train(df)
