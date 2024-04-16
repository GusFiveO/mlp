#! /usr/bin/env python3

import argparse
from neuralNetwork import NeuralNetwork
from utils import load_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

matplotlib.use("TkAgg")

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


def train(df: pd.DataFrame, args):
    df = df.sample(frac=1)
    targets = df["diagnosis"]

    df = df.drop(["diagnosis", "id"], axis=1)

    if args.split or args.train:
        train_features, train_targets, validation_data = (
            NeuralNetwork.stratified_train_test_split(df, targets)
        )

        if args.split:
            plt.hist(train_targets)
            plt.hist(validation_data[1])
            plt.legend(["train", "validation"])
            plt.show()
            return

    model = NeuralNetwork(args.epochs, args.learning_rate, args.shape)

    (
        output,
        log_loss_history,
        accuracy_history,
        precision_history,
        recall_history,
        f1_history,
        cm,
        best_epochs,
    ) = model.fit(
        train_features,
        train_targets,
        validation_data=validation_data,
        initializer="XavierUniform",
        batch_size=args.batch_size,
        momentum=args.momentum,
    )

    result_epochs = len(log_loss_history["train"])
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    _, axs = plt.subplots(1, 5, figsize=(25, 5))
    axs[0].plot(range(0, result_epochs), log_loss_history["train"], label="train loss")
    axs[0].plot(range(0, result_epochs), log_loss_history["valid"], label="valid loss")
    axs[1].plot(range(0, result_epochs), accuracy_history["train"], label="train acc")
    axs[1].plot(range(0, result_epochs), accuracy_history["valid"], label="valid acc")
    axs[2].plot(
        range(0, result_epochs), precision_history["train"], label="train precision"
    )
    axs[2].plot(
        range(0, result_epochs), precision_history["valid"], label="valid precision"
    )
    axs[3].plot(range(0, result_epochs), recall_history["train"], label="train recall")
    axs[3].plot(range(0, result_epochs), recall_history["valid"], label="valid recall")
    axs[4].plot(range(0, result_epochs), f1_history["train"], label="train f1")
    axs[4].plot(range(0, result_epochs), f1_history["valid"], label="valid f1")
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
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    axs[0].set_title("Binary Cross Entropy")
    axs[1].set_title("Accuracy")
    axs[2].set_title("Precision")
    axs[3].set_title("Recall")
    axs[4].set_title("F1")
    plt.show()
    model.save("./")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multilayer perceptron")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-split", action="store_true")
    parser.add_argument("-predict", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--batch_size", type=float)
    parser.add_argument("--shape", type=int, nargs="+", default=[10, 10])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("path", type=str, help="Path to the file or directory")
    args = parser.parse_args()

    if sum([args.split, args.train, args.predict]) != 1:
        parser.error("Exactly one of -split, -train, -predict must be provided.")

    df = pd.read_csv(args.path, header=None)
    if df is None:
        exit()

    if args.predict:
        df = df.drop([df.columns[0]], axis=1)
        try:
            model = NeuralNetwork(None, None, None)
            model.load("./saved_model.pkl")
            output = model.predict(df)
            prediction = pd.Series(output[1].T).round().astype(int)
            prediction.replace({1: "M", 0: "B"}).to_csv(
                "prediction.csv", index=False, header=None
            )
        except Exception as e:
            print("Could'nt load weights")
            print(e)
        exit()

    df.columns = columns_titles
    train(df, args)
