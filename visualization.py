#! /usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from utils import load_csv

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

if __name__ == "__main__":
    try:
        pathname = sys.argv[1]
        df = load_csv(pathname, header=False)
        if df is None:
            exit()
    except Exception:
        print("Please provide a valid pathname")
        exit()
    df.columns = columns_titles
    diagnosis_row = df.pop("diagnosis")
    df = df.drop(["id"], axis=1)
    df = df.select_dtypes(include=np.number)
    normalized_df = (df - df.min()) / (df.max() - df.min()) * 2 - 1
    diagnosis_row = diagnosis_row.map({"M": 1, "B": 0})
    normalized_df = pd.concat([normalized_df, pd.DataFrame(diagnosis_row)], axis=1)

    correlation = normalized_df.corr().abs()
    threshold = 0.9

    # Find pairs with high correlation using np.where and np.triu_indices
    high_corr_pairs = np.where(np.triu(correlation > threshold, k=1))

    high_corr_count = []
    for i, col in enumerate(normalized_df.columns):
        num_high_corr = np.sum(
            high_corr_pairs[0] == i
        )  # Count occurrences of the column index in high_corr_pairs
        high_corr_count.append((col, num_high_corr))

    sorted_high_corr_count = sorted(high_corr_count, key=lambda x: x[1], reverse=True)
    high_corr_columns = []
    for column, count in sorted_high_corr_count:
        if count > 0:
            high_corr_columns.append(column)

    low_corr = normalized_df.drop(high_corr_columns, axis=1).corr().abs()
    high_corr = normalized_df[high_corr_columns].corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.show()
    sns.heatmap(low_corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.show()
    sns.heatmap(high_corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.show()
