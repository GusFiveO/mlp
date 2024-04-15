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
    print(normalized_df.corr())
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        normalized_df.corr(),
        annot=True,
        cmap="coolwarm",
    )
    plt.show()
