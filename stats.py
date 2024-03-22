#! /usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import load_csv

matplotlib.use("TkAgg")

if __name__ == "__main__":
    df = load_csv("./data_mlp.csv")
    if df is None:
        exit()
    diagnosis_row = df["2"]
    df = df.select_dtypes(include=np.number)
    normalized_df = (df - df.min()) / (df.max() - df.min()) * 2 - 1
    normalized_df.insert(0, "diagnosis", diagnosis_row)
    colors = normalized_df["diagnosis"].replace({"B": "red", "M": "green"})
    print(normalized_df)
    pd.plotting.scatter_matrix(normalized_df, c=colors)
    df = df.replace(to_replace="None", value=np.nan)
    print(len(df.dropna()))
    plt.show()
