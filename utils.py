import pandas as pd
import numpy as np


def load_csv(path: str):
    try:
        content = pd.read_csv(path)
        return content
    except Exception as e:
        print(e)
        return None


def random_uniform_generator(size=1, mult=7**5, seed=12345678, mod=(2**31) - 1):
    U = np.zeros(size)
    x = (seed * mult + 1) % mod
    U[0] = x / mod
    for i in range(1, size):
        x = (x * mult + 1) % mod
        U[i] = x / mod
    return U


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(x):
    e_x = np.exp(x)
    ret = e_x / e_x.sum(axis=0)
    return ret
