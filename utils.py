import pandas as pd
import numpy as np


def load_csv(path: str, header=True):
    try:
        if header:
            content = pd.read_csv(path)
        else:
            content = pd.read_csv(path, header=None)
        return content
    except Exception as e:
        print(e)
        return None


def random_uniform_generator(
    max, min, size=1, mult=7**5, seed=12345678, mod=(2**31) - 1
):
    print(seed)
    U = np.zeros(size)
    x = (seed * mult + 1) % mod
    U[0] = x / mod
    for i in range(1, size):
        x = (x * mult + 1) % mod
        U[i] = x / mod
    return min + (max - min) * U


def xavier_uniform_generator(input_shape, output_shape, seed=None):
    x = np.sqrt(6 / (input_shape + output_shape))
    return random_uniform_generator(x, -x, size=input_shape * output_shape, seed=seed)


def xavier_uniform_initializer(input_units, output_units, seed=None):
    # Seed the random number generator for reproducibility
    np.random.seed(seed)

    # Calculate the bound for initializing weights
    bound = np.sqrt(6 / (input_units + output_units))

    # Generate random numbers from a uniform distribution within the bounds
    weights = np.random.uniform(-bound, bound, size=(output_units, input_units))

    return weights


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(x):
    e_x = np.exp(x)
    ret = e_x / (e_x.sum(axis=0) + 1e-15)
    return ret
