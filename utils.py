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
    np.random.seed(seed)

    bound = np.sqrt(6 / (input_units + output_units))

    weights = np.random.uniform(-bound, bound, size=(output_units, input_units))
    return weights


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(x):
    e_x = np.exp(x)
    if e_x.all() != e_x.all():
        max_per_column = np.max(x, axis=0)
        x -= max_per_column
        e_x = np.exp(x)

    ret = e_x / (e_x.sum(axis=0) + 1e-15)
    return ret


def compute_binary_cross_entropy(pred, true):
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))


def compute_accuracy(pred, true):
    pred_binary = np.round(pred).astype(int)
    correct_prediction = np.sum(pred_binary == true)
    total_sample = len(true)
    return correct_prediction / total_sample


def true_positive(ground_truth, prediction):
    return np.sum((prediction == 1) & (ground_truth == 1))


def true_negative(ground_truth, prediction):
    return np.sum((prediction == 0) & (ground_truth == 0))


def false_positive(ground_truth, prediction):
    return np.sum((prediction == 1) & (ground_truth == 0))


def false_negative(ground_truth, prediction):
    return np.sum((prediction == 0) & (ground_truth == 1))


def compute_precision(pred, true):
    binary_pred = np.rint(pred).astype(int)
    true_positives = true_positive(true, binary_pred)
    false_positives = false_positive(true, binary_pred)
    return true_positives / (true_positives + false_positives + 1e-10)


def compute_recall(pred, true):
    binary_pred = np.rint(pred).astype(int)
    true_positives = true_positive(true, binary_pred)
    false_negatives = false_negative(true, binary_pred)
    return true_positives / (true_positives + false_negatives + 1e-10)


def compute_f1(pred, true):
    precision = compute_precision(pred, true)
    recall = compute_recall(pred, true)
    return 2 * precision * recall / (precision + recall + 1e-10)


def confusion_matrix(pred, true):
    tn = true_negative(true, pred)
    fp = false_positive(true, pred)
    fn = false_negative(true, pred)
    tp = true_positive(true, pred)

    confusion_matrix_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        columns=["Negative", "Positive"],
        index=["Negative", "Positive"],
    )
    return confusion_matrix_df


def normalize_features(features):
    features = (features - features.min()) / (features.max() - features.min())
    return features.to_numpy().T


def prepare_targets(targets):
    targets = pd.get_dummies(targets, dtype=int)
    return targets.to_numpy().T


def prepare_data(train_features, train_targets, validation_data):
    train_features = normalize_features(train_features)
    train_targets = prepare_targets(train_targets)
    if validation_data is not None:
        valid_features, valid_targets = validation_data
        valid_features = normalize_features(valid_features)
        valid_targets = prepare_targets(valid_targets)
        return train_features, train_targets, valid_features, valid_targets
    return train_features, train_targets, None, None


def init_histories():
    accuracy_history = {"train": [], "valid": []}
    log_loss_history = {"train": [], "valid": []}
    precision_history = {"train": [], "valid": []}
    recall_history = {"train": [], "valid": []}
    f1_history = {"train": [], "valid": []}
    return (
        accuracy_history,
        log_loss_history,
        precision_history,
        recall_history,
        f1_history,
    )


def append_to_history(histories, values, key):
    for history, value in zip(histories, values):
        history[key].append(value)
