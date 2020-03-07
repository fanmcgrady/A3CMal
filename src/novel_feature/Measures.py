import numpy as np


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    m = np.arange(rows)
    n = y_true.astype(int) - 1
    actual[m, n] = 1
    # vsota = np.sum(actual * np.log(predictions)+(1-actual)*np.log(1-predictions))
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota
