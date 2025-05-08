import numpy as np
from tensorflow import keras

from config import N_TRAIN, N_TEST


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = x_train[:N_TRAIN], y_train[:N_TRAIN]
    x_test, y_test   = x_test[:N_TEST],   y_test[:N_TEST]
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32)  / 255.0
    # Add channel
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]
    return x_train, y_train, x_test, y_test