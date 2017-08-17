import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)   # prevent overflow
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x
