import numpy as np


def MAD(x):
    return np.nanmedian(np.abs(x - np.nanmedian(x)))


def phasefold(t, per, epo):
    # Phase will span -0.5 to 0.5, with transit centred at phase 0
    phase = np.mod(t - epo, per) / per
    phase[phase > 0.5] -= 1
    return phase


def weighted_mean(y, dy):
    w = 1 / dy**2
    mean = np.sum(w * y) / np.sum(w)
    return mean


def weighted_err(y, dy):
    w = 1 / dy**2
    err = 1 / np.sqrt(np.sum(w))
    return err


def weighted_std(y, dy):
    w = 1 / dy**2
    N = len(w)
    mean = np.sum(w * y) / np.sum(w)
    std = np.sqrt(np.sum(w * (y - mean) ** 2) / ((N - 1) * np.sum(w) / N))
    return std
