import numpy as np


def weak(metrics, thresh=9, message="FA: signal too weak"):
    return (metrics["MES"] < thresh) | (metrics["new_MES"] < thresh), message


def invalid_transits(metrics, thresh=2, message="FA: not enough valid transits"):
    return (metrics["new_N_transit"] < thresh), message


def noisy(metrics, thresh=4, message="FA: too much noise at transit timescale"):
    return metrics["Fred"] > thresh, message


def bad_shape(metrics, thresh=0.4, message="FA: bad transit shape"):
    return (metrics["SHP"] > thresh) & (metrics["per"] < 10), message


def non_unique(
    metrics, thresh1=0, thresh2=1, thresh3=0, message="FA: events not unique"
):
    MS1 = (metrics["sig_pri"] / metrics["Fred"] - metrics["FA1"]) < thresh1
    MS2 = (metrics["sig_pri"] - metrics["sig_ter"] - metrics["FA2"]) < thresh2
    MS3 = (metrics["sig_pri"] - metrics["sig_pos"] - metrics["FA2"]) < thresh3
    return MS1 | MS2 | MS3, message


def chases(metrics, thresh=0.8, message="FA: failed 'chases' tests"):
    return (metrics["N_transit"] <= 5) & (metrics["med_chases"] < thresh), message


def dmm(metrics, thresh=1.5, message="FA: inconsistent transit depths"):
    return metrics["DMM"] > thresh, message


def single_event(metrics, thresh=0.8, message="FA: dominated by single event"):
    return (metrics["max_SES"] / metrics["MES"] > thresh) & (
        metrics["N_transit"] <= 5
    ), message


def bad_fit(metrics, thresh=-20, message="FA: bad transit model fit"):
    fit_failed = np.isnan(metrics["transit_aic"])
    chisqr = metrics["transit_chisqr"] > metrics["line_chisqr"]
    aic = metrics["transit_aic"] - metrics["line_aic"] > thresh
    return fit_failed | chisqr | aic, message


def sinusoidal(metrics, thresh=10, message="FA: sinusoidal"):
    return (metrics["sine_sig"] > thresh) & (metrics["per"] < 10), message


def unphysical_duration(metrics, message="FA: unphysical transit duration"):
    low_aRs = (metrics["transit_aRs"] < 1.5) | (metrics["aRs"] < 2)
    long_duration = (
        (metrics["q"] / metrics["trap_qtran"] < 0.5)
        | np.isnan(metrics["sig_sec"])
        | (metrics["trap_qtran"] > 0.5)
    )
    return (low_aRs | long_duration), message


def asymmetric(metrics, thresh=5, message="FA: asymmetric events"):
    diff = abs(metrics["trap_qtran_left"] - metrics["trap_qtran_right"])
    err = np.sqrt(
        metrics["trap_qtran_err_left"] ** 2 + metrics["trap_qtran_err_right"] ** 2
    )
    return diff / err > thresh, message


def odd_even(metrics, thresh=4, message="FP: odd-even depth differences"):
    odd_even = metrics["trap_sig_oe"] > thresh
    wrong_period = (metrics["odd_dep"] < 3 * metrics["odd_err"]) | (
        metrics["even_dep"] < 3 * metrics["even_err"]
    )
    return (odd_even & ~wrong_period), message


def vshaped(metrics, thresh=1.5, message="FP: V-shaped events"):
    return (metrics["transit_b"] + metrics["transit_RpRs"]) > thresh, message


def large(metrics, thresh=25, message="FP: radius too large"):
    return metrics["Rp"] > thresh, message


def secondary(
    metrics, thresh1=1, thresh2=0, thresh3=0, message="significant secondary"
):
    MS1 = ((metrics["sig_sec"] / metrics["Fred"] - metrics["FA1"]) > thresh1) | (
        metrics["Fred"] > 2
    )
    MS2 = (
        (metrics["sig_sec"] - metrics["sig_ter"] - metrics["FA2"]) > thresh2
    ) | np.isnan(metrics["sig_ter"])
    MS3 = (
        (metrics["sig_sec"] - metrics["sig_pos"] - metrics["FA2"]) > thresh3
    ) | np.isnan(metrics["sig_pos"])
    planetary = ((metrics["sig_pri"] - metrics["sig_sec"]) < metrics["FA2"]) & (
        np.abs(metrics["phs_sec"] - 0.5) < 0.25 * metrics["qtran"]
    )
    albedo = (
        (metrics["Rp"] < 25)
        & (metrics["albedo"] < 1)
        & (metrics["dep_sec"] < 0.1 * metrics["dep"])
        & (metrics["transit_b"] < 0.95)
    )
    return (MS1 & MS2 & MS3) & ~(planetary | albedo), message


def offset(metrics, thresh=0.4, message="FP: off-target"):
    return metrics["offset"] > 0.4, message


def check_thresholds(metrics, case, verbose=False):
    if case == "FA":
        tests = [
            weak,
            invalid_transits,
            noisy,
            bad_shape,
            non_unique,
            chases,
            dmm,
            single_event,
            bad_fit,
            sinusoidal,
            unphysical_duration,
            asymmetric,
        ]
    elif case == "FP":
        tests = [
            odd_even,
            vshaped,
            large,
            secondary,
        ]
        if "offset" in metrics:
            tests.append(offset)
    else:
        raise ValueError("Case must be FA or FP")
    if type(metrics) is dict:
        mask = False
    else:
        mask = np.zeros(len(metrics)).astype(bool)
    for test in tests:
        flag, message = test(metrics)
        mask |= flag
        if type(metrics) is dict and verbose and flag:
            print(message)
    if type(metrics) is dict:
        if verbose and not mask:
            print(f"Passed all {case} tests")
    return mask

