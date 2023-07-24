import numpy as np


def weak(metrics):
    # Checks the Multiple Event Statistic (MES, i.e. signal-to-noise ratio)
    message = "FA: signal too weak"
    return (metrics["MES"] < 9) | (metrics["new_MES"] < 9), message


def invalid_transits(metrics):
    # Checks the number of "good" transits left after removing all the "bad" transits
    message="FA: not enough valid transits"
    return (metrics["new_N_transit"] < 2), message


def noisy(metrics):
    # Checks the amount of systematic red noise in the light curve
    message="FA: too much noise at transit timescale"
    return metrics["Fred"] > 4, message


def bad_shape(metrics):
    # Checks whether there are flux increases during the event
    # SHP = 0.0 means the event only decreases the flux
    # SHP = 0.5 means the event has both increases and decreases in flux
    # SHP = 1.0 means the event only increases the flux
    message="FA: bad transit shape"
    return (metrics["SHP"] > 0.4) & (metrics["per"] < 10), message


def non_unique(metrics):
    # Checks the uniqueness of the event compared to noise/other events
    message="FA: events not unique in phased light curve"
    MS1 = (metrics["sig_pri"] / metrics["Fred"] - metrics["FA1"]) < 3
    MS2 = (metrics["sig_pri"] - metrics["sig_ter"] - metrics["FA2"]) < 1
    MS3 = (metrics["sig_pri"] - metrics["sig_pos"] - metrics["FA2"]) < 0
    return MS1 | MS2 | MS3, message


def chases(metrics):
    # Checks the uniqueness of each individual event compared to the local light curve
    message="FA: events not unique in local light curve"
    return (metrics["N_transit"] <= 5) & (metrics["med_chases"] < 0.8), message


def dmm(metrics):
    # Checks for consistency between the mean and median transit depths
    message="FA: inconsistent transit depths"
    return metrics["DMM"] > 1.5, message


def single_event(metrics):
    # Checks for single events dominating the MES
    message="FA: dominated by single event"
    return (metrics["max_SES"] / metrics["MES"] > 0.8) & (
        metrics["N_transit"] <= 5
    ), message


def bad_fit(metrics):
    # Checks that a transit model fit is preferred over a straight line
    message="FA: bad transit model fit"
    fit_failed = np.isnan(metrics["transit_aic"])
    chisqr = metrics["transit_chisqr"] > metrics["line_chisqr"]
    aic = metrics["transit_aic"] - metrics["line_aic"] > -20
    return fit_failed | chisqr | aic, message


def sinusoidal(metrics):
    # Checks for sinusoidal variations in the phased light curve
    message="FA: sinusoidal variations"
    return (metrics["sine_sig"] > 10) & (metrics["per"] < 10), message


def unphysical_duration(metrics):
    # Checks that the orbit is physically realistic
    message="FA: unphysical transit orbit"
    low_aRs = (metrics["transit_aRs"] < 1.5) | (metrics["aRs"] < 2)
    long_duration = (
        (metrics["q"] / metrics["trap_qtran"] < 0.5)
        | np.isnan(metrics["sig_sec"])
        | (metrics["trap_qtran"] > 0.5)
    )
    return (low_aRs | long_duration), message


def asymmetric(metrics):
    # Checks for transit asymmetry
    message="FA: asymmetric events"
    diff = abs(metrics["trap_qtran_left"] - metrics["trap_qtran_right"])
    err = np.sqrt(
        metrics["trap_qtran_err_left"] ** 2 + metrics["trap_qtran_err_right"] ** 2
    )
    return (diff / err) > 5, message


def odd_even(metrics):
    # Checks for differences between odd and even transit depths
    message="FP: odd-even depth differences"
    odd_even = metrics["trap_sig_oe"] > 4
    # The planet may have no odd or even transits
    wrong_period = (metrics["odd_dep"] < 3 * metrics["odd_err"]) | (
        metrics["even_dep"] < 3 * metrics["even_err"]
    )
    return (odd_even & ~wrong_period), message


def vshaped(metrics):
    message="FP: V-shaped events"
    # Checks for both large and highly grazing objects
    return (metrics["transit_b"] + metrics["transit_RpRs"]) > 1.5, message


def large(metrics):
    # Checks the radius of the object
    message="FP: radius too large"
    return metrics["Rp"] > 25, message


def secondary(metrics):
    # Checks for the existence of a significant secondary eclipse
    message="FP: significant secondary"
    MS1 = ((metrics["sig_sec"] / metrics["Fred"] - metrics["FA1"]) > 1) | (
        metrics["Fred"] > 2
    )
    MS2 = ((metrics["sig_sec"] - metrics["sig_ter"] - metrics["FA2"]) > 0) | np.isnan(
        metrics["sig_ter"]
    )
    MS3 = ((metrics["sig_sec"] - metrics["sig_pos"] - metrics["FA2"]) > 0) | np.isnan(
        metrics["sig_pos"]
    )
    # The planet may have been found at the wrong period
    planetary = ((metrics["sig_pri"] - metrics["sig_sec"]) < metrics["FA2"]) & (
        np.abs(metrics["phs_sec"] - 0.5) < 0.25 * metrics["qtran"]
    )
    # The secondary may be consistent with a planet
    albedo = (
        (metrics["Rp"] < 25)
        & (metrics["albedo"] < 1)
        & (metrics["dep_sec"] < 0.1 * metrics["dep"])
        & (metrics["transit_b"] < 0.95)
    )
    return (MS1 & MS2 & MS3) & ~(planetary | albedo), message


def offset(metrics):
    # Checks for offset between PRF fit and star location
    message="FP: off-target"
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
