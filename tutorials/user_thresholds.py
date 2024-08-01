import numpy as np

_default_thresholds = {
    "MES": 6.2,
    "N_transit": 3,
    "SHP": 0.5,
    "MS1": 1,
    "MS2": 1,
    "MS3": 1,
    "chases": 0.78,
    "max_SES_to_MES": 0.8,
    "AIC1": -60,
    "AIC2": -30,
    "SWEET": 15,
    "ASYM": 8,
    "CHI": 7.8,
    "frac_gap": 0.5,
    "V_shape": 1.5,
    "size": 22,
    "MS4": 0,
    "MS5": -1,
    "MS6": -1,
    "offset": 15,
}

#### False alarm checks ####

def weak(metrics, thresholds):
    # Checks the Multiple Event Statistic (MES, i.e. signal-to-noise ratio)
    message = "FA: signal too weak"
    return (metrics["MES"] < thresholds["MES"]), message


def invalid_transits(metrics, thresholds):
    # Checks the number of "good" transits left after removing all the "bad" transits
    message = "FA: not enough valid transits"
    mes = (metrics["new_MES"] < thresholds["MES"])
    ntr = (metrics["new_N_transit"] < thresholds["N_transit"])
    return mes | ntr, message


def bad_shape(metrics, thresholds):
    # Checks whether there are flux increases during the event
    # SHP = 0.0 means the event only decreases the flux
    # SHP = 0.5 means the event has both increases and decreases in flux
    # SHP = 1.0 means the event only increases the flux
    message = "FA: bad transit shape"
    return (metrics["SHP"] > thresholds["SHP"]), message


def non_unique(metrics, thresholds):
    # Checks the uniqueness of the event compared to noise/other events
    message = "FA: events not unique in phased light curve"
    MS1 = (metrics["sig_pri"] / metrics["Fred"] - metrics["FA1"]) < thresholds["MS1"]
    MS2 = (metrics["sig_pri"] - metrics["sig_ter"] - metrics["FA2"]) < thresholds["MS2"]
    MS3 = (metrics["sig_pri"] - metrics["sig_pos"] - metrics["FA2"]) < thresholds["MS3"]
    return MS1 | MS2 | MS3, message


def chases(metrics, thresholds):
    # Checks the uniqueness of each individual event compared to the local light curve
    message = "FA: events not unique in local light curve"
    return (metrics["N_transit"] <= 5) & (metrics["mean_chases"] < thresholds["chases"]), message


def dmm(metrics, thresholds):
    # Checks for consistency between the mean and median transit depths
    message = "FA: inconsistent transit depths"
    return (metrics["DMM"] < 0.5) | (metrics["DMM"] > 1.5), message


def single_event(metrics, thresholds):
    # Checks for single events dominating the MES
    message = "FA: dominated by single event"
    return (metrics["max_SES"] / metrics["MES"] > thresholds["max_SES_to_MES"] ) & (
        metrics["N_transit"] <= 10
    ), message


def bad_fit(metrics, thresholds):
    # Checks that a transit model fit is preferred over a straight line
    message = "FA: bad transit model fit"
    fit_failed = np.isnan(metrics["transit_aic"])
    chisqr = metrics["transit_chisqr"] > metrics["line_chisqr"]
    gen_metrics = (metrics["transit_aic"] - metrics["line_aic"]) > thresholds["AIC1"]
    sub_metrics = (metrics["transit_aic"] - metrics["line_aic"]) > thresholds["AIC2"]
    aic = (gen_metrics & (metrics["N_transit"] <= 10)) | (sub_metrics & (metrics["N_transit"] > 10))
    return fit_failed | chisqr | aic, message


def sinusoidal(metrics, thresholds):
    # Checks for sinusoidal variations in the phased light curve
    message = "FA: sinusoidal variations"
    return (metrics["sine_sig"] > thresholds["SWEET"]) & (metrics["per"] < 10), message


def unphysical_duration(metrics, thresholds):
    # Checks that the orbit is physically realistic
    message = "FA: unphysical transit orbit"
    low_aRs = (metrics["transit_aRs"] < 1.5) | (metrics["aRs"] < 2)
    long_duration = (
        (metrics["q"] / metrics["trap_qtran"] < 0.6)
        | np.isnan(metrics["sig_sec"])
        | (metrics["trap_qtran"] > 0.5)
    )
    return (low_aRs | long_duration), message


def asymmetric(metrics, thresholds):
    # Checks for transit asymmetry
    message = "FA: asymmetric events"
    diff = abs(metrics["trap_qtran_left"] - metrics["trap_qtran_right"])
    err = np.sqrt(
        metrics["trap_qtran_err_left"] ** 2 + metrics["trap_qtran_err_right"] ** 2
    )
    return (diff / err) > thresholds["ASYM"], message


def chi(metrics, thresholds):
    # Checks consistency between transit SNRs
    message = "FA: inconsistent transit SNRs"
    return (metrics["CHI"] < thresholds["CHI"]), message


def data_gapped(metrics, thresholds):
    # Checks the number of "good" transits left after removing all the "bad" transits
    message = "FA: too many transits near gaps"
    return (metrics["N_gap_2.0"]/metrics["N_transit"] >= thresholds["frac_gap"]), message


#### Astrophysical false positive checks ####


def odd_even(metrics, thresholds):
    # Checks for differences between odd and even transits
    message = "FP: odd-even transit differences"
    box_dep_sigma = (metrics["sig_dep"] > 3)
    trap_dep_sigma = (metrics["trap_sig_dep"] > 3)
    trap_epo_sigma = (metrics["trap_sig_epo"] > 10)
    transit_dep_sigma = (metrics["transit_sig_dep"] > 3)
    transit_epo_sigma = (metrics["transit_sig_epo"] > 10)
    return ((box_dep_sigma & (trap_dep_sigma | transit_dep_sigma)) | (transit_epo_sigma | trap_epo_sigma)), message


def vshaped(metrics, thresholds):
    message = "FP: V-shaped events"
    # Checks for both large and highly grazing objects
    return (metrics["transit_b"] + metrics["transit_RpRs"]) > thresholds["V_shape"], message


def large(metrics, thresholds):
    # Checks the radius of the object
    message = "FP: radius too large"
    return (metrics["Rp"] > thresholds["size"]), message


def secondary(metrics, thresholds):
    # Checks for the existence of a significant secondary eclipse
    message = "FP: significant secondary"
    MS4 = ((metrics["sig_sec"] / metrics["Fred"] - metrics["FA1"]) > thresholds["MS4"]) | (
        metrics["Fred"] > 1.8
    )
    MS5 = ((metrics["sig_sec"] - metrics["sig_ter"] - metrics["FA2"]) > thresholds["MS5"]) | np.isnan(
        metrics["sig_ter"]
    )
    MS6 = ((metrics["sig_sec"] - metrics["sig_pos"] - metrics["FA2"]) > thresholds["MS6"]) | np.isnan(
        metrics["sig_pos"]
    )
    # The secondary may be consistent with a planet
    albedo = (
        (metrics["Rp"] < thresholds["size"])
        & (metrics["albedo"] < 1)
        & (metrics["dep_sec"] < 0.1 * metrics["dep"])
        & (metrics["transit_b"] < 0.95)
    )
    return (MS4 & (MS5 | MS6)) & ~albedo, message


def offset(metrics, thresholds):
    # Checks for offset between PRF fit and star location
    message = "FP: off-target"
    return metrics["offset_qual"] > thresholds["offset"], message


#### Check metrics against pass-fail thresholds ####

def check_thresholds(metrics, case, verbose=False, thresholds=_default_thresholds):
    if case == "FA":
        tests = [
            weak,
            invalid_transits,
            bad_shape,
            non_unique,
            chases,
            dmm,
            single_event,
            bad_fit,
            sinusoidal,
            unphysical_duration,
            asymmetric,
            chi,
            data_gapped,
        ]
    elif case == "FP":
        tests = [
            odd_even,
            vshaped,
            large,
            secondary,
        ]
        if "offset_qual" in metrics:
            tests.append(offset)
    else:
        raise ValueError("Case must be FA or FP")
    if type(metrics) is dict:
        mask = False
    else:
        mask = np.zeros(len(metrics)).astype(bool)
    for test in tests:
        flag, message = test(metrics, thresholds)
        mask |= flag
        if type(metrics) is dict and verbose and flag:
            print(message)
    if type(metrics) is dict:
        if verbose and not mask:
            print(f"Passed all {case} tests")
    return mask
