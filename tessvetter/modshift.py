import numpy as np

from scipy.special import erfcinv
from tessvetter.utils import phasefold, weighted_std


def phase_and_significance(phase, MES_series):
    arg = np.argmax(MES_series)
    phs = phase[arg]
    sig = MES_series[arg]
    return arg, phs, sig


def uniqueness(tlc, nTCE=20000):
    phs_pri, sig_pri = np.nan, np.nan
    phs_sec, sig_sec = np.nan, np.nan
    phs_ter, sig_ter = np.nan, np.nan
    phs_pos, sig_pos = np.nan, np.nan
    dep_sec, err_sec = np.nan, np.nan
    Fred = np.nan
    # False alarm thresholds
    tlc.metrics["FA1"] = np.sqrt(2) * erfcinv((tlc.dur / tlc.per) * (1.0 / nTCE))
    tlc.metrics["FA2"] = np.sqrt(2) * erfcinv((tlc.dur / tlc.per))
    # Re-phase light curve to make it easier to work with
    phase = phasefold(tlc.time, tlc.per, tlc.epo)
    phase[phase < 0] += 1
    while True:
        # Primary significance
        _, phs_pri, sig_pri = phase_and_significance(
            phase[tlc.in_tran], tlc.MES_series[tlc.in_tran]
        )
        # Secondary significance - at least 2 transit durations from primary
        mask = (abs(phase - phs_pri) < 2 * tlc.qtran) | (
            abs(phase - phs_pri) > 1 - 2 * tlc.qtran
        )
        if not np.any(~mask):
            break
        arg_sec, phs_sec, sig_sec = phase_and_significance(
            phase[~mask], tlc.MES_series[~mask]
        )
        dep_sec = tlc.dep_series[~mask][arg_sec]
        err_sec = tlc.err_series[~mask][arg_sec]
        # Estimate Fred from data outside of primary and secondary
        non_pri_sec = ~mask & ~(abs(phase - phs_sec) < tlc.qtran)
        if not np.any(non_pri_sec):
            break
        # Red noise is std of measured amplitudes
        red_noise = weighted_std(
            tlc.dep_series[non_pri_sec], tlc.err_series[non_pri_sec]
        )
        # White noise is std of photometric datapoints
        white_noise = weighted_std(tlc.flux[non_pri_sec], tlc.flux_err[non_pri_sec])
        Fred = np.sqrt(tlc.n_in) * red_noise / white_noise
        # Tertiary significance - at least 2 transit durations from primary and secondary
        mask = mask | (abs(phase - phs_sec) < 2 * tlc.qtran)
        if not np.any(~mask):
            break
        _, phs_ter, sig_ter = phase_and_significance(
            phase[~mask], tlc.MES_series[~mask]
        )
        # Positive significance - at least 3 transit durations from primary and secondary
        mask = (
            (abs(phase - phs_pri) < 3 * tlc.qtran)
            | (abs(phase - phs_pri) > 1 - 3 * tlc.qtran)
            | (abs(phase - phs_sec) < 3 * tlc.qtran)
        )
        if not np.any(~mask):
            break
        _, phs_pos, sig_pos = phase_and_significance(
            phase[~mask], -tlc.MES_series[~mask]
        )
        break
    tlc.metrics["phs_pri"] = phs_pri
    tlc.metrics["phs_sec"] = phs_sec
    tlc.metrics["phs_ter"] = phs_ter
    tlc.metrics["phs_pos"] = phs_pos
    tlc.metrics["sig_pri"] = sig_pri
    tlc.metrics["sig_sec"] = sig_sec
    tlc.metrics["sig_ter"] = sig_ter
    tlc.metrics["sig_pos"] = sig_pos
    tlc.metrics["dep_sec"] = dep_sec
    tlc.metrics["err_sec"] = err_sec
    tlc.metrics["Fred"] = Fred
