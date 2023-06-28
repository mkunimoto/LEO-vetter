import numpy as np

from tessvetter.utils import phasefold, weighted_mean
from tessvetter.models import TransitModel


def transit_events(tlc, frac=0.7):
    if "transit_aic" not in tlc.metrics:
        print(f"{tlc.tic}.{tlc.planetno}: warning: no transit model provided")
    if "sig_r" not in tlc.metrics:
        print(
            f"{tlc.tic}.{tlc.planetno}: missing SES and MES time series. Getting now..."
        )
        tlc.get_SES_MES()
    deps = np.zeros(tlc.N_transit)
    errs = np.zeros(tlc.N_transit)
    tlc.SES = np.zeros(tlc.N_transit)
    tlc.rubble = np.zeros(tlc.N_transit)
    tlc.chases = np.zeros(tlc.N_transit)
    tlc.redchi2 = np.zeros(tlc.N_transit)
    # Search range for chases metric is between 1.5 durations and 0.1 times the period away
    chases_tran = (abs(tlc.phase) > 1.5 * tlc.qtran) & (abs(tlc.phase) < 0.1)
    # Get metrics for each transit event
    for i in range(tlc.N_transit):
        epoch = tlc.tran_epochs[i]
        in_epoch = tlc.in_tran & (tlc.epochs == epoch)
        # Compute the transit time, depth, and SES for this transit
        transit_time = tlc.epo + tlc.per * epoch
        n_in = np.sum(in_epoch)
        dep = tlc.zpt - weighted_mean(tlc.flux[in_epoch], tlc.flux_err[in_epoch])
        err = np.sqrt((tlc.sig_w**2 / n_in) + tlc.sig_r**2)
        deps[i], errs[i] = dep, err
        tlc.SES[i] = dep / err
        # Find the most significant nearby event
        chases_epoch = (
            chases_tran
            & (tlc.epochs == epoch)
            & (np.abs(tlc.SES_series) > frac * tlc.SES[i])
        )
        if np.any(chases_epoch):
            tlc.chases[i] = np.min(np.abs(tlc.time[chases_epoch] - transit_time)) / (
                0.1 * tlc.per
            )
        else:
            tlc.chases[i] = 1
        # Find how much of the transit falls in gaps
        fit_epoch = tlc.fit_tran & (tlc.epochs == epoch)
        n_obs = np.sum(fit_epoch)
        cadence = np.nanmedian(np.diff(tlc.time[fit_epoch]))
        n_exp = 4 * tlc.dur / cadence
        tlc.rubble[i] = n_obs / n_exp
        if ("transit_aic" in tlc.metrics) and ~np.isnan(tlc.metrics["transit_aic"]):
            tm = TransitModel(
                tlc.metrics["transit_per"],
                tlc.metrics["transit_epo"],
                tlc.metrics["transit_RpRs"],
                tlc.metrics["transit_aRs"],
                tlc.metrics["transit_b"],
                tlc.metrics["transit_u1"],
                tlc.metrics["transit_u2"],
                tlc.metrics["transit_zpt"],
            )
            resid = tm.residual(
                tm.params,
                tlc.time[fit_epoch],
                tlc.flux[fit_epoch],
                tlc.flux_err[fit_epoch],
            )
            chi2 = np.sum(resid**2)
            tlc.redchi2[i] = chi2 / (np.sum(fit_epoch) - 6)
        else:
            tlc.redchi2[i] = np.nan
    O = tlc.SES
    E = tlc.dep / errs
    chi2 = np.sum((O - E) ** 2 / E)
    tlc.metrics["CHI"] = tlc.metrics["MES"] / np.sqrt(chi2 / (tlc.N_transit - 1))
    tlc.metrics["med_chases"] = np.nanmedian(tlc.chases)
    tlc.metrics["mean_chases"] = np.nanmean(tlc.chases)
    tlc.metrics["max_SES"] = np.nanmax(tlc.SES)
    tlc.metrics["DMM"] = np.nanmean(deps) / np.nanmedian(deps)


def recompute_MES(tlc, chases=0.01, rubble=0.75):
    if "med_chases" not in tlc.metrics:
        print(
            f"{tlc.tic}.{tlc.planetno}: missing individual transit metrics not. Getting now..."
        )
        transit_events(tlc)
    rubble_flag = tlc.rubble <= rubble
    zuma_flag = tlc.SES < 0
    redchi2_flag = tlc.redchi2 > 5
    bad_epochs = rubble_flag | zuma_flag | redchi2_flag
    if tlc.N_transit <= 5:
        chases_flag = tlc.chases < chases
        bad_epochs |= chases_flag
    use_tran = tlc.in_tran & ~np.isin(tlc.epochs, tlc.tran_epochs[bad_epochs])
    if np.any(use_tran):
        new_N_transit = len(tlc.tran_epochs[~bad_epochs])
        new_n_in = np.sum(use_tran)
        dep = tlc.zpt - weighted_mean(tlc.flux[use_tran], tlc.flux_err[use_tran])
        err = np.sqrt((tlc.sig_w**2 / new_n_in) + (tlc.sig_r**2 / new_N_transit))
        new_MES = dep / err
    else:
        new_N_transit = 0
        new_MES = 0
    tlc.metrics["new_N_transit"] = new_N_transit
    tlc.metrics["new_MES"] = new_MES
