import numpy as np

from lmfit import minimize
from leo_vetter.models import TrapezoidModel, TransitModel
from leo_vetter.utils import phasefold, weighted_mean


def diff_significance(val1, val2, err1, err2):
    diff = np.abs(val1 - val2)
    sigma = np.sqrt(err1**2 + err2**2)
    return diff / sigma


def box(tlc):
    if np.any(tlc.odd_tran):
        n_in = np.sum(tlc.odd_tran)
        N_transit = len(np.unique(tlc.epochs[tlc.odd_tran]))
        odd_dep = tlc.zpt - weighted_mean(
            tlc.flux[tlc.odd_tran], tlc.flux_err[tlc.odd_tran]
        )
        odd_dep_err = np.sqrt((tlc.sig_w**2 / n_in) + (tlc.sig_r**2 / N_transit))
    else:
        odd_dep, odd_dep_err = np.nan, np.nan
    if np.any(tlc.even_tran):
        n_in = np.sum(tlc.even_tran)
        N_transit = len(np.unique(tlc.epochs[tlc.even_tran]))
        even_dep = tlc.zpt - weighted_mean(
            tlc.flux[tlc.even_tran], tlc.flux_err[tlc.even_tran]
        )
        even_dep_err = np.sqrt((tlc.sig_w**2 / n_in) + (tlc.sig_r**2 / N_transit))
    else:
        even_dep, even_dep_err = np.nan, np.nan
    tlc.metrics["odd_dep"] = odd_dep
    tlc.metrics["odd_dep_err"] = odd_dep_err
    tlc.metrics["even_dep"] = even_dep
    tlc.metrics["even_dep_err"] = even_dep_err
    tlc.metrics["sig_dep"] = diff_significance(
        odd_dep, even_dep, odd_dep_err, even_dep_err
    )


def trapezoid(tlc):
    try:
        if np.isnan(tlc.metrics["trap_aic"]):
            raise ValueError
        tm = TrapezoidModel(
            tlc.metrics["trap_per"],
            tlc.metrics["trap_epo"],
            tlc.metrics["trap_dep"],
            tlc.metrics["trap_qtran"],
            tlc.metrics["trap_qin"],
            tlc.metrics["trap_zpt"],
        )
        for param in ["per", "qtran", "qin", "zpt"]:
            tm.params[param].vary = False
        phase = np.mod(tlc.time - tlc.epo, 2 * tlc.per) / tlc.per
        phase[phase > 1] -= 2
        odd_dep, odd_dep_err = np.nan, np.nan
        odd_epo, odd_epo_err = np.nan, np.nan
        even_dep, even_dep_err = np.nan, np.nan
        even_epo, even_epo_err = np.nan, np.nan
        if np.any(tlc.odd_tran):
            fit_tran = abs(phase) < 2 * tlc.qtran
            odd_fit = minimize(
                tm.residual,
                tm.params,
                args=(tlc.time[fit_tran], tlc.flux[fit_tran], tlc.flux_err[fit_tran]),
            )
            if odd_fit.success:
                odd_dep = odd_fit.params["dep"].value
                odd_dep_err = odd_fit.params["dep"].stderr
                odd_epo = odd_fit.params["epo"].value
                odd_epo_err = odd_fit.params["epo"].stderr
                if odd_dep_err is None:
                    odd_dep_err = np.nan
                if odd_epo_err is None:
                    odd_epo_err = np.nan
        if np.any(tlc.even_tran):
            fit_tran = abs(phase) > 1 - 2 * tlc.qtran
            even_fit = minimize(
                tm.residual,
                tm.params,
                args=(tlc.time[fit_tran], tlc.flux[fit_tran], tlc.flux_err[fit_tran]),
            )
            if even_fit.success:
                even_dep = even_fit.params["dep"].value
                even_dep_err = even_fit.params["dep"].stderr
                even_epo = even_fit.params["epo"].value
                even_epo_err = even_fit.params["epo"].stderr
                if even_dep_err is None:
                    even_dep_err = np.nan
                if even_epo_err is None:
                    even_epo_err = np.nan
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed odd/even trapezoid fit")
        tlc.metrics["trap_odd_dep"] = np.nan
        tlc.metrics["trap_odd_dep_err"] = np.nan
        tlc.metrics["trap_even_dep"] = np.nan
        tlc.metrics["trap_even_dep_err"] = np.nan
        tlc.metrics["trap_odd_epo"] = np.nan
        tlc.metrics["trap_odd_epo_err"] = np.nan
        tlc.metrics["trap_even_epo"] = np.nan
        tlc.metrics["trap_even_epo_err"] = np.nan
        tlc.metrics["trap_sig_dep"] = np.nan
        tlc.metrics["trap_sig_epo"] = np.nan
        return
    tlc.metrics["trap_odd_dep"] = odd_dep
    tlc.metrics["trap_odd_dep_err"] = odd_dep_err
    tlc.metrics["trap_even_dep"] = even_dep
    tlc.metrics["trap_even_dep_err"] = even_dep_err
    tlc.metrics["trap_odd_epo"] = odd_epo
    tlc.metrics["trap_odd_epo_err"] = odd_epo_err
    tlc.metrics["trap_even_epo"] = even_epo
    tlc.metrics["trap_even_epo_err"] = even_epo_err
    tlc.metrics["trap_sig_dep"] = diff_significance(
        odd_dep, even_dep, odd_dep_err, even_dep_err
    )
    tlc.metrics["trap_sig_epo"] = diff_significance(
        odd_epo, even_epo, odd_epo_err, even_epo_err
    )


def transit(tlc, cap_b=True):
    try:
        if np.isnan(tlc.metrics["transit_aic"]):
            raise ValueError
        tm = TransitModel(
            tlc.metrics["transit_per"],
            tlc.metrics["transit_epo"],
            tlc.metrics["transit_RpRs"],
            tlc.metrics["transit_aRs"],
            tlc.metrics["transit_b"],
            tlc.metrics["transit_u1"],
            tlc.metrics["transit_u2"],
            tlc.metrics["transit_zpt"],
            cap_b=cap_b,
        )
        for param in ["per", "b", "aRs", "zpt"]:
            tm.params[param].vary = False
        phase = np.mod((tlc.time - tlc.epo) / tlc.per, 2)
        phase[phase > 1] -= 2
        odd_RpRs, odd_RpRs_err = np.nan, np.nan
        odd_epo, odd_epo_err = np.nan, np.nan
        even_RpRs, even_RpRs_err = np.nan, np.nan
        even_epo, even_epo_err = np.nan, np.nan
        if np.any(tlc.odd_tran):
            fit_tran = abs(phase) < 2 * tlc.qtran
            odd_fit = minimize(
                tm.residual,
                tm.params,
                args=(tlc.time[fit_tran], tlc.flux[fit_tran], tlc.flux_err[fit_tran]),
            )
            if odd_fit.success:
                odd_RpRs = odd_fit.params["RpRs"].value
                odd_RpRs_err = odd_fit.params["RpRs"].stderr
                odd_epo = odd_fit.params["epo"].value
                odd_epo_err = odd_fit.params["epo"].stderr
                if odd_RpRs_err is None:
                    odd_RpRs_err = np.nan
                if odd_epo_err is None:
                    odd_epo_err = np.nan
        if np.any(tlc.even_tran):
            fit_tran = abs(phase) > 1 - 2 * tlc.qtran
            even_fit = minimize(
                tm.residual,
                tm.params,
                args=(tlc.time[fit_tran], tlc.flux[fit_tran], tlc.flux_err[fit_tran]),
            )
            if even_fit.success:
                even_RpRs = even_fit.params["RpRs"].value
                even_RpRs_err = even_fit.params["RpRs"].stderr
                even_epo = even_fit.params["epo"].value
                even_epo_err = even_fit.params["epo"].stderr
                if even_RpRs_err is None:
                    even_RpRs_err = np.nan
                if even_epo_err is None:
                    even_epo_err = np.nan
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed odd/even transit fit")
        tlc.metrics["transit_odd_RpRs"] = np.nan
        tlc.metrics["transit_odd_RpRs_err"] = np.nan
        tlc.metrics["transit_even_RpRs"] = np.nan
        tlc.metrics["transit_even_RpRs_err"] = np.nan
        tlc.metrics["transit_odd_epo"] = np.nan
        tlc.metrics["transit_odd_epo_err"] = np.nan
        tlc.metrics["transit_even_epo"] = np.nan
        tlc.metrics["transit_even_epo_err"] = np.nan
        tlc.metrics["transit_sig_dep"] = np.nan
        tlc.metrics["transit_sig_epo"] = np.nan
        return
    tlc.metrics["transit_odd_RpRs"] = odd_RpRs
    tlc.metrics["transit_odd_RpRs_err"] = odd_RpRs_err
    tlc.metrics["transit_even_RpRs"] = even_RpRs
    tlc.metrics["transit_even_RpRs_err"] = even_RpRs_err
    tlc.metrics["transit_odd_epo"] = odd_epo
    tlc.metrics["transit_odd_epo_err"] = odd_epo_err
    tlc.metrics["transit_even_epo"] = even_epo
    tlc.metrics["transit_even_epo_err"] = even_epo_err
    tlc.metrics["transit_sig_dep"] = diff_significance(
        odd_RpRs, even_RpRs, odd_RpRs_err, even_RpRs_err
    )
    tlc.metrics["transit_sig_epo"] = diff_significance(
        odd_epo, even_epo, odd_epo_err, even_epo_err
    )
