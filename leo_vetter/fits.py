import numpy as np

from lmfit import minimize
from leo_vetter.models import LinearModel, TrapezoidModel, TransitModel, SineModel
from leo_vetter.utils import phasefold


def linear(tlc):
    try:
        lm = LinearModel(tlc.zpt, 0)
        fit = minimize(
            lm.residual,
            lm.params,
            args=(
                tlc.phase[tlc.fit_tran] * tlc.per,
                tlc.flux[tlc.fit_tran],
                tlc.flux_err[tlc.fit_tran],
            ),
        )
        if not fit.success:
            raise ValueError
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed linear model fit")
        tlc.metrics["line_chisqr"] = np.nan
        tlc.metrics["line_aic"] = np.nan
        return
    tlc.metrics["line_chisqr"] = fit.redchi
    tlc.metrics["line_aic"] = fit.aic


def trapezoid(tlc):
    try:
        if tlc.dep < 0:
            raise ValueError
        best_chisqr = np.inf
        for qin in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            tm = TrapezoidModel(tlc.per, tlc.epo, tlc.dep, tlc.qtran, qin, tlc.zpt)
            tm.params["qin"].vary = False
            fit = minimize(
                tm.residual,
                tm.params,
                args=(
                    tlc.time[tlc.fit_tran],
                    tlc.flux[tlc.fit_tran],
                    tlc.flux_err[tlc.fit_tran],
                ),
            )
            chisqr = fit.redchi
            if (chisqr < best_chisqr) and fit.success:
                best_fit = fit
                best_chisqr = chisqr
        tm = TrapezoidModel(params=best_fit.params)
        tm.params["qin"].vary = True
        full_fit = minimize(
            tm.residual,
            tm.params,
            args=(
                tlc.time[tlc.fit_tran],
                tlc.flux[tlc.fit_tran],
                tlc.flux_err[tlc.fit_tran],
            ),
        )
        if not full_fit.success:
            raise ValueError
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed trapezoid model fit")
        tlc.metrics["trap_chisqr"] = np.nan
        tlc.metrics["trap_aic"] = np.nan
        for param in ["per", "epo", "dep", "qtran", "qin", "zpt"]:
            tlc.metrics[f"trap_{param}"] = np.nan
            tlc.metrics[f"trap_{param}_err"] = np.nan
        return
    tlc.metrics["trap_chisqr"] = full_fit.redchi
    tlc.metrics["trap_aic"] = full_fit.aic
    for param in ["per", "epo", "dep", "qtran", "qin", "zpt"]:
        tlc.metrics[f"trap_{param}"] = full_fit.params[param].value
        tlc.metrics[f"trap_{param}_err"] = full_fit.params[param].stderr
        if tlc.metrics[f"trap_{param}_err"] is None:
            tlc.metrics[f"trap_{param}_err"] = np.nan


def half_trapezoid(tlc, side):
    if "trap_aic" not in tlc.metrics:
        print(f"{tlc.tic}.{tlc.planetno}: missing trapezoid fit. Getting now...")
        trapezoid(tlc)
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
        if side == "right":
            mask = (tlc.phase > 0) & (tlc.phase < 2 * tlc.qtran)
        elif side == "left":
            mask = (tlc.phase < 0) & (tlc.phase > -2 * tlc.qtran)
        for param in ["per", "epo", "dep"]:
            tm.params[param].vary = False
        fit = minimize(
            tm.residual,
            tm.params,
            args=(tlc.time[mask], tlc.flux[mask], tlc.flux_err[mask]),
        )
        if not fit.success:
            raise ValueError
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed {side} trapezoid fit")
        tlc.metrics[f"trap_qtran_{side}"] = np.nan
        tlc.metrics[f"trap_qtran_err_{side}"] = np.nan
        return
    tlc.metrics[f"trap_qtran_{side}"] = fit.params["qtran"].value
    tlc.metrics[f"trap_qtran_err_{side}"] = fit.params["qtran"].stderr
    if tlc.metrics[f"trap_qtran_err_{side}"] is None:
        tlc.metrics[f"trap_qtran_err_{side}"] = np.nan


def transit(tlc, u1, u2, cap_b=False):
    tlc.metrics["transit_u1"] = u1
    tlc.metrics["transit_u2"] = u2
    try:
        if tlc.dep < 0:
            raise ValueError
        best_chisqr = np.inf
        for b in [0.1, 0.3, 0.5, 0.7, 0.9]:
            RpRs = np.sqrt(tlc.dep)
            sinterm = np.sin(tlc.dur * np.pi / tlc.per) ** 2
            aRs = np.sqrt(((1 + RpRs) ** 2 - b**2 * (1 - sinterm)) / sinterm)
            tm = TransitModel(
                tlc.per, tlc.epo, RpRs, aRs, b, u1, u2, tlc.zpt, cap_b=cap_b
            )
            tm.params["b"].vary = False
            fit = minimize(
                tm.residual,
                tm.params,
                args=(
                    tlc.time[tlc.fit_tran],
                    tlc.flux[tlc.fit_tran],
                    tlc.flux_err[tlc.fit_tran],
                ),
            )
            chisqr = fit.redchi
            if (chisqr < best_chisqr) and fit.success:
                best_fit = fit
                best_chisqr = chisqr
        tm = TransitModel(params=best_fit.params)
        tm.params["b"].vary = True
        full_fit = minimize(
            tm.residual,
            tm.params,
            args=(
                tlc.time[tlc.fit_tran],
                tlc.flux[tlc.fit_tran],
                tlc.flux_err[tlc.fit_tran],
            ),
        )
        if not full_fit.success:
            raise ValueError
    except Exception:
        print(f"{tlc.tic}.{tlc.planetno}: failed transit model fit")
        tlc.metrics["transit_chisqr"] = np.nan
        tlc.metrics["transit_aic"] = np.nan
        for param in ["per", "epo", "RpRs", "aRs", "b", "zpt"]:
            tlc.metrics[f"transit_{param}"] = np.nan
            tlc.metrics[f"transit_{param}_err"] = np.nan
        return
    tlc.metrics["transit_chisqr"] = full_fit.redchi
    tlc.metrics["transit_aic"] = full_fit.aic
    for param in ["per", "epo", "RpRs", "aRs", "b", "zpt"]:
        tlc.metrics[f"transit_{param}"] = full_fit.params[param].value
        tlc.metrics[f"transit_{param}_err"] = full_fit.params[param].stderr
        if tlc.metrics[f"transit_{param}_err"] is None:
            tlc.metrics[f"transit_{param}_err"] = np.nan


def sweet(tlc):
    best_sig = 0
    for per in [0.5 * tlc.per, tlc.per, 2 * tlc.per]:
        phase = phasefold(tlc.time, per, tlc.epo) * per
        sm = SineModel(per, np.nanstd(tlc.flux), 3 * np.pi / 2.0, tlc.zpt)
        sm.params["per"].vary = False
        try:
            fit = minimize(
                sm.residual,
                sm.params,
                args=(
                    phase[~tlc.near_tran],
                    tlc.flux[~tlc.near_tran],
                    tlc.flux_err[~tlc.near_tran],
                ),
            )
            if fit.success:
                sig = fit.params["amp"].value / fit.params["amp"].stderr
                if sig > best_sig:
                    best_sig = sig
                    best_amp = fit.params["amp"].value
                    best_fit = fit
        except TypeError:
            continue
    if best_sig == 0:
        print(f"{tlc.tic}.{tlc.planetno}: failed SWEET test")
        tlc.metrics["sine_sig"] = np.nan
        tlc.metrics["sine_amp"] = np.nan
        return
    tlc.metrics["sine_sig"] = best_sig
    tlc.metrics["sine_amp"] = best_amp
