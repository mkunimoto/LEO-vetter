import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
import os

from brokenaxes import brokenaxes
from scipy.stats import binned_statistic

from leo_vetter.utils import MAD
from leo_vetter.models import TransitModel, TrapezoidModel

_label_colour = "C0"
_data_colour = "0.7"
_bin_colour = "k"
_odd_colour = "C1"
_even_colour = "C2"


def binned_data(time, flux, nbins):
    bin_cent, _, _ = binned_statistic(time, time, statistic="mean", bins=nbins)
    bin_mean, _, _ = binned_statistic(time, flux, statistic="mean", bins=nbins)
    bin_std, _, _ = binned_statistic(time, flux, statistic="std", bins=nbins)
    bin_count, _, _ = binned_statistic(time, flux, statistic="count", bins=nbins)
    bin_err = bin_std / np.sqrt(bin_count)
    return bin_cent, bin_mean, bin_err


def expanded_phase(phase, flux, deps):
    phase1 = np.copy(phase)
    phase1[phase1 < -0.25] += 1
    phase2 = phase[(phase > -0.25) & (phase < 0.25)]
    flux2 = flux[(phase > -0.25) & (phase < 0.25)]
    deps2 = deps[(phase > -0.25) & (phase < 0.25)]
    phase2 += 1
    phase3 = np.concatenate((phase1, phase2))
    flux3 = (np.concatenate((flux, flux2)) - 1) * 1e6
    deps3 = np.concatenate((deps, deps2)) * 1e6
    return (
        phase3[np.argsort(phase3)],
        flux3[np.argsort(phase3)],
        deps3[np.argsort(phase3)],
    )


def modshift_box(label, ax, x, y, dy, phs, qtran, xtext=0.5, ytext=0.9, fs=7):
    cent = phs - 1 if phs > 0.75 else phs
    mask = (x > cent - qtran * 1.5) & (x < cent + qtran * 1.5)
    ax.errorbar(x[mask], y[mask], yerr=dy[mask], fmt="o", color="b", ms=3, capsize=2)
    ax.set_xlim([cent - qtran * 1.5, cent + qtran * 1.5])
    ax.axhline(y=0, ls="--", alpha=0.5, color="k")
    ax.text(
        xtext,
        ytext,
        label,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=fs,
    )
    ax.tick_params(axis="both", which="both", labelsize=fs)


def modshift_oddeven(label, ax, phase, flux, deps, phs, qtran, xtext=0.5, ytext=0.9):
    phase2, flux2, deps2 = expanded_phase(phase, flux, deps)
    bin_cent, bin_mean, bin_err = binned_data(phase2, flux2, int(10 * 1.5 / qtran))
    modshift_box(label, ax, bin_cent, bin_mean, bin_err, phs, qtran, xtext, ytext)


def modshift_text(tlc, ax, fs=7):
    labels = [
        "Pri",
        "Sec",
        "Ter",
        "Pos",
        "FA1",
        "FA2",
        "F_red",
        "Pri-Ter",
        "Pri-Pos",
        "Sec-Ter",
        "Sec-Pos",
        "Odd-Even",
        "DMM",
        "Shape",
    ]
    vals = [
        tlc.metrics["sig_pri"],
        tlc.metrics["sig_sec"],
        tlc.metrics["sig_ter"],
        tlc.metrics["sig_pos"],
        tlc.metrics["FA1"],
        tlc.metrics["FA2"],
        tlc.metrics["Fred"],
        tlc.metrics["sig_pri"] - tlc.metrics["sig_ter"],
        tlc.metrics["sig_pri"] - tlc.metrics["sig_pos"],
        tlc.metrics["sig_sec"] - tlc.metrics["sig_ter"],
        tlc.metrics["sig_sec"] - tlc.metrics["sig_pos"],
        tlc.metrics["sig_dep"],
        tlc.metrics["DMM"],
        tlc.metrics["SHP"],
    ]
    table = ax.table(
        cellText=[[f"{val:.2f}" for val in vals]], colLabels=labels, loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    table.scale(1.2, 1)
    ax.axis("off")


def plot_modshift(tlc, save_fig=False, save_file=None):
    phase, flux, deps = expanded_phase(tlc.phase, tlc.flux, tlc.dep_series)
    bin_cent, bin_mean, bin_err = binned_data(phase, flux, int(10 * 1.5 / tlc.qtran))
    # Set up plot
    fig = plt.figure(figsize=(6, 8))
    fs = 7
    gs = gridspec.GridSpec(nrows=5, ncols=3, hspace=0.4, wspace=0.3)
    # Metric scores
    axText = fig.add_subplot(gs[0, :])
    axText.set_title(f"TIC-{tlc.tic}.{tlc.planetno}: Modshift results", fontsize=fs)
    axText.text(
        0.5,
        0.8,
        f"P = {tlc.metrics['per']:.4f} days, E = {tlc.metrics['epo']:.4f} days",
        fontsize=fs,
        ha="center",
        va="center",
        transform=axText.transAxes,
    )
    modshift_text(tlc, axText, fs)
    # Phase diagram
    axPhase = fig.add_subplot(gs[1, :])
    axPhase.plot(phase, flux, "r.", ms=1)
    axPhase.plot(bin_cent, bin_mean, "b.", ms=2)
    axPhase.set_xlabel("Phase", fontsize=fs)
    # Depth time series
    axDeps = fig.add_subplot(gs[2, :])
    axDeps.plot(phase, -deps, "k", ms=1)
    axDeps.axhline(y=0, color="r")
    axDeps.axhline(y=3 * tlc.err * 1e6, color="b")
    axDeps.axhline(y=-3 * tlc.err * 1e6, color="b")
    for ax in [axPhase, axDeps]:
        ax.set_ylabel("Flux (ppm)", fontsize=fs)
        ax.axvspan(0.75, 1.25, alpha=0.5, color="grey")
        ax.set_xlim([-0.25, 1.25])
        ax.tick_params(axis="both", which="both", labelsize=fs)
    # Modshift results
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=3, subplot_spec=gs[3:, :], wspace=0.5
    )
    # Primary event
    axPri = fig.add_subplot(gs_sub[0, 0])
    axPri.set_ylabel("Flux (ppm)", fontsize=fs)
    modshift_box(
        "Primary", axPri, bin_cent, bin_mean, bin_err, tlc.metrics["phs_pri"], tlc.qtran
    )
    # Secondary events
    if not np.isnan(tlc.metrics["phs_sec"]):
        phs = tlc.metrics["phs_sec"]
        axSec = fig.add_subplot(gs_sub[1, 0])
        modshift_box("Secondary", axSec, bin_cent, bin_mean, bin_err, phs, tlc.qtran)
        axSec.set_xlabel("Phase", fontsize=fs)
        axSec.set_ylabel("Flux (ppm)", fontsize=fs)
        phs = phs - 1 if phs > 0.75 else phs
        axDeps.axvline(x=phs, ymax=0.03, marker="^", color="k")
    # Tertiary events
    if not np.isnan(tlc.metrics["phs_ter"]):
        phs = tlc.metrics["phs_ter"]
        axTer = fig.add_subplot(gs_sub[1, 1])
        modshift_box("Tertiary", axTer, bin_cent, bin_mean, bin_err, phs, tlc.qtran)
        axTer.set_xlabel("Phase", fontsize=fs)
        phs = phs - 1 if phs > 0.75 else phs
        axDeps.axvline(x=phs, ymax=0.03, marker="^", color="k")
    # Positive events
    if not np.isnan(tlc.metrics["phs_pos"]):
        phs = tlc.metrics["phs_pos"]
        axPos = fig.add_subplot(gs_sub[1, 2])
        modshift_box(
            "Positive", axPos, bin_cent, bin_mean, bin_err, phs, tlc.qtran, 0.5, 0.1
        )
        axPos.set_xlabel("Phase", fontsize=fs)
        phs = phs - 1 if phs > 0.75 else phs
        axDeps.axvline(x=phs, ymin=0.97, marker="v", color="k")
    # Odd and even events
    axOdd = fig.add_subplot(gs_sub[0, 1])
    axEven = fig.add_subplot(gs_sub[0, 2])
    phase2 = np.mod(tlc.time - tlc.epo, 2 * tlc.per) / tlc.per
    phase2[phase2 > 1] -= 2
    odd_tran = abs(phase2) < 0.5
    modshift_oddeven(
        "Odd",
        axOdd,
        tlc.phase[odd_tran],
        tlc.flux[odd_tran],
        tlc.dep_series[odd_tran],
        tlc.metrics["phs_pri"],
        tlc.qtran,
    )
    modshift_oddeven(
        "Even",
        axEven,
        tlc.phase[~odd_tran],
        tlc.flux[~odd_tran],
        tlc.dep_series[~odd_tran],
        tlc.metrics["phs_pri"],
        tlc.qtran,
    )
    for ax in [axOdd, axEven]:
        ax.set_ylim([axPri.get_ylim()[0], axPri.get_ylim()[1]])
    if save_fig:
        if save_file is None:
            save_file = f"{tlc.tic}.{tlc.planetno}.modshift.png"
        plt.savefig(save_file, bbox_inches="tight", dpi=150)


def plot_raw_det(gs, tlc, gap=20):
    diff = np.diff(tlc.time)
    start = np.append(tlc.time[0], tlc.time[1:][diff > gap]) - 1
    end = np.append(tlc.time[:-1][diff > gap], tlc.time[-1]) + 1
    bnds = [[start[i], end[i]] for i in range(len(start))]
    rawdet_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=2, ncols=1, subplot_spec=gs[0:2, :], hspace=0
    )
    axRaw = brokenaxes(
        xlims=bnds, subplot_spec=rawdet_gs[0, :], wspace=0.05, despine=False, d=0
    )
    axDet = brokenaxes(
        xlims=bnds, subplot_spec=rawdet_gs[1, :], wspace=0.05, despine=False, d=0
    )
    axRaw.plot(tlc.time, tlc.raw, "k.", ms=5)
    axDet.plot(tlc.time, tlc.flux, "k.", ms=5)
    for i in range(len(start) - 1):
        axRaw.axvline(x=start[i + 1], ls="--", color="k")
        axDet.axvline(x=start[i + 1], ls="--", color="k")
        axRaw.axvline(x=end[i], ls="--", color="k")
        axDet.axvline(x=end[i], ls="--", color="k")
    axRaw.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axRaw.tick_params(axis="y", which="both", labelsize=7)
    axDet.tick_params(axis="both", which="both", labelsize=7)
    axRaw.set_title(f"TIC-{tlc.tic}.{tlc.planetno} - Summary", fontsize=9)
    axDet.set_xlabel("BJD - 2457000", fontsize=7)
    # Mark all transits
    mid_transit = tlc.epo + np.unique(tlc.epochs) * tlc.per
    for epoch in np.unique(tlc.epochs):
        mid_transit = tlc.epo + epoch * tlc.per
        if epoch % 2 == 0:
            colour = _odd_colour
        else:
            colour = _even_colour
        if np.any((mid_transit > start) & (mid_transit < end)):
            axRaw.axvline(x=mid_transit, ymax=0.1, color=colour)
            axDet.axvline(x=mid_transit, ymax=0.1, color=colour)
    axRaw.set_ylabel("Relative flux", fontsize=7)
    axDet.set_ylabel("Relative flux", fontsize=7)
    axRaw.big_ax.text(
        0.005,
        0.95,
        "Raw",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="top",
        transform=axRaw.big_ax.transAxes,
    )
    axDet.big_ax.text(
        0.005,
        0.95,
        "Detrended",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="top",
        transform=axDet.big_ax.transAxes,
    )


def sort_lightcurve(time, flux, per, epo):
    phase = np.mod(time - epo, per) / per
    phase[phase > 0.5] -= 1
    sort = np.argsort(phase)
    return phase[sort], time[sort], flux[sort]


def plot_full_phase(ax, time, flux, per, epo, dur):
    phase, time, flux = sort_lightcurve(time, flux, per, epo)
    phase[phase < -0.25] += 1
    bin_cent, bin_mean, bin_err = binned_data(phase, flux, int(10 * per / dur))
    ax.plot(phase, flux, ".", color=_data_colour)
    ax.plot(bin_cent, bin_mean, ".", color=_bin_colour)
    ax.set_xlim([-0.25, 0.75])
    ax.text(
        0.01,
        0.05,
        "Phase diagram",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Phase", fontsize=7)
    ax.set_ylabel("Relative flux\n", fontsize=7)
    ax.tick_params(axis="both", which="both", labelsize=7)


def plot_close_phase(ax, time, flux, per, epo, dur):
    qtran = dur / per
    phase, time, flux = sort_lightcurve(time, flux, per, epo)
    near_tran = abs(phase) < 1.5 * qtran
    bin_cent, bin_mean, bin_err = binned_data(phase[near_tran], flux[near_tran], 30)
    ax.plot(phase[near_tran] * per * 24, flux[near_tran], ".", color=_data_colour)
    ax.plot(bin_cent * per * 24, bin_mean, ".", color=_bin_colour)
    ax.set_xlim([-1.5 * dur * 24, 1.5 * dur * 24])
    ax.text(
        0.02,
        0.05,
        "Primary",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Hours from midtransit", fontsize=7)
    ax.set_ylabel("Relative flux\n", fontsize=7)
    ax.tick_params(axis="both", which="both", labelsize=7)


def plot_odd_even(axOdd, axEven, time, flux, per, epo, dur, sig):
    qtran = dur / per
    # Identify odd and even cadences
    phase = np.mod(time - epo, 2 * per) / per
    phase[phase > 1] -= 2
    odd_tran = abs(phase) < 1.5 * qtran
    even_tran = abs(phase) > 1 - 1.5 * qtran
    # Phase fold
    phase = np.mod(time - epo, per) / per
    phase[phase > 0.5] -= 1
    if np.any(odd_tran):
        odd_cent, odd_mean, odd_err = binned_data(phase[odd_tran], flux[odd_tran], 30)
        axOdd.plot(phase[odd_tran] * per * 24, flux[odd_tran], ".", color=_data_colour)
        axOdd.plot(odd_cent * per * 24, odd_mean, ".", color=_bin_colour)
    if np.any(even_tran):
        even_cent, even_mean, even_err = binned_data(
            phase[even_tran], flux[even_tran], 30
        )
        axEven.plot(
            phase[even_tran] * per * 24, flux[even_tran], ".", color=_data_colour
        )
        axEven.plot(even_cent * per * 24, even_mean, ".", color=_bin_colour)
    axEven.text(
        0.96,
        0.05,
        f"({sig:.1f}$\sigma$)",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=axEven.transAxes,
    )
    for ax, label in zip([axOdd, axEven], ["Odd", "Even"]):
        ax.set_xlim([-1.5 * dur * 24, 1.5 * dur * 24])
        ax.set_xlabel("Hours from midtransit", fontsize=7)
        ax.text(
            0.04,
            0.05,
            label,
            fontsize=8,
            color=_label_colour,
            path_effects=[pe.withStroke(linewidth=4, foreground="w")],
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )
        ax.tick_params(axis="both", which="both", labelsize=7)
        ax.set_yticks([])


def plot_secondary(ax, time, flux, per, epo, dur, phs, dep, sig):
    qtran = dur / per
    phase, time, flux = sort_lightcurve(time, flux, per, epo)
    phase -= phs
    phase[phase < -0.5] += 1
    phase[phase > 0.5] -= 1
    near_tran = abs(phase) < 1.5 * qtran
    if np.any(near_tran):
        bin_cent, bin_mean, bin_err = binned_data(phase[near_tran], flux[near_tran], 30)
        ax.plot(phase[near_tran] * per * 24, flux[near_tran], ".", color=_data_colour)
        ax.plot(bin_cent * per * 24, bin_mean, ".", color=_bin_colour)
    ax.set_xlim([-1.5 * dur * 24, 1.5 * dur * 24])
    ax.text(
        0.02,
        0.05,
        "Secondary",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.05,
        f"{int(dep)} ppm ({sig:.1f}$\sigma$)",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlabel(f"Hours from secondary (phase {phs:.2f})", fontsize=7)
    ax.set_ylabel("Relative flux\n", fontsize=7)
    ax.tick_params(axis="both", which="both", labelsize=7)


def plot_half_phase(ax, time, flux, per, epo, dur):
    qtran = dur / per
    phase, time, flux = sort_lightcurve(time, flux, per, epo)
    phase -= 0.5
    phase[phase < -0.5] += 1
    near_tran = abs(phase) < 1.5 * qtran
    if np.any(near_tran):
        bin_cent, bin_mean, bin_err = binned_data(phase[near_tran], flux[near_tran], 30)
        ax.plot(phase[near_tran] * per * 24, flux[near_tran], ".", color=_data_colour)
        ax.plot(bin_cent * per * 24, bin_mean, ".", color=_bin_colour)
    ax.set_xlim([-1.5 * dur * 24, 1.5 * dur * 24])
    ax.text(
        0.02,
        0.05,
        "Half phase",
        fontsize=8,
        color=_label_colour,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")],
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlabel("Hours from secondary (phase 0.5)", fontsize=7)
    ax.tick_params(axis="both", which="both", labelsize=7)
    ax.set_yticks([])


def plot_individual_transits(ax, time, flux, per, epo, dur):
    qtran = dur / per
    phase, time, flux = sort_lightcurve(time, flux, per, epo)
    near_tran = abs(phase) < qtran
    N_transit = np.unique(np.round((time - epo) / per).astype(int))
    if len(N_transit) > 30:
        N_transit = N_transit[-30:]
    midpts = epo + per * N_transit
    minf, maxf = np.nanmin(flux[~near_tran]), np.nanmax(flux[near_tran])
    deltay = maxf - minf
    # model_phase, model_flux = phase_diagram(tlc.mtime, tlc.model, tlc.epo, tlc.per, 0.5)
    for i in range(len(N_transit)):
        idx = np.abs(time - midpts[i]) < 2 * dur
        ax.plot(phase[idx] * per * 24, flux[idx] - i * deltay, ".k-", ms=5, lw=1)
        # iax.plot(model_phase*24, model_flux - i*deltay,"r")
    ax.tick_params(axis="both", which="both", labelsize=7)
    ax.set_xlabel("Hours from midtransit", fontsize=7)
    ax.set_xlim([-2 * dur * 24, 2 * dur * 24])
    ax.set_yticks([])


def plot_text(ax, tlc, star, per, epo, dur, dep, spacing=0.043):
    # Define properties to include on summary page
    properties = [
        "TCE properties:",
        f"Period = {per:.4f} days",
        f"Epoch = {epo:.4f} BJTD",
        f"Duration = {24*dur:.2f} hours",
        f"Depth = {int(dep*1e6)} ppm",
        f"S/N = {tlc.metrics['MES']:.1f}",
        "",
        "Model properties:",
        f"$Rp/Rs$ = {tlc.metrics['transit_RpRs']:.3f} [{tlc.metrics['transit_RpRs_err']:.3f}]",
        f"$b$ = {tlc.metrics['transit_b']:.3f} [{tlc.metrics['transit_b_err']:.3f}]",
        f"$a/Rs$ = {tlc.metrics['transit_aRs']:.3f} [{tlc.metrics['transit_aRs_err']:.3f}]",
        "",
        "Planet properties:",
        f"$Rp$ = {tlc.metrics['Rp']:.3f} [{tlc.metrics['Rp_err']:.3f}] $R_\oplus$",
        f"$a$ = {tlc.metrics['a']:.3f} [{tlc.metrics['a_err']:.3f}] AU",
        f"$Seff$ = {tlc.metrics['Seff']:.1f} [{tlc.metrics['Seff_err']:.1f}] $S_\oplus$",
        "",
        "Stellar properties:",
        f"$Rs$ = {star['rad']:.2f} [{star['e_rad']:.2f}] $R_\odot$",
        f"$Ms$ = {star['mass']:.2f} [{star['e_mass']:.2f}] $M_\odot$",
        f"$Teff$ = {star['Teff']:.0f} [{star['e_Teff']:.0f}] K",
    ]
    line = 0.98
    for prop in properties:
        if prop in [
            "TCE properties:",
            "Model properties:",
            "Planet properties:",
            "Stellar properties:",
        ]:
            ax.annotate(prop, [0, line], fontsize=8, fontweight="bold")
        else:
            ax.annotate(prop, [0, line], fontsize=8)
        line -= spacing
    ax.axis("off")


def transit_setup(tlc):
    per = tlc.metrics["transit_per"]
    epo = tlc.metrics["transit_epo"]
    dur = tlc.metrics["transit_dur"]
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
    mtime = np.linspace(epo - 0.25 * per, epo + 0.75 * per, int(100 * per / dur))
    model = tm.model(tm.params, mtime)
    odd_params = tm.params
    odd_params["RpRs"].value = tlc.metrics["transit_odd_RpRs"]
    odd_params["epo"].value = tlc.metrics["transit_odd_epo"]
    odd_model = tm.model(odd_params, mtime)
    even_params = tm.params
    even_params["RpRs"].value = tlc.metrics["transit_even_RpRs"]
    even_params["epo"].value = tlc.metrics["transit_even_epo"]
    even_model = tm.model(even_params, mtime)
    return per, epo, dur, mtime, model, odd_model, even_model


def trapezoid_setup(tlc):
    per = tlc.metrics["trap_per"]
    epo = tlc.metrics["trap_epo"]
    dur = tlc.metrics["trap_dur"]
    tm = TrapezoidModel(
        tlc.metrics["trap_per"],
        tlc.metrics["trap_epo"],
        tlc.metrics["trap_dep"],
        tlc.metrics["trap_qtran"],
        tlc.metrics["trap_qin"],
        tlc.metrics["trap_zpt"],
    )
    mtime = np.linspace(epo - 0.25 * per, epo + 0.75 * per, int(100 * per / dur))
    model = tm.model(tm.params, mtime)
    odd_params = tm.params
    odd_params["dep"].value = tlc.metrics["trap_odd_dep"]
    odd_params["epo"].value = tlc.metrics["trap_odd_epo"]
    odd_model = tm.model(odd_params, mtime)
    even_params = tm.params
    even_params["dep"].value = tlc.metrics["trap_even_dep"]
    even_params["epo"].value = tlc.metrics["trap_even_epo"]
    even_model = tm.model(even_params, mtime)
    return per, epo, dur, mtime, model, odd_model, even_model


def plot_summary(tlc, star, save_fig=False, save_file=None):
    # Set up plotting
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(nrows=5, ncols=6, hspace=0.5, wspace=0.3)
    axPhase = fig.add_subplot(gs[2, :4])
    axClose = fig.add_subplot(gs[3, :2])
    oddeven_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs[3, 2:4], wspace=0
    )
    axOdd = fig.add_subplot(oddeven_gs[:, 0])
    axEven = fig.add_subplot(oddeven_gs[:, 1])
    axSec = fig.add_subplot(gs[4, :2])
    axHalf = fig.add_subplot(gs[4, 2:4])
    axTrans = fig.add_subplot(gs[2:, 4])
    axText = fig.add_subplot(gs[2:, -1])
    if ~np.isnan(tlc.metrics["transit_aic"]):
        per, epo, dur, mtime, model, odd_model, even_model = transit_setup(tlc)
        low = np.nanmin(model)
        dep = max(model) - min(model)
        plot_model = True
    elif ~np.isnan(tlc.metrics["trap_aic"]):
        per, epo, dur, mtime, model, odd_model, even_model = trapezoid_setup(tlc)
        low = np.nanmin(model)
        dep = max(model) - min(model)
        plot_model = True
    else:
        per, epo, dur = tlc.per, tlc.epo, tlc.dur
        low = 1 - tlc.dep
        dep = tlc.dep
        plot_model = False
    if np.isnan(dur):
        dur = tlc.metrics["dur"]
    if ~np.isnan(tlc.metrics["trap_sig_dep"]):
        sig_dep = tlc.metrics["trap_sig_dep"]
    elif ~np.isnan(tlc.metrics["transit_sig_dep"]):
        sig_dep = tlc.metrics["transit_sig_dep"]
    elif ~np.isnan(tlc.metrics["sig_dep"]):
        sig_dep = tlc.metrics["sig_dep"]
    else:
        sig_dep = np.nan
    # Plot raw and detrended light curves
    plot_raw_det(gs, tlc)
    # Plot full phase diagram
    plot_full_phase(axPhase, tlc.time, tlc.flux, per, epo, dur)
    if plot_model:
        axPhase.plot((mtime - epo) / per, model, "r")
    # Plot close-up phase diagram
    plot_close_phase(axClose, tlc.time, tlc.flux, per, epo, dur)
    if plot_model:
        axClose.plot((mtime - epo) * 24, model, "r")
    # Plot odd transits
    plot_odd_even(
        axOdd,
        axEven,
        tlc.time,
        tlc.flux,
        per,
        epo,
        dur,
        sig_dep,
    )
    if plot_model:
        axOdd.plot((mtime - epo) * 24, odd_model, color=_odd_colour)
    if plot_model:
        axEven.plot((mtime - epo) * 24, even_model, color=_even_colour)
    # Plot secondary
    phs = tlc.metrics["phs_sec"]
    if ~np.isnan(phs):
        if phs > 0.75:
            phs -= 1
        axPhase.axvline(x=phs, ymax=0.03, marker="^", color="k")
        plot_secondary(
            axSec,
            tlc.time,
            tlc.flux,
            per,
            epo,
            dur,
            phs,
            tlc.metrics["dep_sec"] * 1e6,
            tlc.metrics["sig_sec"],
        )
    else:
        axSec.axis("off")
    # Plot half phase
    plot_half_phase(axHalf, tlc.time, tlc.flux, per, epo, dur)
    axHalf.set_ylim(axSec.get_ylim())
    # Plot individual transits
    plot_individual_transits(axTrans, tlc.time, tlc.flux, per, epo, dur)
    # Add planet properties
    plot_text(axText, tlc, star, per, epo, dur, dep)
    # Set axis limits
    sigma = 1.4826 * MAD(tlc.flux[~tlc.near_tran])
    for ax in [axPhase, axClose, axOdd, axEven]:
        ax.set_ylim([low - 4 * sigma, 1 + 4 * sigma])
    if save_fig:
        if save_file is None:
            save_file = f"{tlc.tic}.{tlc.planetno}.summary.png"
        plt.savefig(save_file, bbox_inches="tight", dpi=150)


def plot_summary_with_diff(
    tlc, star, tdi=None, pixel_data=None, save_fig=False, save_file=None
):
    # Set up plotting
    fig = plt.figure(figsize=(12.5, 7))
    gs = gridspec.GridSpec(nrows=5, ncols=8, hspace=0.5, wspace=0.3)
    axPhase = fig.add_subplot(gs[2, :4])
    axClose = fig.add_subplot(gs[3, :2])
    oddeven_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs[3, 2:4], wspace=0
    )
    axOdd = fig.add_subplot(oddeven_gs[:, 0])
    axEven = fig.add_subplot(oddeven_gs[:, 1])
    axSec = fig.add_subplot(gs[4, :2])
    axHalf = fig.add_subplot(gs[4, 2:4])
    axTrans = fig.add_subplot(gs[2:, 4])
    axText = fig.add_subplot(gs[2:, 7])
    pix_gs = gridspec.GridSpecFromSubplotSpec(
        nrows=3, ncols=2, subplot_spec=gs[2:, 5:7], hspace=0.5
    )
    axSNR1 = fig.add_subplot(pix_gs[0, 0])
    axSNR2 = fig.add_subplot(pix_gs[0, 1])
    axDiff1 = fig.add_subplot(pix_gs[1, 0])
    axDiff2 = fig.add_subplot(pix_gs[1, 1])
    axDir1 = fig.add_subplot(pix_gs[2, 0])
    axDir2 = fig.add_subplot(pix_gs[2, 1])
    if ~np.isnan(tlc.metrics["transit_aic"]):
        per, epo, dur, mtime, model, odd_model, even_model = transit_setup(tlc)
        low = np.nanmin(model)
        dep = max(model) - min(model)
        plot_model = True
    elif ~np.isnan(tlc.metrics["trap_aic"]):
        per, epo, dur, mtime, model, odd_model, even_model = trapezoid_setup(tlc)
        low = np.nanmin(model)
        dep = max(model) - min(model)
        plot_model = True
    else:
        per, epo, dur = tlc.per, tlc.epo, tlc.dur
        low = 1 - tlc.dep
        dep = tlc.dep
        plot_model = False
    if np.isnan(dur):
        dur = tlc.metrics["dur"]
    if ~np.isnan(tlc.metrics["trap_sig_dep"]):
        sig_dep = tlc.metrics["trap_sig_dep"]
    elif ~np.isnan(tlc.metrics["transit_sig_dep"]):
        sig_dep = tlc.metrics["transit_sig_dep"]
    elif ~np.isnan(tlc.metrics["sig_dep"]):
        sig_dep = tlc.metrics["sig_dep"]
    else:
        sig_dep = np.nan
    # Plot raw and detrended light curves
    plot_raw_det(gs, tlc)
    # Plot full phase diagram
    plot_full_phase(axPhase, tlc.time, tlc.flux, per, epo, dur)
    if plot_model:
        axPhase.plot((mtime - epo) / per, model, "r")
    # Plot close-up phase diagram
    plot_close_phase(axClose, tlc.time, tlc.flux, per, epo, dur)
    if plot_model:
        axClose.plot((mtime - epo) * 24, model, "r")
    # Plot odd transits
    plot_odd_even(
        axOdd,
        axEven,
        tlc.time,
        tlc.flux,
        per,
        epo,
        dur,
        sig_dep,
    )
    if plot_model:
        axOdd.plot((mtime - epo) * 24, odd_model, color=_odd_colour)
    if plot_model:
        axEven.plot((mtime - epo) * 24, even_model, color=_even_colour)
    # Plot secondary
    phs = tlc.metrics["phs_sec"]
    if ~np.isnan(phs):
        if phs > 0.75:
            phs -= 1
        axPhase.axvline(x=phs, ymax=0.03, marker="^", color="k")
        plot_secondary(
            axSec,
            tlc.time,
            tlc.flux,
            per,
            epo,
            dur,
            phs,
            tlc.metrics["dep_sec"] * 1e6,
            tlc.metrics["sig_sec"],
        )
    else:
        axSec.axis("off")
    # Plot half phase
    plot_half_phase(axHalf, tlc.time, tlc.flux, per, epo, dur)
    axHalf.set_ylim(axSec.get_ylim())
    # Plot individual transits
    plot_individual_transits(axTrans, tlc.time, tlc.flux, per, epo, dur)
    # Add planet properties
    plot_text(axText, tlc, star, per, epo, dur, dep)
    # Set axis limits
    sigma = 1.4826 * MAD(tlc.flux[~tlc.near_tran])
    for ax in [axPhase, axClose, axOdd, axEven]:
        ax.set_ylim([low - 4 * sigma, 1 + 4 * sigma])
    # Plot difference and direct images
    if (tdi is not None) and (pixel_data is not None):
        images = pixel_data[0]
        catalogue = pixel_data[1]
        tdi.draw_pix_catalog(
            images["diffSNRImage"],
            catalogue,
            catalogue["extent"],
            ax=axSNR1,
            fs=7,
            ss=10,
            filterStars=True,
            dMagThreshold=4,
        )
        tdi.draw_pix_catalog(
            images["diffSNRImage"],
            catalogue,
            catalogue["extentClose"],
            ax=axSNR2,
            fs=7,
            ss=40,
            filterStars=True,
            dMagThreshold=4,
            close=True,
        )
        tdi.draw_pix_catalog(
            images["diffImage"],
            catalogue,
            catalogue["extent"],
            ax=axDiff1,
            fs=7,
            ss=10,
            filterStars=True,
            dMagThreshold=4,
        )
        tdi.draw_pix_catalog(
            images["diffImage"],
            catalogue,
            catalogue["extentClose"],
            ax=axDiff2,
            fs=7,
            ss=40,
            filterStars=True,
            dMagThreshold=4,
            close=True,
        )
        tdi.draw_pix_catalog(
            images["meanOutTransit"],
            catalogue,
            catalogue["extent"],
            ax=axDir1,
            fs=7,
            ss=10,
            filterStars=True,
            dMagThreshold=4,
        )
        tdi.draw_pix_catalog(
            images["meanOutTransit"],
            catalogue,
            catalogue["extentClose"],
            ax=axDir2,
            fs=7,
            ss=40,
            filterStars=True,
            dMagThreshold=4,
            close=True,
        )
        axSNR1.set_title("Difference SNR", fontsize=7)
        axSNR2.set_title("Difference SNR (close)", fontsize=7)
        axDiff1.set_title("Difference image", fontsize=7)
        axDiff2.set_title("Difference image (close)", fontsize=7)
        axDir1.set_title("Direct image", fontsize=7)
        axDir2.set_title("Direct image (close)", fontsize=7)
    for ax in [axSNR1, axSNR2, axDiff1, axDiff2, axDir1, axDir2]:
        ax.axis("off")
    if save_fig:
        if save_file is None:
            save_file = f"{tlc.tic}.{tlc.planetno}.full.png"
        plt.savefig(save_file, bbox_inches="tight", dpi=150)


def plot_diffimages(
    tic,
    planetno,
    tdi,
    sectors,
    pixel_data,
    annotate=False,
    save_fig=False,
    save_file=None,
):
    n_sectors = len(sectors)
    fs = 10
    # Set up plotting
    fig, ax = plt.subplots(max(n_sectors, 2), 4, figsize=(4 * 2, max(n_sectors, 2) * 2))
    fig.suptitle(f"TIC-{tic}.{planetno}: Difference images", fontsize=fs)
    ax[0, 0].set_title("Diff Image\n", fontsize=fs)
    ax[0, 1].set_title("Diff Image\n(Close-up)", fontsize=fs)
    ax[0, 2].set_title("Direct Image\n", fontsize=fs)
    ax[0, 3].set_title("Direct Image\n(Close-up)", fontsize=fs)
    # Plot each differece image result on its own line
    for i in range(n_sectors):
        sector = sectors[i]
        images = pixel_data[i][0]
        catalogue = pixel_data[i][1]
        ax[i, 0].set_ylabel(f"Sector {sector}", fontsize=fs)
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        tdi.draw_pix_catalog(
            images["diffImage"],
            catalogue,
            catalogue["extent"],
            ax=ax[i, 0],
            fs=fs,
            ss=50,
            filterStars=True,
            dMagThreshold=4,
            annotate=annotate,
        )
        tdi.draw_pix_catalog(
            images["diffImage"],
            catalogue,
            catalogue["extentClose"],
            ax=ax[i, 1],
            fs=fs,
            ss=200,
            filterStars=True,
            dMagThreshold=4,
            annotate=annotate,
            close=True,
        )
        tdi.draw_pix_catalog(
            images["meanOutTransit"],
            catalogue,
            catalogue["extent"],
            ax=ax[i, 2],
            fs=fs,
            ss=50,
            filterStars=True,
            dMagThreshold=4,
            annotate=annotate,
        )
        tdi.draw_pix_catalog(
            images["meanOutTransit"],
            catalogue,
            catalogue["extentClose"],
            ax=ax[i, 3],
            fs=fs,
            ss=200,
            filterStars=True,
            dMagThreshold=4,
            annotate=annotate,
            close=True,
        )
        for n in [1, 2, 3]:
            ax[i, n].axis("off")
    if n_sectors < 2:
        for n in [0, 1, 2, 3]:
            ax[1, n].axis("off")
    if save_fig:
        if save_file is None:
            save_file = f"{tic}.{planetno}.diffimages.png"
        plt.savefig(save_file, bbox_inches="tight", dpi=100)
