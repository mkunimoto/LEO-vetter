import numpy as np
import csv
import os

from leo_vetter.utils import phasefold, weighted_mean, weighted_err, weighted_std
from leo_vetter import fits, oddeven, individual, modshift, parameters


class TCELightCurve:
    def __init__(self, tic, time, raw, flux, flux_err, per, epo, dur, planetno=1):
        self.tic = int(tic)
        self.planetno = int(planetno)
        self.time = time
        self.raw = raw
        self.flux = flux
        self.flux_err = flux_err
        self.per = per
        self.epo = epo
        self.dur = dur
        self.qtran = dur / per
        # Phase spans -0.5 to 0.5 with transit at 0
        self.phase = phasefold(time, per, epo)
        # Cadences in-transit
        self.in_tran = abs(self.phase) < 0.5 * self.qtran
        # Cadences in odd and even transits
        phase2 = np.mod(time - epo, 2 * per) / per
        phase2[phase2 > 1] -= 2
        self.odd_tran = abs(phase2) < 0.5 * self.qtran
        self.even_tran = abs(phase2) > 1 - 0.5 * self.qtran
        # Cadences within 1 transit duration
        self.near_tran = abs(self.phase) < self.qtran
        # Cadences within 2 transit durations
        self.fit_tran = abs(self.phase) < 2 * self.qtran
        # Cadences before transit
        self.before_tran = (self.phase < -0.5 * self.qtran) & (
            self.phase > -1.5 * self.qtran
        )
        # Cadences after transit
        self.after_tran = (self.phase > 0.5 * self.qtran) & (
            self.phase < 1.5 * self.qtran
        )
        # Actual number of transits accounting for gaps
        self.epochs = np.round((time - epo) / per)
        self.tran_epochs = np.unique(self.epochs[self.in_tran])
        self.N_transit = len(self.tran_epochs)
        # Number of transit datapoints
        self.n_in = np.sum(self.in_tran)
        self.n_before = np.sum(self.before_tran)
        self.n_after = np.sum(self.after_tran)
        # Out-of-transit flux and transit depth
        self.zpt = weighted_mean(
            self.flux[~self.near_tran], self.flux_err[~self.near_tran]
        )
        self.dep = self.zpt - weighted_mean(
            self.flux[self.in_tran], self.flux_err[self.in_tran]
        )
        # Initialize dict for storing metrics
        metrics = {}
        metrics["tic"] = self.tic
        metrics["planetno"] = self.planetno
        metrics["per"] = per
        metrics["epo"] = epo
        metrics["dur"] = dur
        metrics["qtran"] = self.qtran
        metrics["N_transit"] = self.N_transit
        metrics["n_in"] = self.n_in
        metrics["n_before"] = self.n_before
        metrics["n_after"] = self.n_after
        metrics["zpt"] = self.zpt
        metrics["dep"] = self.dep
        self.metrics = metrics

    def get_SES_MES(self, replace=False):
        if hasattr(self, "MES_series") and not replace:
            print("SES/MES series already computed")
            return
        N = len(self.time)
        dep_SES = np.zeros(N)
        n_SES = np.zeros(N)
        dep_MES = np.zeros(N)
        n_MES = np.zeros(N)
        N_transit_MES = np.zeros(N)
        bin_flux = np.zeros(N)
        bin_flux_err = np.zeros(N)
        phase = phasefold(self.time, self.per, self.epo)
        phase[phase < 0] += 1
        dur_half = 0.5 * self.dur
        left_idx, right_idx = 0, 0
        time_diff_dur_minus, time_diff_dur_plus = self.time - dur_half, self.time + dur_half
        phase_sorted_idxs = np.argsort(phase)
        phase_sorted = phase[phase_sorted_idxs]
        dur_per_frac_half = 0.5 * self.qtran
        for i in np.arange(N):
            
            # time is sorted; # Get individual transit depth at this cadence, i.e. only use datapoints close in time
            while left_idx < N and self.time[left_idx] < time_diff_dur_minus[i]:
                left_idx += 1
            if right_idx < left_idx:
                right_idx = left_idx
            while right_idx < N and self.time[right_idx] <= time_diff_dur_plus[i]:
                right_idx += 1
            
            n_SES[i] = right_idx - left_idx
            dep_SES[i] = self.zpt - weighted_mean(self.flux[left_idx:right_idx], self.flux_err[left_idx:right_idx])
            
            # Get overall transit depth at this cadence, i.e. use all datapoints close in phase
            p = phase[i]
            
            # find indices within +- range from the current phase index
            left_p  = np.searchsorted(phase_sorted, p - dur_per_frac_half, side='right')
            right_p = np.searchsorted(phase_sorted, p + dur_per_frac_half, side='left')
            idx_window = phase_sorted_idxs[left_p:right_p]

            # handle phase wrapping
            if p < dur_per_frac_half:
                wrap = np.searchsorted(phase_sorted, p - dur_per_frac_half + 1.0, side='left')
                idx_wrap = phase_sorted_idxs[wrap:]
                all_tran_idxs = np.concatenate((idx_window, idx_wrap))
            elif p > (1.0 - dur_per_frac_half):
                wrap = np.searchsorted(phase_sorted, p + dur_per_frac_half - 1.0, side='right')
                idx_wrap = phase_sorted_idxs[:wrap]
                all_tran_idxs = np.concatenate((idx_window, idx_wrap))
            else:
                all_tran_idxs = idx_window

            # avoid possible duplicates
            all_tran_idxs = np.unique(all_tran_idxs)
            
            n_MES[i] = len(all_tran_idxs)
            dep_MES[i] = self.zpt - weighted_mean(self.flux[all_tran_idxs], self.flux_err[all_tran_idxs])

            # compute local epochs relative to current timestamp            
            all_tran_time = self.time[all_tran_idxs]
            local_epochs = np.round((all_tran_time - self.time[i]) / self.per)
            N_transit_MES[i] = len(np.unique(local_epochs))
            
            # Get running mean and uncertainty of out-of-transit fluxes, binned over transit timescale
            mask_near_tran = ~self.near_tran[left_idx:right_idx]
            bin_flux[i] = weighted_mean(self.flux[left_idx:right_idx][mask_near_tran], 
                                        self.flux_err[left_idx:right_idx][mask_near_tran])
            bin_flux_err[i] = weighted_err(self.flux[left_idx:right_idx][mask_near_tran], 
                                           self.flux_err[left_idx:right_idx][mask_near_tran])
            
        # Estimate white and red noise following Hartman & Bakos (2016)
        mask = ~np.isnan(bin_flux) & ~self.near_tran
        std = weighted_std(self.flux[mask], self.flux_err[mask])
        bin_std = weighted_std(bin_flux[mask], bin_flux_err[mask])
        expected_bin_std = (
            std
            * np.sqrt(np.nanmean(bin_flux_err[mask] ** 2))
            / np.sqrt(np.nanmean(self.flux_err[mask] ** 2))
        )
        self.sig_w = std
        sig_r2 = bin_std**2 - expected_bin_std**2
        self.sig_r = np.sqrt(sig_r2) if sig_r2 > 0 else 0
        # Estimate signal-to-pink-noise following Pont et al. (2006)
        self.err = np.sqrt(
            (self.sig_w**2 / self.n_in) + (self.sig_r**2 / self.N_transit)
        )
        err_SES = np.sqrt((self.sig_w**2 / n_SES) + self.sig_r**2)
        err_MES = np.sqrt((self.sig_w**2 / n_MES) + (self.sig_r**2 / N_transit_MES))
        self.SES_series = dep_SES / err_SES
        self.dep_series = dep_MES
        self.err_series = err_MES
        self.MES_series = dep_MES / err_MES
        self.metrics["sig_w"] = self.sig_w
        self.metrics["sig_r"] = self.sig_r
        self.metrics["err"] = self.err
        self.metrics["MES"] = self.dep / self.err
        Fmin = np.nanmin(-self.dep_series)
        Fmax = np.nanmax(-self.dep_series)
        self.metrics["SHP"] = Fmax / (Fmax - Fmin)

    def compute_flux_metrics(
        self,
        star,
        verbose=True,
        cap_b=True,
        frac=0.7,
        gap=0.3,
        chases=0.01,
        rubble=0.75,
        redchi2=5,
        A=0.3,
    ):
        if verbose:
            print("Estimating SES and MES time series...")
        self.get_SES_MES()
        if verbose:
            print("Fitting linear, trapezoid, and transit models...")
        fits.linear(self)
        fits.trapezoid(self)
        fits.half_trapezoid(self, "left")
        fits.half_trapezoid(self, "right")
        fits.transit(self, star["u1"], star["u2"], cap_b=cap_b)
        if verbose:
            print("Running SWEET test...")
        fits.sweet(self)
        if verbose:
            print("Getting odd-even metrics...")
        oddeven.box(self)
        oddeven.trapezoid(self)
        oddeven.transit(self, cap_b=cap_b)
        if verbose:
            print("Checking individual transit events...")
        individual.transit_events(self, frac=frac, gap=gap)
        individual.recompute_MES(self, chases=chases, rubble=rubble, redchi2=redchi2)
        if verbose:
            print("Running modshift...")
        modshift.uniqueness(self)
        if verbose:
            print("Estimating derived parameters...")
        parameters.derived_parameters(self, star, A=A)
        if verbose:
            print("Done!")

    def save_metrics(self, save_file=None):
        if save_file is None:
            save_file = f"{self.tic}.{self.planetno}.metrics"
        with open(save_file, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(self.metrics.keys())
            writer.writerow(self.metrics.values())
