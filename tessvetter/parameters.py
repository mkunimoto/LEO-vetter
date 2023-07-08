import numpy as np

from tessvetter.constants import *


def planet_radius(RpRs, Rs, RpRs_err, Rs_err):
    Rp = Rsun_in_Rearth * RpRs * Rs
    Rp_err = Rsun_in_Rearth * np.sqrt((RpRs_err * Rs) ** 2 + (Rs_err * RpRs) ** 2)
    return Rp, Rp_err


def semi_major_axis(per, Ms, per_err, Ms_err):
    a = (Ms * (per / 365.25) ** 2) ** (1.0 / 3)
    a_err = 365.25 ** (-2.0 / 3.0) * np.sqrt(
        (Ms_err * 1.0 / 3 * (per / Ms) ** (2.0 / 3)) ** 2
        + (per_err * 2.0 / 3 * (Ms / per) ** (1.0 / 3)) ** 2
    )
    return a, a_err


def equilibrium_temp(a, Rs, Teff, a_err, Rs_err, Teff_err, A=0.3):
    Teq = Teff * (1 - A) ** (1.0 / 4) * np.sqrt(0.5 * Rs * Rsun_in_AU / a)
    Teq_err = (
        (1 - A) ** (1.0 / 4)
        * np.sqrt(0.5 * Rsun_in_AU)
        * np.sqrt(
            (Teff_err * np.sqrt(Rs / a)) ** 2
            + (Rs_err * Teff / (2 * np.sqrt(Rs * a))) ** 2
            + (a_err * Teff * np.sqrt(Rs / a**3) / 2.0) ** 2
        )
    )
    return Teq, Teq_err


def insolation(a, Rs, Teff, a_err, Rs_err, Teff_err):
    Seff = (Rs / a) ** 2 * (Teff / Tsun) ** 4
    Seff_err = (1.0 / Tsun) ** 4 * np.sqrt(
        (Rs_err * 2 * Rs / a**2 * Teff**4) ** 2
        + (a_err * 2 * Rs**2 / a**3 * Teff**4) ** 2
        + (Teff_err * (Rs / a) ** 2 * 4 * Teff**3) ** 2
    )
    return Seff, Seff_err


def albedo_from_sec(dep_sec, a, Rp, dep_sec_err, a_err, Rp_err):
    albedo = dep_sec * (a * AU / (Rp * Rearth)) ** 2
    albedo_err = albedo * np.sqrt(
        (dep_sec_err / dep_sec) ** 2 + (2 * a_err / a) ** 2 + (2 * Rp_err / Rp) ** 2
    )
    return albedo, albedo_err


def radius_ratio(dep, dep_err):
    RpRs = np.sqrt(dep)
    RpRs_err = 0.5 * dep_err / np.sqrt(dep)
    return RpRs, RpRs_err


def transit_duration(per, RpRs, b, aRs, per_err, RpRs_err, b_err, aRs_err):
    C1 = (1 + RpRs) ** 2 - b**2
    C3 = aRs**2 - b**2
    C1_err = np.sqrt((RpRs_err * 2 * (1 + RpRs)) ** 2 + (b_err * 2 * b) ** 2)
    C3_err = np.sqrt((aRs_err * 2 * aRs) ** 2 + (b_err * 2 * b) ** 2)
    dur = per / np.pi * np.arcsin(np.sqrt(C1 / C3))
    dur_err = (
        1.0
        / np.pi
        * np.sqrt(
            (per_err * np.arcsin(np.sqrt(C1 / C3))) ** 2
            + (C1_err * per * np.sqrt(C3**2 / (C1 * (C3 - C1))) / (2 * C3)) ** 2
            + (C3_err * per * C1 * np.sqrt(C3**2 / (C1 * (C3 - C1))) / (2 * C3**2))
            ** 2
        )
    )
    return dur, dur_err


def trapezoid_duration(per, qtran, per_err, qtran_err):
    dur = per * qtran
    dur_err = np.sqrt((qtran * per_err) ** 2 + (per * qtran_err) ** 2)
    return dur, dur_err


def get_q(per, rho):
    return np.arcsin(0.2375 / rho**(1./3) / per**(2./3)) / np.pi


def derived_parameters(tlc, star, A=0.3):
    Rs = star["rad"]
    Ms = star["mass"]
    Teff = star["Teff"]
    Rs_err = star["e_rad"]
    Ms_err = star["e_mass"]
    Teff_err = star["e_Teff"]
    rho = star["rho"]
    # Set default parameters
    per, per_err = tlc.metrics["per"], np.nan
    RpRs, RpRs_err = np.sqrt(tlc.metrics["dep"]), np.nan
    transit_dur, transit_dur_err = np.nan, np.nan
    trap_dur, trap_dur_err = np.nan, np.nan
    # Improve parameters from model fits, if available
    if "trap_aic" in tlc.metrics:
        per, per_err = tlc.metrics["trap_per"], tlc.metrics["trap_per_err"]
        RpRs, RpRs_err = radius_ratio(
            tlc.metrics["trap_dep"],
            tlc.metrics["trap_dep_err"],
        )
        # Transit duration from trapezoid model fit
        trap_dur, trap_dur_err = trapezoid_duration(
            tlc.metrics["trap_per"],
            tlc.metrics["trap_qtran"],
            tlc.metrics["trap_per_err"],
            tlc.metrics["trap_qtran_err"],
        )
    if "transit_aic" in tlc.metrics:
        per, per_err = tlc.metrics["transit_per"], tlc.metrics["transit_per_err"]
        RpRs, RpRs_err = tlc.metrics["transit_RpRs"], tlc.metrics["transit_RpRs_err"]
        # Transit duration from transit model fit
        transit_dur, transit_dur_err = transit_duration(
            tlc.metrics["transit_per"],
            tlc.metrics["transit_RpRs"],
            tlc.metrics["transit_b"],
            tlc.metrics["transit_aRs"],
            tlc.metrics["transit_per_err"],
            tlc.metrics["transit_RpRs_err"],
            tlc.metrics["transit_b_err"],
            tlc.metrics["transit_aRs_err"],
        )
    # Expected qtran (duration/period) for a circular orbit 
    qtran_exp = get_q(per, rho)
    # Planet radius
    Rp, Rp_err = planet_radius(RpRs, Rs, RpRs_err, Rs_err)
    # Semi-major axis
    a, a_err = semi_major_axis(per, Ms, per_err, Ms_err)
    # Semi-major axis/Stellar radius
    aRs = a / (Rs * Rsun_in_AU)
    aRs_err = np.sqrt((a_err / Rs) ** 2 + (a * Rs_err / Rs**2) ** 2) / Rsun_in_AU
    # Equilibrium temperature
    Teq, Teq_err = equilibrium_temp(a, Rs, Teff, a_err, Rs_err, Teff_err, A)
    # Insolation
    Seff, Seff_err = insolation(a, Rs, Teff, a_err, Rs_err, Teff_err)
    if tlc.metrics["sig_sec"] > 0:
        # Albedo estimated from secondary
        albedo, albedo_err = albedo_from_sec(
            tlc.metrics["dep_sec"], a, Rp, tlc.metrics["err_sec"], a_err, Rp_err
        )
    else:
        albedo, albedo_err = np.nan, np.nan
    tlc.metrics["trap_dur"], tlc.metrics["trap_dur_err"] = trap_dur, trap_dur_err
    tlc.metrics["transit_dur"], tlc.metrics["transit_dur_err"] = (
        transit_dur,
        transit_dur_err,
    )
    tlc.metrics["q"] = qtran_exp
    tlc.metrics["Rp"], tlc.metrics["Rp_err"] = Rp, Rp_err
    tlc.metrics["a"], tlc.metrics["a_err"] = a, a_err
    tlc.metrics["aRs"], tlc.metrics["aRs_err"] = aRs, aRs_err
    tlc.metrics["Teq"], tlc.metrics["Teq_err"] = Teq, Teq_err
    tlc.metrics["Seff"], tlc.metrics["Seff_err"] = Seff, Seff_err
    tlc.metrics["albedo"], tlc.metrics["albedo_err"] = albedo, albedo_err
