import numpy as np
import pandas as pd
import os

from scipy.interpolate import NearestNDInterpolator

data_dir = os.path.join(os.path.dirname(__file__), "data")
table15 = pd.read_csv(os.path.join(data_dir, "claret_2017_table15.csv.gz"))
table25 = pd.read_csv(os.path.join(data_dir, "claret_2017_table25.csv.gz"))
table15 = table15[table15["xi"] == 2]
table25 = table25[table25["xi"] == 2]


def get_interpolator(cool):
    """
    Limb-darkening parameters are from Claret (2017)
    https://ui.adsabs.harvard.edu/abs/2017A%26A...600A..30C/abstract
    Tables downloaded from Vizier
    http://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J%2FA%2BA%2F600%2FA30
    """
    table = table15 if cool else table25
    points = table[["teff", "logg", "feh"]].values
    values = table[["u1", "u2"]].values
    interp_u1 = NearestNDInterpolator(points, values.T[0], rescale=True)
    interp_u2 = NearestNDInterpolator(points, values.T[1], rescale=True)
    return interp_u1, interp_u2


def quadratic_ldc(teff, logg, feh=0.0):
    if np.isscalar(teff):
        cool = True if teff < 3500.0 else False
        interp_u1, interp_u2 = get_interpolator(cool)
        u1 = interp_u1(teff, logg, feh)
        u2 = interp_u2(teff, logg, feh)
    else:
        if feh == 0.0:
            feh = np.zeros_like(teff)
        interp_u1, interp_u2 = get_interpolator(False)
        u1 = interp_u1(teff, logg, feh)
        u2 = interp_u2(teff, logg, feh)
        interp_u1, interp_u2 = get_interpolator(True)
        cool = teff < 3500
        u1[cool] = interp_u1(teff[cool], logg[cool], feh[cool])
        u2[cool] = interp_u2(teff[cool], logg[cool], feh[cool])
    return u1, u2
