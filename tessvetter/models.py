import numpy as np
import batman

from lmfit import Parameters
from tessvetter.utils import phasefold


class Model:
    def residual(self, params, t, y, dy):
        _model = self.model(params, t)
        resid = np.sqrt((y - _model) ** 2 / dy**2)
        return resid


class LinearModel(Model):
    def __init__(self, zpt=None, slope=None, params=None):
        if params is None:
            params = Parameters()
            params.add("zpt", value=zpt, min=0)
            params.add("slope", value=slope)
        self.params = params

    def model(self, params, t):
        _model = params["zpt"].value + params["slope"].value * t
        return _model


class TrapezoidModel(Model):
    def __init__(
        self, per=None, epo=None, dep=None, qtran=None, qin=None, zpt=None, params=None
    ):
        if params is None:
            params = Parameters()
            params.add("per", value=per, min=0.9 * per, max=1.1 * per)
            params.add("epo", value=epo, min=epo - 0.1 * per, max=epo + 0.1 * per)
            params.add("dep", value=dep, min=0, max=1)
            params.add("qtran", value=qtran, min=0, max=1)
            params.add("qin", value=qin, min=0, max=0.5)
            params.add("zpt", value=zpt, min=0)
        self.params = params

    def model(self, params, t):
        dep = params["dep"].value
        qtran = params["qtran"].value
        qin = params["qin"].value
        phase = np.abs(phasefold(t, params["per"].value, params["epo"].value))
        transit = np.zeros(len(phase))
        qflat = qtran * (1 - 2 * qin)
        transit[phase <= 0.5 * qflat] = -dep
        in_eg = (phase > 0.5 * qflat) & (phase <= 0.5 * qtran)
        transit[in_eg] = -dep + (
            (dep / (0.5 * (qtran - qflat))) * (phase[in_eg] - 0.5 * qflat)
        )
        _model = transit + params["zpt"].value
        return _model


class TransitModel(Model):
    def __init__(
        self,
        per=None,
        epo=None,
        RpRs=None,
        aRs=None,
        b=None,
        u1=None,
        u2=None,
        zpt=None,
        params=None,
        cap_b=True,
    ):
        if params is None:
            params = Parameters()
            params.add("per", value=per, min=0.9 * per, max=1.1 * per)
            params.add("epo", value=epo, min=epo - 0.1 * per, max=epo + 0.1 * per)
            if cap_b:
                params.add("b", value=b, min=0, max=1)
                params.add("RpRs", value=RpRs, min=0, max=1)
            else:
                params.add("b", value=b, min=0)
                params.add("delta", value=b - RpRs, max=1)
                params.add("RpRs", expr="b - delta")
            params.add("aRs", value=aRs, min=0)
            params.add("u1", value=u1, vary=False)
            params.add("u2", value=u2, vary=False)
            params.add("zpt", value=zpt, min=0)
        self.params = params

    def model(self, params, t):
        bparams = batman.TransitParams()
        bparams.t0 = params["epo"].value
        bparams.per = params["per"].value
        bparams.rp = params["RpRs"].value
        bparams.a = params["aRs"].value
        bparams.inc = np.arccos(params["b"].value / params["aRs"].value) * 180.0 / np.pi
        bparams.ecc = 0.0
        bparams.w = 90.0
        bparams.u = [params["u1"].value, params["u2"].value]
        bparams.limb_dark = "quadratic"
        m = batman.TransitModel(bparams, t)
        _model = m.light_curve(bparams) - 1 + params["zpt"].value
        return _model


class SineModel(Model):
    def __init__(self, per=None, amp=None, phs=None, zpt=None, params=None):
        if params is None:
            params = Parameters()
            params.add("per", value=per, min=0.9 * per, max=1.1 * per)
            params.add("amp", value=amp, min=0)
            params.add("phs", value=phs, min=0, max=2 * np.pi)
            params.add("zpt", value=zpt)
        self.params = params

    def model(self, params, t):
        _model = params["zpt"].value + params["amp"].value * np.sin(
            2 * np.pi * t / params["per"].value + params["phs"].value
        )
        return _model
