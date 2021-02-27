import numpy as np
import envos.nconst as nc
from envos.log import set_logger

logger = set_logger(__name__)

def calc_dependent_params(
        T: float = None,
        CR_au: float = None,
        Ms_Msun: float = None,
        t_yr: float = None,
        Omega: float = None,
        maxj: float = None,
        Mdot_smpy: float = None,
        meanmolw: float = None,
    ):

    m0 = 0.975
    if T is not None:
        cs = np.sqrt(nc.kB * T / (meanmolw * nc.amu))
        Mdot = cs ** 3 * m0 / nc.G
    elif Mdot_smpy is not None:
        Mdot = Mdot_smpy * nc.smpy
        cs = (Mdot * nc.G / m0) ** (1 / 3)
        T = cs ** 2 * meanmolw * nc.amu / nc.kB

    if Ms_Msun is not None:
        Ms = Ms_Msun * nc.Msun
        t = Ms / Mdot
    elif t_yr is not None:
        t = t_yr * nc.yr
        M = Mdot * t

    if CR_au is not None:
        CR = CR_au * nc.au
        maxj = np.sqrt(CR * nc.G * Ms)
        Omega = maxj / (0.5 * cs * m0 * t) ** 2
    elif maxj is not None:
        CR = maxj ** 2 / (nc.G * Ms)
        Omega = maxj / (0.5 * cs * m0 * t) ** 2
    elif Omega is not None:
        maxj = (0.5 * cs * m0 * t) ** 2 * Omega
        CR = maxj ** 2 / (nc.G * Ms)

    return {"T":T, "cs":cs, "Mdot":Mdot, "Ms":Ms, "t":t, "CR": CR, "maxj": maxj, "Omega": Omega}



class PhysicalParameters:
    def __init__(
        self,
        *,
        T: float = None,
        CR_au: float = None,
        Ms_Msun: float = None,
        t_yr: float = None,
        Omega: float = None,
        maxj: float = None,
        Mdot_smpy: float = None,
        meanmolw: float = 2.3,
        cavangle_deg: float = 0,
    ):
        self.meanmolw = meanmolw

        self._calc_T_cs_Mdot(self.meanmolw, T, Mdot_smpy)
        self._calc_Ms_t(self.Mdot, Ms_Msun, t_yr)
        self._calc_CR_maxj_Omega(self.cs, self.Ms, self.t, CR_au, maxj, Omega)

        self.cavangle = np.radians(cavangle_deg)
        self.r_inlim_tsc = self.cs * self.Omega ** 2 * self.t ** 3
        self.log()

    def _calc_T_cs_Mdot(self, meanmolw, T=None, Mdot_smpy=None):
        m0 = 0.975
        if T is not None:
            cs = np.sqrt(nc.kB * T / (meanmolw * nc.amu))
            Mdot = cs ** 3 * m0 / nc.G
        elif Mdot_smpy is not None:
            Mdot = Mdot_smpy * nc.smpy
            cs = (Mdot * nc.G / m0) ** (1 / 3)
            T = cs ** 2 * meanmolw * nc.amu / nc.kB
        self.T = T
        self.cs = cs
        self.Mdot = Mdot

    def _calc_Ms_t(self, Mdot, Ms_Msun=None, t_yr=None):
        if Ms_Msun is not None:
            Ms = Ms_Msun * nc.Msun
            t = Ms / Mdot
        elif t_yr is not None:
            t = t_yr * nc.yr
            M = Mdot * t
        self.Ms = Ms
        self.t = t

    def _calc_CR_maxj_Omega(
        self, cs, Ms, t, CR_au=None, maxj=None, Omega=None
    ):
        m0 = 0.975
        if CR_au is not None:
            CR = CR_au * nc.au
            maxj = np.sqrt(CR * nc.G * Ms)
            Omega = maxj / (0.5 * cs * m0 * t) ** 2
        elif maxj is not None:
            CR = maxj ** 2 / (nc.G * Ms)
            Omega = maxj / (0.5 * cs * m0 * t) ** 2
        elif Omega is not None:
            maxj = (0.5 * cs * m0 * t) ** 2 * Omega
            CR = maxj ** 2 / (nc.G * Ms)
        self.CR = CR
        self.maxj = maxj
        self.Omega = Omega

    def log(self):
        logger.info("Model Parameters:")
        self._logp("Tenv", "K", self.T)
        self._logp("cs", "km/s", self.cs, nc.kms)
        self._logp("t", "yr", self.t, nc.yr)
        self._logp("Ms", "Msun", self.Ms, nc.Msun)
        self._logp("Omega", "s^-1", self.Omega)
        self._logp("Mdot", "Msun/yr", self.Mdot, nc.Msun / nc.yr)
        self._logp("maxj", "au*km/s", self.maxj, nc.kms * nc.au)
        self._logp("maxj", "pc*km/s", self.maxj, nc.kms * nc.pc)
        self._logp("CR", "au", self.CR, nc.au)
        self._logp("CB", "au", self.CR / 2, nc.au)
        self._logp("meanmolw", "", self.meanmolw)
        self._logp("cavangle", "deg", np.rad2deg(self.cavangle))
        self._logp("Omega*t", "", self.Omega * self.t)
        self._logp(
            "rinlim_tsc", "au", self.cs * self.Omega ** 2 * self.t ** 3, nc.au
        )
        self._logp("rinlim_tsc", "cs*t", self.Omega ** 2 * self.t ** 2)
        logger.info("")

    @staticmethod
    def _logp(name, unit, value, unitval=1):
        logger.info(
            "    "
            + name.ljust(12)
            + f"= {value/unitval:10.2g} "
            + unit.ljust(8)
        )
