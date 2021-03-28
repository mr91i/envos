#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Any
from envos import tools, cubicsolver, tsc
from .nconst import G, kB, amu, au
from .gpath import run_dir
from .log import set_logger
from . import tools

logger = set_logger(__name__)


class ModelBase:
    def read_grid(self, grid):
        for k, v in grid.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.vR = self.vr * np.sin(self.tt) + self.vt * np.cos(self.tt)
        self.vz = self.vr * np.cos(self.tt) - self.vt * np.sin(self.tt)

    def save_pickle(self, filename, filepath=None):
        if filepath is None:
            filepath = os.path.join(run_dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        pd.to_pickle(self, filepath)
        logger.info(f"Saved : {filepath}")

    def read_pickle(self, filename=None, filepath=None):
        if (filename is not None) and (filepath is None):
            filepath = os.path.join(run_dir, filename)
        tools.setattr_from_pickle(self, filepath)

@dataclass
class CircumstellarModel(ModelBase):
    grid: Any = None
    rc_ax: np.ndarray = None
    tc_ax: np.ndarray = None
    pc_ax: np.ndarray = None
    rr: np.ndarray = None
    tt: np.ndarray = None
    pp: np.ndarray = None
    R: np.ndarray = None
    z: np.ndarray = None
    ppar: Any = None
    rhogas: np.ndarray = None
    rhodust: np.ndarray = None
    vr: np.ndarray = None
    vt: np.ndarray = None
    vp: np.ndarray = None
    Tgas: np.ndarray = None
    Tdust: np.ndarray = None
    f_dg: float = None
    molname: str = None
    radmcdir: str = None
    filepath: str = None

    def __post_init__(self):
        logger.info("Constructing a model of circumstellar material")
        if self.filepath is not None:
            self.read_pickle(filepath=self.filepath)

        if self.grid is not None:
            self.set_grid(self.grid)

        if self.ppar is not None:
            self.set_physical_parameters(self.ppar)

    def __str__(self):
        return tools.dataclass_str(self)

    def set_physical_parameters(self, ppar):
        self.ppar = ppar

    def set_grid(self, grid):
        self.read_grid(grid)

    def set_gas_density(self, density):
        self.rhogas = density

    def set_gas_velocity(self, vr, vt, vp):
        self.vr = vr
        self.vt = vt
        self.vp = vp
        self.set_cylindrical_velocity()

    def set_gas_temperature(self, temperature):
        self.Tgas = temperature
        self.Tdust = temperature

    def set_fdg(self, f_dg):
        self.f_dg = f_dg

        if hasattr(self, "rhogas"):
            self.rhodust = self.rhogas * f_dg

    def set_molname(self, molname):
        self.molname = molname

    def set_radmcdir(self, radmcdir):
        self.radmcdir = radmcdir


class CassenMoosmanInnerEnvelope(ModelBase):
    def __init__(self, grid, Mdot, CR, Ms, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.read_grid(grid)
        self.calc_kinematic_structure(Mdot, CR, Ms, cavangle)
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self, Mdot, CR, Ms, cavangle):
        csol = np.frompyfunc(self._sol_with_cubic, 2, 1)
        zeta = CR / self.rr
        self.mu0 = csol(np.cos(self.tt), zeta).astype(float)
        sin0 = np.sqrt(1 - self.mu0 ** 2)
        sin = np.sin(self.tt)
        mu_to_mu0 = 1 - zeta * sin0 ** 2
        v0 = np.sqrt(G * Ms / self.rr)
        self.vr = -v0 * np.sqrt(1 + mu_to_mu0)
        self.vt = v0 * zeta * sin0 ** 2 * \
            self.mu0 / sin * np.sqrt(1 + mu_to_mu0)
        self.vp = v0 * sin0 ** 2 / np.sin(self.tt) * np.sqrt(zeta)
        P2 = 1 - 3 / 2 * sin0 ** 2
        rho = -Mdot / (4 * np.pi * self.rr ** 2 *
                       self.vr * (1 + 2 * zeta * P2))
        cavmask = np.array(self.mu0 <= np.cos(cavangle), dtype=float)
        self.rho = rho * cavmask

    @staticmethod
    def _sol_with_cubic(m, zeta):
        allsols = np.round(cubicsolver.solve(zeta, 0, 1 - zeta, -m).real, 8)
        sols = [sol for sol in allsols if 0 <= sol <= 1]
        return sols[0] if len(sols) != 0 else np.nan


class SimpleBallisticInnerEnvelope(ModelBase):
    def __init__(self, grid, Mdot, CR, M, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.read_grid(grid)
        self.calc_kinematic_structure(Mdot, CR, M, cavangle)
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self, Mdot, CR, M, cavangle):
        vff = np.sqrt(2 * G * M / self.rr)
        CB = CR / 2
        rho_prof = Mdot / (4 * np.pi * self.rr ** 2 * vff)
        mu_cav = np.cos(cavangle)
        cavmask = np.array(np.cos(self.tt) <= mu_cav, dtype=float)
        cbmask = np.array(self.rr >= CB, dtype=float)
        self.rho = rho_prof * cbmask * cavmask
        self.vr = -vff * np.sqrt((1 - CB / self.rr).clip(0))
        self.vt = np.zeros_like(self.rr)
        self.vp = vff / np.sqrt(self.rr / CB)
        self.mu0 = np.cos(self.tt)


class TerebeyOuterEnvelope(ModelBase):
    def __init__(self, grid, t, cs, Omega, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.read_grid(grid)
        self.calc_kinematic_structure(t, cs, Omega, cavangle)
        self.set_cylindrical_velocity()
        self.rin_lim = 0

    def calc_kinematic_structure(self, t, cs, Omega, cavangle):
        res = tsc.get_tsc(self.rc_ax, self.tc_ax, t, cs, Omega, mode="read")
        cavmask = np.array(self.tt >= cavangle, dtype=float)
        self.rho = res["rho"][:, :, np.newaxis] * cavmask
        self.vr = res["vr"][:, :, np.newaxis]
        self.vt = res["vt"][:, :, np.newaxis]
        self.vp = res["vp"][:, :, np.newaxis]
        self.Delta = res["Delta"]
        P2 = 1 - 3 / 2 * np.sin(self.tt) ** 2
        r_exp_shu = cs * Omega ** 2 * t ** 3
        self.r_exp = r_exp_shu * 0.4 / (1 + self.Delta * P2)
        self.rin_lim = cs * Omega ** 2 * t ** 3

class Disk(ModelBase):
    def calc_kinematic_structure_from_Sigma(self, Sigma, Ms, cs_disk):
        OmegaK = np.sqrt(G * Ms / self.R ** 3)
        H = cs_disk / OmegaK
        rho0 = Sigma / (np.sqrt(2 * np.pi) * H)
        self.rho = rho0 * np.exp(-0.5 * (self.z / H) ** 2)
        self.vr = np.zeros_like(self.rho)
        self.vt = np.zeros_like(self.rho)
        self.vp = OmegaK * self.R


class ExptailDisk(Disk):
    def __init__(
        self, grid, Ms, Rd, Td=30, fracMd=0.1, meanmolw=2.3, index=-1
    ):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.read_grid(grid)
        Mdisk = fracMd * Ms
        Sigma = self.get_Sigma(Mdisk, Rd, index)
        cs_disk = np.sqrt(kB * Td / (meanmolw * amu))
        self.calc_kinematic_structure_from_Sigma(Sigma, Ms, cs_disk)
        self.set_cylindrical_velocity()

    def get_Sigma(self, Mdisk, Rd, ind):
        Sigma0 = Mdisk / (2 * np.pi * Rd ** 2) / (1 - 2 / np.e)
        power = (self.R / au) ** ind
        exptail = np.exp(-((self.R / Rd) ** (2 + ind)))
        return Sigma0 * power * exptail
