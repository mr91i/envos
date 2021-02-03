#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import radmc3dPy.data as rmcd
from dataclasses import dataclass
from typing import Any
from envos import tools, cubicsolver, tsc
import envos.nconst as nc
from envos import global_paths as rc
from envos.log import set_logger

logger = set_logger(__name__)


class ModelBase:
    def read_grid(self, grid):
        for k, v in grid.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.vR = self.vr * np.sin(self.tt) + self.vt * np.cos(self.tt)
        self.vz = self.vr * np.cos(self.tt) - self.vt * np.sin(self.tt)

    def save(self, filename, filepath=None):
        if filepath is None:
            filepath = os.path.join(rc.run_dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        pd.to_pickle(self, filepath)
        logger.info(f"Saved : {filepath}")


######################################################################


@dataclass
class KinematicModel(ModelBase):
    grid: Any
    rho: np.ndarray
    vr: np.ndarray
    vt: np.ndarray
    vp: np.ndarray

    def __post_init__(self):
        self.read_grid(self.grid)
        self.set_cylindrical_velocity()


class CircumstellarModel(ModelBase):
    def __init__(self, grid=None, ppar=None):
#     grid: Any = None
#     rho: np.ndarray = None
#     vr: np.ndarray  = None
#     vt: np.ndarray = None
#     vp: np.ndarray = None
#     filepath: str=None
#     radmc_dir: str=None
#     ppar: "Any" = None
#     ispec: str=None
#     f_dg: float = None
#     tgas_eq_tdust: int = 1

#    def __post_init__(self):
        self.read_grid(grid)

#        logger.info("Constructing a model from radmc3d output")

#        if filepath is not None:
#            read_model(filepath, base=self)
#            return

        self.ppar = ppar
#        self.rmcdata = None
        #self.read_radmc3d_output(radmc_dir, ispec)
        # self.set_values(f_dg, tgas_eq_tdust)

    def set_density(self, density):
        self.rho = density

    def set_velocity(self, vr, vt, vp ):
        self.vr = vr
        self.vt = vt
        self.vp = vp
        self.set_cylindrical_velocity()

    def set_temperature(self, temperature):
        self.Tgas = temperature

    def set_fdg(self, f_dg):
        self.f_dg = f_dg

    def set_molname(self, molname):
        self.molname = molname

    def set_radmcdir(self, radmcdir):
        self.radmcdir = radmcdir

    def add_physical_parameters(self, ppar):
        self.ppar = ppar

    def read_radmc3d_output(self, radmc_dir, ispec):
        cwd = os.getcwd()
        os.chdir(radmc_dir)
        rmcdata = rmcd.radmc3dData()
        rmcdata.readDustDens()
        rmcdata.readDustTemp()
        rmcdata.readGasTemp()
        rmcdata.readGasVel()
        rmcdata.readGasDens(ispec=ispec)
        os.chdir(cwd)
        self.rmcdata = rmcdata

    def set_grid(self):
        self.rc_ax = self.rmcdata.grid.x
        self.tc_ax = self.rmcdata.grid.y
        self.pc_ax = self.rmcdata.grid.z
        self.rr, self.tt, self.pp = np.meshgrid(
            self.rc_ax, self.tc_ax, self.pc_ax, indexing="ij"
        )
        self.R = self.rr * np.sin(self.tt)
        self.z = self.rr * np.cos(self.tt)

    def set_values(self, f_dg, tgas_eq_tdust):
        self.rhodust = self.get_value("rhodust")
        self.rhogas = self.rhodust / f_dg
        self.ndens_mol = self.get_value("ndens_mol")
        self.Tdust = self.get_value("dusttemp")
        if tgas_eq_tdust:
            self.Tgas = self.Tdust
        else:
            self.Tgas = self.get_value("gastemp")
        self.vr = self.get_value("gasvel", index=0)
        self.vt = self.get_value("gasvel", index=1)
        self.vp = self.get_value("gasvel", index=2)
        self.set_cylindrical_velocity()

    def get_value(self, key, index=0):
        val = getattr(self.rmcdata, key)
        if len(val) != 0:
            logger.debug(f"Setting {key}")
            return val[:, :, :, index]
        else:
            logger.info(f"Tried to set {key} but not found.")
            return None


######################################################################
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
        # mu_over_mu0 = 1 - zeta * (1 - self.mu0 ** 2)
        mu_to_mu0 = 1 - zeta * sin0 ** 2
        v0 = np.sqrt(nc.G * Ms / self.rr)
        self.vr = -v0 * np.sqrt(1 + mu_to_mu0)
        self.vt = (
            v0
            * zeta
            * sin0 ** 2
            * self.mu0
            / np.sin(self.tt)
            * np.sqrt(1 + mu_to_mu0)
        )
        self.vp = v0 * sin0 ** 2 / np.sin(self.tt) * np.sqrt(zeta)
        P2 = 1 - 3 / 2 * sin0 ** 2
        rho = -Mdot / (
            4 * np.pi * self.rr ** 2 * self.vr * (1 + 2 * zeta * P2)
        )
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
        vff = np.sqrt(2 * nc.G * self.Ms / self.rr)
        CB = self.CR / 2
        rho_prof = self.Mdot / (4 * np.pi * self.rr ** 2 * vff)
        mu_cav = np.cos(cavangle)
        cavmask = np.array(np.cos(self.tt) <= mu_cav, dtype=float)
        cbmask = np.array(self.rr >= CB, dtype=float)
        self.rho = rho_prof * cbmask * cavmask
        self.vr = -vff * np.sqrt((1 - CB / self.rr).clip(0))
        self.vt = np.zeros_like(self.rr)
        self.vp = vff / np.sqrt(self.rr / CB)
        self.mu0 = self.mu


##################################################


class TerebeyOuterEnvelope(ModelBase):
    def __init__(self, grid, t, cs, Omega, cavangle=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.read_grid(grid)
        self.calc_kinematic_structure(t, cs, Omega, cavangle)
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self, t, cs, Omega, cavangle):
        res = tsc.get_tsc(self.rc_ax, self.tc_ax, t, cs, Omega, mode="read")
        cavmask = np.array(self.tt <= cavangle, dtype=float)
        self.rho = res["rho"][:, :, np.newaxis] * cavmask
        self.vr = res["vr"][:, :, np.newaxis]
        self.vt = res["vt"][:, :, np.newaxis]
        self.vp = res["vp"][:, :, np.newaxis]
        self.Delta = res["Delta"]
        P2 = 1 - 3 / 2 * np.sin(self.tt) ** 2
        r_exp_shu = cs * Omega ** 2 * t ** 3
        self.r_exp = r_exp_shu * 0.4 / (1 + self.Delta * P2)


##################################################


class Disk(ModelBase):
    def calc_kinematic_structure_from_Sigma(self, Sigma, Ms, cs_disk):
        OmegaK = np.sqrt(nc.G * Ms / self.R ** 3)
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
        cs_disk = np.sqrt(nc.kB * Td / (meanmolw * nc.amu))
        self.calc_kinematic_structure_from_Sigma(Sigma, Ms, cs_disk)
        self.set_cylindrical_velocity()

    def get_Sigma(self, Mdisk, Rd, ind):
        Sigma0 = Mdisk / (2 * np.pi * Rd ** 2) / (1 - 2 / np.e)
        power = (self.R / nc.au) ** ind
        exptail = np.exp(-((self.R / Rd) ** (2 + ind)))
        return Sigma0 * power * exptail


##################################################


class ThermalKinematicModel(ModelBase):
    def __init__(
        self,
        filepath=None,
        radmc_dir=None,
        ppar=None,
        ispec=None,
        f_dg=None,
        tgas_eq_tdust=True,
    ):
        logger.info("Constructing a model from radmc3d output")

        if filepath is not None:
            read_model(filepath, base=self)
            return
        self.ppar = ppar
        self.rmcdata = None
        self.read_radmc3d_output(radmc_dir, ispec)
        self.set_grid()
        self.set_values(f_dg, tgas_eq_tdust)

    def add_physical_parameters(self, ppar):
        self.ppar = ppar

    def read_radmc3d_output(self, radmc_dir, ispec):
        cwd = os.getcwd()
        os.chdir(radmc_dir)
        rmcdata = rmcd.radmc3dData()
        rmcdata.readDustDens()
        rmcdata.readDustTemp()
        rmcdata.readGasTemp()
        rmcdata.readGasVel()
        rmcdata.readGasDens(ispec=ispec)
        os.chdir(cwd)
        self.rmcdata = rmcdata

#    def set_kinematic_structure(kinema):

    def set_grid(self):
        self.rc_ax = self.rmcdata.grid.x
        self.tc_ax = self.rmcdata.grid.y
        self.pc_ax = self.rmcdata.grid.z
        self.rr, self.tt, self.pp = np.meshgrid(
            self.rc_ax, self.tc_ax, self.pc_ax, indexing="ij"
        )
        self.R = self.rr * np.sin(self.tt)
        self.z = self.rr * np.cos(self.tt)

    def set_values(self, f_dg, tgas_eq_tdust):
        self.rhodust = self.get_value("rhodust")
        self.rhogas = self.rhodust / f_dg
        self.ndens_mol = self.get_value("ndens_mol")
        self.Tdust = self.get_value("dusttemp")
        if tgas_eq_tdust:
            self.Tgas = self.Tdust
        else:
            self.Tgas = self.get_value("gastemp")
        self.vr = self.get_value("gasvel", index=0)
        self.vt = self.get_value("gasvel", index=1)
        self.vp = self.get_value("gasvel", index=2)
        self.set_cylindrical_velocity()

    def get_value(self, key, index=0):
        val = getattr(self.rmcdata, key)
        if len(val) != 0:
            logger.debug(f"Setting {key}")
            return val[:, :, :, index]
        else:
            logger.info(f"Tried to set {key} but not found.")
            return None


######################################################################
# Functions                                                          #
######################################################################
def read_model(filepath, base=None):
    if ".pkl" in filepath:
        dtype = "pickle"

    if dtype == "pickle":
        cls = pd.read_pickle(filepath)
        if base is None:
            return cls
        else:
            for k, v in cls.__dict__.items():
                setattr(base, k, v)


def save_kmodel_hdf5_spherical(model, filename="flow.vtk", filepath=None):
    if filepath is None:
        filepath = os.path.join(rc.run_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    from evtk.hl import gridToVTK

    gridToVTK(
        filepath,
        model.ri_ax / nc.au,
        model.ti_ax,
        model.pi_ax,
        cellData={
            "den": model.rho,
            "ur": model.vr,
            "uth": model.vt,
            "uph": model.vp,
        },
    )


def save_kmodel_hdf5_certesian(
    model, xi, yi, zi, filename="flow.vtk", filepath=None
):
    if filepath is None:
        filepath = os.path.join(rc.run_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    from evtk.hl import gridToVTK
    from scipy.interpolate import (
        interpn,
    )  # , RectBivariateSpline, RegularGridInterpolator

    xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing="ij")
    xc = tools.make_array_center(xi)
    yc = tools.make_array_center(yi)
    zc = tools.make_array_center(zi)
    xxc, yyc, zzc = np.meshgrid(xc, yc, zc, indexing="ij")
    rr_cert = np.sqrt(xxc ** 2 + yyc ** 2 + zzc ** 2)
    tt_cert = np.arccos(zzc / rr_cert)
    pp_cert = np.arctan2(yyc, xxc)

    def interper(val):
        return interpn(
            (model.rc_ax, model.tc_ax, model.pc_ax),
            val,
            np.stack([rr_cert, tt_cert, pp_cert], axis=-1),
            bounds_error=False,
            fill_value=np.nan,
        )

    den_cert = interper(model.rho)
    vr_cert = interper(model.vr)
    vt_cert = interper(model.vt)
    vp_cert = interper(model.vp)
    uux = (
        vr_cert * np.sin(tt_cert) * np.cos(pp_cert)
        + vt_cert * np.cos(tt_cert) * np.cos(pp_cert)
        - vp_cert * np.sin(pp_cert)
    )
    uuy = (
        vr_cert * np.sin(tt_cert) * np.sin(pp_cert)
        + vt_cert * np.cos(tt_cert) * np.sin(pp_cert)
        + vp_cert * np.cos(pp_cert)
    )
    uuz = vr_cert * np.cos(tt_cert) - vt_cert * np.sin(tt_cert)
    os.makedir(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    gridToVTK(
        filepath,
        xi / nc.au,
        yi / nc.au,
        zi / nc.au,
        cellData={"den": den_cert, "ux": uux, "uy": uuy, "uz": uuz},
    )
