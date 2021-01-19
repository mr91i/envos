#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import radmc3dPy.data as rmcd
from dataclasses import dataclass
from envos import grid, tools, cubicsolver, tsc, log, config
from envos import nconst as nc

logger = log.set_logger(__name__, ini=True)


class ModelBase:
    def read_grid(self, grid):
        for k, v in grid.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.vR = self.vr * np.sin(self.tt) + self.vt * np.cos(self.tt)
        self.vz = self.vr * np.cos(self.tt) - self.vt * np.sin(self.tt)

    def save(self, filename, filepath=None):
        if filepath is None:
            filepath = os.path.join(config.dp_run, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        pd.to_pickle(self, filepath)
        logger.info(f"Saved : {filepath}")


# class KinematicModel(ModelBase):
# class CircumstellarModel(ModelBase):
#    def __init__(self, filepath=None, grid=None, ppar=None):
# class CircumstellarModel(ModelBase):
#class CircumstellarModel(ModelBase):
class ModelBuilder:
    # def __init__(self, filepath=None, grid=None, ppar=None):
    def __init__(self, config):
        self.grid = config.grid
        self.ppar = config.ppar
        self.inenv = config.model.inenv
        self.outenv = config.model.outenv
        self.disk = config.model.disk

    #        if filepath is not None:
    #            read_model(filepath, base=self)
    def calc_kinematic_structure(self, inenv="CM", disk=None, outenv=None):
        ### Set grid
        if (self.grid is None) and (grid is None):
            raise Exception("grid is not set.")
        grid = grid or self.grid

        if (self.ppar is None) and (ppar is None):
            raise Exception("ppar is not set.")
        ppar = ppar or self.ppar

        ### Set models
        # Set inenv
        if not isinstance(inenv, str):
            self.inenv = inenv
        elif inenv == "CM":
            self.inenv = CassenMoosmanInnerEnvelope(grid, ppar.Mdot, ppar.CR, ppar.M)
        elif inenv == "Simple":
            self.inenv = SimpleBalisticInnerEnvelope(grid, ppar.Mdot, ppar.CR, ppar.M)
        else:
            raise Exception("No Envelope.")

        # Set outenv
        # if self.grid.rc_ax[-1] > ppar.r_inlim_tsc:
        if hasattr(outenv, "rho"):
            self.outenv = outenv
        elif outenv == "TSC":
            self.outenv = TerebeyOuterEnvelope(grid, ppar.t, ppar.cs, ppar.Omega)
        elif outenv is not None:
            raise Exception("Unknown outenv type")

        # Set disk
        if hasattr(self, "rho"):
            self.disk = disk
        elif disk == "exptail":
            self.disk = ExptailDisk(
                grid,
                ppar.Ms,
                ppar.CR,
                Td=30,
                fracMd=0.1,
                meanmolw=ppar.meanmolw,
                index=-1.0,
            )
        elif disk is not None:
            raise Exception("Unknown disk type")

        ### Make kmodel
        conds = [np.ones_like(self.rr, dtype=bool)]
        regs = [self.inenv]

        if self.outenv is not None:
            cond1 = self.outenv.rho > self.inenv.rho
            cond2 = grid.rr > self.outenv.r_exp
            conds.append(cond1 & cond2)
            regs.append(self.outenv)

        if self.disk is not None:
            conds.append(self.disk.rho > self.inenv.rho)
            regs.append(self.disk)

        rho = np.select(conds[::-1], [r.rho for r in regs[::-1]])
        vr = np.select(conds[::-1], [r.vr for r in regs[::-1]])
        vt = np.select(conds[::-1], [r.vt for r in regs[::-1]])
        vp = np.select(conds[::-1], [r.vp for r in regs[::-1]])

        self.kmodel = KinematicModel(rho, vr, vt, vp)

    def get_kinematic_structure(self):
        return self.kmodel

    def calc_thermal_structure(self):
        rc = RadmcController(
            nphot=radmc.nphot,
            n_thread=radmc.n_thread,
            scattering_mode_max=radmc.scattering_mode_max,
            mc_scat_maxtauabs=radmc.mc_scat_maxtauabs,
            f_dg=radmc.f_dg,
            opac=radmc.opac,
            Lstar_Lsun=radmc.Lstar_Lsun,
            mfrac_H2=radmc.mfrac_H2,
            T_const=radmc.T_const,
            Rstar_Rsun=radmc.Rstar_Rsun,
            temp_mode=radmc.temp_mode,
            molname=radmc.molname,
            molabun=radmc.molabun,
            iline=radmc.iline,
            mol_rlim=radmc.mol_rlim,
            run_dir=radmc.run_dir,
            radmc_dir=radmc.radmc_dir,
            storage_dir=radmc.storage_dir,
        )
        rc.set_model(self.kmodel)
        rc.set_radmc_input()
        rc.exe_mctherm()
        self.tkmodel = rc.get_model()
        self.tkmodel.save("tkmodel.pkl")

    def read_model(self):
        self.tkmodel = ThermalKinematicMode("run/tkmodel.pkl")

    def get_model(self):
        return self.tkmodel


####################################################################################################
@dataclass
class KinematicModel(ModelBase):
    rho: np.ndarray
    vr: np.ndarray
    vt: np.ndarray
    vp: np.ndarray

    def __post_init__(self):
        self.set_cylindrical_velocity()


####################################################################################################
class CassenMoosmanInnerEnvelope(ModelBase):
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
        csol = np.frompyfunc(self._sol_with_cubic, 2, 1)
        zeta = self.CR / self.rr
        self.mu0 = csol(np.cos(self.tt), zeta).astype(np.float64)
        sin0 = np.sqrt(1 - self.mu0 ** 2)
        # mu_over_mu0 = 1 - zeta * (1 - self.mu0 ** 2)
        mu_over_mu0 = 1 - zeta * sin0 ** 2
        v0 = np.sqrt(nc.G * self.Ms / self.rr)
        self.vr = -v0 * np.sqrt(1 + mu_over_mu0)
        self.vt = (
            v0
            * zeta
            * sin0 ** 2
            * self.mu0
            / np.sin(self.tt)
            * np.sqrt(1 + mu_over_mu0)
        )
        self.vp = v0 * sin0 ** 2 / np.sin(self.tt) * np.sqrt(zeta)
        P2 = 1 - 3 / 2 * sin0 ** 2
        rho = -self.Mdot / (4 * np.pi * self.rr ** 2 * self.vr * (1 + 2 * zeta * P2))
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
        vff = np.sqrt(2 * nc.G * self.Mstar / self.rr)
        CB = self.CR / 2
        rho_prof = self.Mdot / (4 * np.pi * self.rr ** 2 * vff)
        mu_cav = np.cos(cavangle)
        cavmask = np.array(mu0 <= mu_cav, dtype=float)
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
    def calc_kinematic_structure_from_Sigma(self, Sigma):
        OmegaK = np.sqrt(nc.G * self.Ms / self.R ** 3)
        H = self.cs / OmegaK
        self.rho = Sigma / (np.sqrt(2 * np.pi) * H) * np.exp(-0.5 * (self.z / H) ** 2)
        self.vr = np.zeros_like(self.rho)
        self.vt = np.zeros_like(self.rho)
        self.vp = OmegaK * self.R


class CMviscDisk(Disk):
    def __init__(self, grid, Ms, CR, Td, fracMd, meanmolw):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Ms = Ms
        self.CR = CR
        self.Mdisk = frac_Md * self.Mstar
        self.cs = np.sqrt(nc.kB * Td / (meanmolw * nc.amu))
        self.read_grid(grid)
        Sigma = self.get_Sigma()
        self.calc_kinematic_structure_from_Sigma(Sigma)

    def get_Sigma(self):
        logger.error("Still Constructing!")
        exit()
        u = self.R / self.CR
        P = (
            (3 * 0.01 * self.cs ** 2)
            / np.sqrt(nc.G * self.Ms / self.CR ** 3)
            * self.Mstar ** 6
            / self.Mfin ** 5
            / self.Mdot
            / self.CR ** 2
        )
        P_rd2 = (
            3
            * 0.1
            * self.Tdisk
            / self.Tenv
            * np.sqrt(self.Mfin / self.Mstar)
            * np.sqrt(nc.G * self.Mfin / self.r_CR)
            / self.cs_disk
        )
        a3 = 0.2757347731  # = (2^10/3^11)^(0.25)
        ue = np.sqrt(3 * P) * (1 - 56 / 51 * a3 / P ** 0.25)
        y = np.where(
            u < 1,
            2 * np.sqrt(1 - u.clip(max=1))
            + 4 / 3 / np.sqrt(u) * (1 - (1 + 0.5 * u) * np.sqrt(1 - u.clip(max=1))),
            4 / 3 / np.sqrt(u),
        )
        y = np.where(u <= ue, y - 4 / 3 / np.sqrt(ue.clip(1e-30)), 0)
        Sigma = 0.5 / P * y * self.Ms / (np.pi * self.CR ** 2)
        return Sigma


class ExptailDisk(Disk):
    def __init__(self, grid, Ms, Rd, Td=30, fracMd=0.1, meanmolw=2.3, index=-1):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Ms = Ms
        self.cs = np.sqrt(nc.kB * Td / (meanmolw * nc.amu))
        self.read_grid(grid)
        Mdisk = frac_Md * self.Mstar
        Sigma = self.get_Sigma(Mdisk, Rd, index)
        self.calc_kinematic_structure_from_Sigma(Sigma)
        self.set_cylindrical_velocity()

    def get_Sigma(self, Mdisk, Rd, ind):
        Sigma0 = Mdisk / (2 * np.pi * Rd ** 2) / (1 - 2 / np.e)
        power = (self.R / nc.au) ** ind
        exptail = np.exp(-((self.R / self.Rd) ** (2 + ind)))
        return Sigma0 * power * exptail


##################################################


class ThermalKinematicModel(ModelBase):
    def __init__(
        self, filepath=None, dpath_radmc=None, ispec=None, f_dg=None, tgas_eq_tdust=True
    ):
        if filepath is not None:
            read_model(filepath, base=self)
            return

        self.rmcdata = None
        self.read_radmc3d_output()
        self.set_grid()
        self.set_values()

        ## getting data

    def set_grid():
        self.rc_ax = rmcdata.grid.x
        self.tc_ax = rmcdata.grid.y
        self.pc_ax = rmcdata.grid.z
        self.rr, self.tt, self.pp = np.meshgrid(
            self.rc_ax, self.tc_ax, self.pc_ax, indexing="ij"
        )
        self.R = self.rr * np.sin(self.tt)
        self.z = self.rr * np.cos(self.tt)

    def set_values(self):
        self.rhodust = self.get_value("rhodust")
        self.Tdust = self.get_value("dusttemp")
        self.Tgas = self.get_value("gastemp") if not tgas_eq_tdust else self.dtemp
        self.vr = self.get_value("gasvel", index=0)
        self.vt = self.get_value("gasvel", index=1)
        self.vp = self.get_value("gasvel", index=2)
        self.ndens_mol = self.get_value("ndens_mol")
        self.rhogas = self.rhodust / f_dg
        self.set_cylindrical_velocity()

    def read_radmc3d_output(self):
        cwd = os.getcwd()
        os.chdir(dpath_radmc)
        rmcdata = rmcd.radmc3dData()
        rmcdata.readDustDens()
        rmcdata.readDustTemp()
        rmcdata.readGasTemp()
        rmcdata.readGasVel()
        rmcdata.readGasDens(ispec=ispec)
        os.chdir(cwd)
        self.rmcdata = rmcdata

    def get_value(self, key, index=0):
        val = getattr(self.rmcdata, key)
        if len(val) != 0:
            logger.debug(f"Setting {key}")
            return val[:, :, :, index]
        else:
            logger.info(f"Tried to set {key} but not found.")
            return None


####################################################################################################
## Functions                                                                                       #
####################################################################################################
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
        filepath = os.path.join(config.dp_run, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    from evtk.hl import gridToVTK

    gridToVTK(
        filepath,
        model.ri_ax / nc.au,
        model.ti_ax,
        model.pi_ax,
        cellData={"den": model.rho, "ur": model.vr, "uth": model.vt, "uph": model.vp},
    )


def save_kmodel_hdf5_certesian(model, xi, yi, zi, filename="flow.vtk", filepath=None):
    if filepath is None:
        filepath = os.path.join(config.dp_run, filename)
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
