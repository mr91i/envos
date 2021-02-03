#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from envos import tools
import envos.nconst as nc
from envos import global_paths as rc
from envos.models import CircumstellarModel
from envos.log import set_logger

logger = set_logger(__name__)

"""
 Classes
"""

@dataclass
class RadmcController:
    nphot: int = 1e6
    n_thread: int = 1
    scattering_mode_max: int = 0
    mc_scat_maxtauabs: float = 5.0
    tgas_eq_tdust: bool = True
    f_dg: float = 0.01
    opac: str = "silicate"
    Lstar_Lsun: float = 1.0
    mfrac_H2: float = 0.74
    T_const: float = 10.0
    Rstar_Rsun: float = 1.0
    temp_mode: str = "mctherm"
    molname: str = None
    molabun: float = None
    iline: int = None
    mol_rlim: float = 1000.0
    run_dir: str = None
    radmc_dir: str = None
    storage_dir: str = None

    def __post_init__(self):
        self.run_dir = self.run_dir or rc.run_dir  # global?
        self.radmc_dir = self.radmc_dir or rc.radmc_dir  # local
        self.storage_dir = self.storage_dir or rc.storage_dir  # local
        rc.make_dirs(
            run=self.run_dir, radmc=self.radmc_dir, storage=self.storage_dir
        )

        if self.iline is not None:
            self.calc_line = True

    def set_model(self, model):
        if isinstance(model, str) and os.path.isfile(model):
            self.model = pd.read_pickle(self.model_pkl)
        if model is not None:
            self.model = model
        else:
            raise Exception("Unknown model.")

    def set_userdefined_T_func(self, func):
        """
        Set a user-defined function of temperature.
        The function takes an instance of this class (i.e. self)
        as an argument.
        """
        self.T_func = func

    #def set_radmc_input(self):
    def set_mctherm_inpfiles(self):
        logger.info("Setting input files used in radmc3d")
        md = self.model

        rc, tc, pc = md.rc_ax, md.tc_ax, md.pc_ax
        rr, tt, pp = np.meshgrid(rc, tc, pc, indexing="ij")
        self.ntot = rr.size
        ri, ti, pi = md.ri_ax, md.ti_ax, md.pi_ax

        #rhog, vr, vt, vp = md.rho, md.vr, md.vt, md.vp
        rhog = md.rho
        self._rhog = rhog
        if np.max(rhog) == 0:
            raise Exception("Zero density")
        rhod = rhog * self.f_dg

        lam1, lam2, lam3, lam4 = 0.1, 7, 25, 1e4
        n12, n23, n34 = 20, 100, 30
        lam12 = np.geomspace(lam1, lam2, n12, endpoint=False)
        lam23 = np.geomspace(lam2, lam3, n23, endpoint=False)
        lam34 = np.geomspace(lam3, lam4, n34, endpoint=True)
        lam = np.concatenate([lam12, lam23, lam34])
        nlam = lam.size

        Tstar = self.Rstar_Rsun ** (-0.5) * self.Lstar_Lsun ** 0.25 * nc.Tsun

        """
        Setting Radmc Parameters
        """
        # set grid
        self._save_input_file(
            "amr_grid.inp",
            "1",
            "0",
            "100",
            "0",
            "1 1 0",
            f"{rc.size:d} {tc.size:d} {pc.size:d}",
            *ri,
            *ti,
            *pi,
        )

        # set wavelength
        self._save_input_file("wavelength_micron.inp", f"{nlam:d}", *lam)

        # set a star
        self._save_input_file(
            "stars.inp",
            "2",
            f"1 {nlam}",
            f"{self.Rstar_Rsun*nc.Rsun:13.8e} 0 0 0 0",
            *lam,
            f"{-Tstar:13.8e}",
        )
        # set dust density
        self._save_input_file(
            "dust_density.inp", "1", f"{self.ntot:d}", "1", *rhod.ravel(order="F")
        )

        # set_dust_opacity
        opacs = [self.opac]
        text_lines = ["2", f"{len(opacs)}", "==========="]
        for op in opacs:
            self._copy_from_storage(f"dustkappa_{op}.inp")
            text_lines += ["1", "0", f"{op}", "----------"]
        self._save_input_file("dustopac.inp", *text_lines)

        # set_input
        param_dict = {
            "nphot": int(self.nphot),
            "scattering_mode_max": self.scattering_mode_max,
            "iranfreqmode": 1,
            "mc_scat_maxtauabs": self.mc_scat_maxtauabs,
            "tgas_eq_tdust": int(self.tgas_eq_tdust),
        }
        self._save_input_file(
            "radmc3d.inp", *[f"{k} = {v}" for k, v in param_dict.items()]
        )

        if self.temp_mode == "mctherm":
            # remove gas_temperature.inp and dust_temperature.inp
            remove_file("gas_temperature.inp")
            remove_file("dust_temperature.dat")

        elif self.temp_mode == "const":
            self._set_constant_temperature(self.T_const)

        elif self.temp_mode == "user":
            self._set_userdefined_temperature(self.T_func)
        else:
            raise Exception(f"Unknown temp_mode: {self.temp_mode}")

    def set_lineobs_inpfiles(self):
        n_mol = self._rhog / (2 * nc.amu / self.mfrac_H2) * self.molabun
        vr, vt, vp = self.model.vr, self.model.vt, self.model.vp
        # n_mol = np.where(rr < self.mol_rlim, n_mol, 0)

        # set molcular number density
        self._save_input_file(
            f"numberdens_{self.molname}.inp",
            "1",
            f"{self.ntot:d}",
            *n_mol.ravel(order="F"),
        )

        # set_mol_lines
        self._copy_from_storage(f"molecule_{self.molname}.inp")
        self._save_input_file(
            "lines.inp",
            "2",
            "1",
            f"{self.molname}  leiden 0 0 0",
        )

        # set_gas_velocity
        _zipped_vel = zip(
            vr.ravel(order="F"),
            vt.ravel(order="F"),
            vp.ravel(order="F"),
        )
        self._save_input_file(
            "gas_velocity.inp", "1", f"{self.ntot:d}", *_zipped_vel
        )

    def _save_input_file(self, filename, *text_lines):
        filepath = os.path.join(self.radmc_dir, filename)
        mapped_lines = map(self._strfunc, text_lines)
        text = "\n".join(mapped_lines)

        with open(filepath, "w+") as f:
            f.write(text)
        fname = os.path.basename(f.name)
        dname = os.path.dirname(f.name)
        logger.info(f"Saved {fname} in {dname}")

    def _strfunc(self, line):
        if isinstance(line, str):
            return line
        if isinstance(line, int):
            return str(line)
        elif isinstance(line, float):
            return f"{line:13.8e}"
        elif isinstance(line, (tuple, list, np.ndarray)):
            if np.ndim(line) == 1:
                return " ".join([self._strfunc(var) for var in line])
        raise Exception("Unknown line format: ", line)

    def _copy_from_storage(self, filename):
        src = os.path.join(self.storage_dir, filename)
        dst = os.path.join(self.radmc_dir, filename)
        tools.filecopy(src, dst)

    def _set_constant_temperature(self, T_const=None, vlocal_fwhm=None):
        if T_const is not None:
            v_fwhm = np.sqrt(T_const * (16 * np.log(2) * nc.kB) / self.mol_mass)
        elif vlocal_fwhm is not None:
            T_const = m_mol * vlocal_fwhm**2 /(16 * np.log(2) * nc.kB )
        logger.info(f"Constant temperature is  {T_const}")
        logger.info(f"v_FWHM_kms is  {vlocal_fwhm/nc.kms}")
        T = np.full_like(self.rr, T_const)
        self.set_temperature(T, len(T.ravel()))

    def _set_userdefined_temperature(self):
        T = self.tfunc(self)
        T = T.clip(min=0.1, max=10000)
        self._set_temperature(T, len(T.ravel()))

    def _set_temperature(self, temp, ntot):
        """
        Set gas & dust temperature
        """
        self._save_input_file(
            "gas_temperature.inp",
            "1",
            f"{ntot:d}",
            *temp.ravel(order="F"),
        )
        self._save_input_file(
            "dust_temperature.inp",
            "1",
            f"{ntot:d}",
            "1",
            *temp.ravel(order="F"),
        )

    def run_mctherm(self):
        logger.info("Executing RADMC3D with mctherm mode")

        if not os.path.isdir(self.radmc_dir):
            msg = "radmc working directory not found: {self.radmc_dir}"
            raise FileNotFoundError(msg)

        tools.shell(
            f"radmc3d mctherm setthreads {self.n_thread}",
            cwd=self.radmc_dir,
            error_keyword="ERROR",
            log_prefix="    ",
        )

    def get_model(self):
        model = self.model
        model.set_radmcdir(self.radmc_dir)
        #model = CircumstellarModel(radmc_dir=self.radmc_dir)
        model.set_fdg(self.f_dg)
        model.set_molname(self.molname)
        return model

        #return ThermalKinematicModel(
        #    radmc_dir=self.radmc_dir, ispec=self.molname, f_dg=self.f_dg
        #)


"""
 Functions
"""


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed {file_path}")



def del_mctherm_files(dpath):
    remove_file(f"{dpath}/radmc3d.out")
