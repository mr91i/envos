import numpy as np
from typing import Callable, Any
from dataclasses import dataclass
from envos.grid import Grid
from envos.physical_params import PhysicalParameters
from envos.log import set_logger


@dataclass
class Config:
    def set_grid(
        self,
        ri_ax=None,
        ti_ax=None,
        pi_ax=None,
        *,
        rau_lim=None,
        theta_lim=(0, np.pi / 2),
        phi_lim=(0, 2 * np.pi),
        nr=None,
        ntheta=None,
        nphi=1,
        dr_to_r=None,
        aspect_ratio=1.0,
        logr=True
    ):

        self.grid = Grid(
            ri_ax=ri_ax,
            ti_ax=ti_ax,
            pi_ax=pi_ax,
            rau_lim=rau_lim,
            theta_lim=theta_lim,
            phi_lim=phi_lim,
            nr=nr,
            ntheta=ntheta,
            nphi=nphi,
            dr_to_r=dr_to_r,
            aspect_ratio=aspect_ratio,
            logr=logr,
        )

    def set_physical_parameters(
        self,
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

        self.ppar = PhysicalParameters(
            T=T,
            CR_au=CR_au,
            Ms_Msun=Ms_Msun,
            t_yr=t_yr,
            Omega=Omega,
            maxj=maxj,
            Mdot_smpy=Mdot_smpy,
            meanmolw=meanmolw,
            cavangle_deg=cavangle_deg,
        )

    def set_model_input(
        self,
        inenv="CM",
        outenv=None,
        disk=None,
        rot_ccw: bool = False,
        usr_density_func: Callable = None,
    ):
        self.model = ModelConfig(
            inenv=inenv,
            outenv=outenv,
            disk=disk,
            rot_ccw=rot_ccw,
            usr_density_func=usr_density_func,
        )

    def set_radmc_input(
        self,
        nphot: int = 1e6,
        n_thread: int = 1,
        scattering_mode_max: int = 0,
        mc_scat_maxtauabs: float = 5.0,
        f_dg: float = 0.01,
        opac: str = "silicate",
        Lstar_Lsun: float = 1.0,
        mfrac_H2: float = 0.74,
        T_const: float = 10.0,
        Rstar_Rsun: float = 1.0,
        temp_mode: str = "mctherm",
        molname: str = None,
        molabun: float = None,
        iline: int = None,
        mol_rlim: float = 1000.0,
        run_dir: str = None,
        radmc_dir: str = None,
        storage_dir: str = None,
    ):

        self.radmc = RadmcConfig(
            nphot=nphot,
            n_thread=n_thread,
            scattering_mode_max=scattering_mode_max,
            mc_scat_maxtauabs=mc_scat_maxtauabs,
            f_dg=f_dg,
            opac=opac,
            Lstar_Lsun=Lstar_Lsun,
            mfrac_H2=mfrac_H2,
            T_const=T_const,
            Rstar_Rsun=Rstar_Rsun,
            temp_mode=temp_mode,
            molname=molname,
            molabun=molabun,
            iline=iline,
            mol_rlim=mol_rlim,
            run_dir=run_dir,
            radmc_dir=radmc_dir,
            storage_dir=storage_dir,
        )


@dataclass
class ModelConfig:  # (Input):
    inenv: Any
    outenv: Any
    disk: Any
    rot_ccw: bool
    usr_density_func: Callable


@dataclass
class RadmcConfig:
    nphot: int
    n_thread: int
    scattering_mode_max: int
    mc_scat_maxtauabs: float
    f_dg: float
    opac: str
    Lstar_Lsun: float
    mfrac_H2: float
    T_const: float
    Rstar_Rsun: float
    temp_mode: str
    molname: str
    molabun: float
    iline: int
    mol_rlim: float
    run_dir: str
    radmc_dir: str
    storage_dir: str


#    Tenv: float = None
#    Mdot_smpy: float =  None
#    CR_au: float =  None
#    Ms_Msun: float = None
#    t_yr: float =  None
#    Omega: float =  None
#    maxj: float =  None
#    meanmolwe: float = 2.3
#    cavangle_deg: float = 0
#    Tdisk: float = 30
#    frac_Md: float = 0.1