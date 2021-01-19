import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    def set_grid(self,
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
        self, T: float = None,
        CR_au: float = None,
        Ms_Msun: float = None,
        t_yr: float = None,
        Omega: float = None,
        maxj: float = None,
        Mdot_Msyr: float = None,
        meanmolw: float = 2.3,
        cavangle_deg: float = 0):

        self.ppar = PhysicalParameters(
                T
                CR_au=CR_au,
                Ms_Msun=Ms_Msun,
                t_yr=t_yr,
                Omega=Omega,
                maxj=maxj,
                Mdot_Msyr=Mdot_Msyr,
                meanmolw=meanmolw,
                cavangle_deg=cavangle_deg,
            )
    def set_model_input(self, inenv="CM", outenv=None, disk=None):

