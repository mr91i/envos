#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
from pmodes import cst, tools, cubicsolver, tsc
logger = logging.getLogger(__name__)
#######################

def read_model_pkl(read_path):
    return pd.read_pickle(read_path)

class GridModel:
    def set_grid(self, inp):
        self.ri_ax = inp.ri_ax
        self.ti_ax = inp.ti_ax
        self.pi_ax = inp.pi_ax
        self.rc_ax = 0.5*(self.ri_ax[0:-1] + self.ri_ax[1:])
        self.tc_ax = 0.5*(self.ti_ax[0:-1] + self.ti_ax[1:])
        self.pc_ax = 0.5*(self.pi_ax[0:-1] + self.pi_ax[1:])
        self.rr, self.tt, self.pp = np.meshgrid(self.rc_ax, self.tc_ax, self.pc_ax, indexing='ij')
        self.RR = self.rr * np.sin(self.tt)
        self.zz = self.rr * np.cos(self.tt)
        self.mm = np.round(np.cos(self.tt), 15)
        self.ss = np.where(self.mm == 1, 1e-100, np.sqrt(1-self.mm**2))

    def set_params_from_inp(self, inp):
        for k, v in inp.__dict__.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                raise Exception("Duplicated value", k)

class KinematicModel(GridModel):
    def __init__(self, inp=None):
        self.inenv = None
        self.outenv = None
        self.disk = None
        if inp is not None:
            self.set_from_inp(inp)

    def set_from_inp(self, inp):
        self.set_grid(inp)
        self.mean_mol_weight = inp.mean_mol_weight
        self.set_params(inp.Tenv, inp.CR_au, inp.M_Msun, inp.t_yr, inp.Omega, inp.j0, inp.Mdot_Msyr)
        self.inenv = InnerEnvelope(inp, self.Mdot, self.CR, self.M)
        if inp.add_outenv and (self.rc_ax[-1] > OuterEnvelope.rinlim_func(self)):
            self.outenv = OuterEnvelope(inp, self.t, self.cs, self.Omega)
        if inp.add_disk:
            self.disk = Disk(inp, self.M)
        self.update_structure()

    def add_inenv(self, Inenv):
        self.inenv = Inenv

    def add_outenv(self, Outenv):
        self.outenv = Outenv

    def add_disk(self, Disk):
        self.disk = Disk

    def initialize_structure(self):
        self.rho = np.zeros_like(self.rr)
        self.vr = np.zeros_like(self.rr)
        self.vt = np.zeros_like(self.rr)
        self.vp = np.zeros_like(self.rr)

    def update_structure(self):
        self.initialize_structure()
        if self.inenv is not None:
            rho_i, vr_i, vt_i, vp_i = self.inenv.rho, self.inenv.vr, self.inenv.vt, self.inenv.vp
            mask_i = self.inenv.maskfunc(self)
            self.rho = np.where(mask_i, rho_i, self.rho)
            self.vr = np.where(mask_i, vr_i, self.vr)
            self.vt = np.where(mask_i, vt_i, self.vt)
            self.vp = np.where(mask_i, vp_i, self.vp)

        if self.outenv is not None:
            rho_o, vr_o, vt_o, vp_o = self.outenv.rho, self.outenv.vr, self.outenv.vt, self.outenv.vp
            mask_o = self.outenv.maskfunc(self)
            self.rho = np.where(mask_o, rho_o, self.rho)
            self.vr = np.where(mask_o, vr_o, self.vr)
            self.vt = np.where(mask_o, vt_o, self.vt)
            self.vp = np.where(mask_o, vp_o, self.vp)

        if self.disk is not None:
            rho_d, vr_d, vt_d, vp_d = self.disk.rho, self.disk.vr, self.disk.vt, self.disk.vp
            mask_d = self.disk.maskfunc(self) # rho_d > np.nan_to_num(self.rho)
            self.rho = np.where(mask_d, rho_d, self.rho)
            self.vr = np.where(mask_d, vr_d, self.vr)
            self.vt = np.where(mask_d, vt_d, self.vt)
            self.vp = np.where(mask_d, vp_d, self.vp)

    def set_params(self, T, CR_au, M_Msun, t_yr, Omega, j0, Mdot_Msyr):  # choose 3
        #       eq1(cs,T),  eq2(Mdot,cs) ==> eq3(CR,M,j0), eq4(M,t), eq5(j0,Omega) : 5 vars 3 eqs
        if sum((a is not None) for a in (T, CR_au, M_Msun, t_yr, Omega, j0, Mdot_Msyr)) != 3:
            raise Exception("Too many given parameters.")
        m0 = 0.975
        try:
            if T is not None:
                cs = np.sqrt(cst.kB*T/(self.mean_mol_weight * cst.amu))
                Mdot = cs**3 * m0 / cst.G
            elif Mdot_Msyr is not None:
                Mdot = Mdot_Msyr * cst.Msun / cst.yr
                T = (Mdot*cst.G/m0)**(2/3) * self.mean_mol_weight * cst.amu / cst.kB
                cs = np.sqrt(cst.kB*T/(self.mean_mol_weight * cst.amu))

            if M_Msun is not None:
                M = M_Msun * cst.Msun
                t = M / Mdot
            elif t_yr is not None:
                t = t_yr * cst.yr
                M = Mdot * t

            if CR_au is not None:
                CR = CR_au * cst.au
                j0 = np.sqrt(CR * cst.G * M)
                Omega = j0 / (0.5*cs*m0*t)**2
            elif j0 is not None:
                CR = j0**2 / (cst.G * M)
                Omega = j0 / (0.5*cs*m0*t)**2
            elif Omega is not None:
                j0 = (0.5*cs*m0*t)**2 * Omega
                CR = j0**2 / (cst.G * M)
        except Exception as e:
            raise Exception(e, "Something wrong in parameter equations.")
        self.Tenv = T
        self.cs = cs
        self.Mdot = Mdot
        self.CR = CR
        self.M = M
        self.t = t
        self.Omega = Omega
        self.j0 = j0

    def print_params(self, inp):
        def logpara(var, val, unit):
            s = f'is {val:10.2g} ' if np.isscalar(val) \
                  else f'is [{val[0]:10.2g}:{val[-1]:10.2g}] (n = {len(val):d})'
            logger.info(var.ljust(10) + s + unit.ljust(10))
        logger.info('Model Parameters:')
        logger.info(f'Model {self.model}')
        logpara('rc', self.rc_ax, 'au')
        logpara('tc', self.tc_ax, 'rad')
        logpara('pc', self.pc_ax, 'rad')
        logpara('T', self.Tenv, 'K')
        logpara('cs', self.cs/cst.kms, 'km/s')
        logpara('t', self.t/cst.yr, 'yr')
        logpara('M', self.M/cst.Msun, 'Msun')
        logpara('Omega', self.Omega, 's^-1')
        logpara('dM/dt', self.Mdot/(cst.Msun/cst.yr), 'M/yr')
        # logpara('r_lim', self.rlim_TSC/cst.au, 'au')
        logpara('j0', self.j0/(cst.kms*cst.au), 'au*km/s')
        logpara('j0', self.j0/(cst.kms*cst.pc), 'pc*km/s')
        logpara('r_CB', self.CR/2/cst.au, 'au')
        logpara('r_CR', self.CR/cst.au, 'au')
        logpara('c.a.', inp.cavangle_deg, 'degree')
        logger.info(f"tau is {self.Omega*self.t}")
        logger.info(f"xin_TSC is {self.rlim_TSC/self.cs/self.t}")

    def save_pickle(self, save_path=None):
        save_path = save_path if save_path is not None else self.pkl_filename
        #pd.to_pickle(self.__dict__, savefile, protocol=2)
        os.mkedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        pd.to_pickle(self, save_path)
        logger.info(f'Saved : {save_path}\n')

## Do not use only this one
#class InnerEnvelopeModel(ModelBase):
class InnerEnvelope(GridModel):
    def __init__(self, inp, Mdot, CR, M):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Mdot = Mdot
        self.CR = CR
        self.Mstar = M
        self.mean_mol_weight = inp.mean_mol_weight
        self.cavangle_deg = inp.cavangle_deg
        self.set_grid(inp)
        self.calc_kinematic_structure(inp.envelope_model, rot_ccw=inp.rot_ccw, inenv_density_mode=inp.inenv_density_mode, rho0_1au=inp.rho0_1au, rho_index=inp.rho_index)
        #self.calc_kinematic_structure(inp, rot_ccw=inp.rot_ccw, inenv_density_mode=inp.inenv_density_mode

    def calc_kinematic_structure(self, model, rot_ccw=False, inenv_density_mode=None, rho0_1au=None, rho_index=None):
        if model == 'CM':
            rho, vr, vt, vp, zeta, mu0 = self.get_kinematics_CassenMoosman()
        elif model == 'Simple':
            rho, vr, vt, vp, zeta, mu0 = self.get_kinematics_SimpleBalistic()
        else:
            raise Exception("Unknown model : ", model)
        if np.isnan([rho, vr, vt, vp, zeta, mu0]).any():
            raise Exception("Bad values.", rho, vr, vt, vp, zeta, mu0)
        if inenv_density_mode == "powerlaw":
            rho = self.rho0_1au*(self.rr/cst.au)**self.rho_index
        elif inenv_density_mode == "Shu":
            rho = self.Mdot/(4 * np.pi * self.rr**2 * vff)
        cav_mask = np.where(mu0 <= np.cos(self.cavangle_deg/180*np.pi), 1, 0)
        self.rho = rho * cav_mask
        self.vr = vr
        self.vt = vt
        self.vp = vp if not rot_ccw else -vp
        self.zeta = zeta
        self.mu0 = mu0
        self.uR = self.vr * self.ss + self.vt * self.mm
        self.uz = self.vr * self.mm - self.vt * self.ss

    def get_kinematics_CassenMoosman(self):
        def sol_with_cubic(m, zeta):
            sols = [ round(sol, 8) for sol in cubicsolver.solve(zeta, 0, 1-zeta, -m).real if 0 <= round(sol, 8) <= 1 ]
            return sols[0] if len(sols) != 0 else np.nan
        zeta = self.CR / self.rr
        csol = np.frompyfunc(sol_with_cubic, 2, 1)
        mu0 = csol(self.mm, zeta).astype(np.float64)
        sin0 = np.sqrt(1 - mu0**2)
        v0 = np.sqrt(cst.G*self.Mstar / self.rr)
        mu_over_mu0 = 1 - zeta*(1 - mu0**2)
        vr = - v0 * np.sqrt(1 + mu_over_mu0)
        vt = v0 * zeta*sin0**2*mu0/self.ss * np.sqrt(1 + mu_over_mu0)
        vp = v0 * sin0**2/self.ss * np.sqrt(zeta)
        rho = - self.Mdot / (4 * np.pi * self.rr**2 * vr * (1 + zeta*(3*mu0**2-1) ) )
        return rho, vr, vt, vp, zeta, mu0

    def get_kinematics_SimpleBalistic(self, ):
        vff = np.sqrt(2 * cst.G*self.Mstar / self.rr)
        CB = self.CR / 2
        rho_prof = self.Mdot/(4 * np.pi * self.rr**2 * vff)
        rho = np.where(self.rr >= CB, rho_prof, rho_prof*0 )
        vr = - vff * np.sqrt( (1-CB/self.rr).clip(0) )
        vt = np.zeros_like(self.rr)
        vp = vff/np.sqrt(self.rr/CB)
        zeta = np.zeros_like(self.rr)
        mu0 = self.mm
        return rho, vr, vt, vp, zeta, mu0

    def maskfunc(self, model):
        return np.full_like(self.rr, True)

class OuterEnvelope(GridModel):
    def __init__(self, inp, t, cs, Omega):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.t = t
        self.cs = cs
        self.Omega = Omega
        self.rin_lim = self.cs * self.Omega**2 * self.t**3 * 1
        self.set_grid(inp)
        self.tsc = tsc.solve_TSC(t, cs, Omega, run=True, r_crit=self.rin_lim)
        self.rho = self.tsc.frho(self.rr, self.tt)
        self.vr, self.vt, self.vp = self.tsc.fvelo(self.rr, self.tt)
        if inp.rot_ccw:
            self.vp *= -1

    @staticmethod
    def rinlim_func(model):
        return  model.cs * model.Omega**2 * model.t**3 * 0.1

    def maskfunc(self, model):
        return model.rr <  self.rinlim_func(model)


class Disk(GridModel):
    def __init__(self, inp, M):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Mstar = M
        self.M_fin = inp.M_fin
        self.Tdisk = inp.Tdisk
        self.frac_Md = inp.frac_Md
        self.mean_mol_weight = inp.mean_mol_weight
        self.Mdisk = self.frac_Md * self.Mstar
        self.cs_disk = np.sqrt(cst.kB * self.Tdisk / (self.mean_mol_weight * cst.amu))

        self.set_grid(inp)
        self.set_params()
        self.calc_disk(mode=inp.mode)
        if rot_ccw:
            self.vp *= -1

    def calc_disk(self, mode='expcutoff', ind=-1):
    ## Still constructing
        if mode == 'CM_visc': # Viscous case
            u = self.RR/self.r_CR
#           P = self.Mfin / self.Mdot / ( self.r_CR**2 / (3*0.01*self.csd**2) * np.sqrt(cst.G*self.M /self.r_CR_fin**3) )
            P = (3*0.01*self.cs_disk**2) / np.sqrt(cst.G*self.Mstar / self.r_CR**3) * \
                self.Mstar**6/self.Mfin**5/self.Mdot/self.r_CR**2
            P_rd2 = 3 * 0.1 * self.Tdisk/self.Tenv * \
                np.sqrt(self.Mfin/self.Mstar) * \
                np.sqrt(cst.G * self.Mfin/self.r_CR)/self.cs_disk
            a3 = 0.2757347731  # = (2^10/3^11)^(0.25)
            ue = np.sqrt(3*P)*(1-56/51*a3/P**0.25)
            y = np.where(u < 1, 2*np.sqrt(1-u.clip(max=1)) + 4/3/np.sqrt(u) *
                         (1-(1+0.5*u)*np.sqrt(1-u.clip(max=1))), 4/3/np.sqrt(u))
            y = np.where(u <= ue, y - 4/3/np.sqrt( ue.clip(1e-30) ), 0)
            Sigma = 0.5/P * y * self.Mstar/(np.pi*self.r_CR**2)
        elif mode == 'S+94':
            pass
#           A = 4*a/(m0 * Omega0 * cst.Rsun)
#           u = R/R_CR
#           V =
#           Sigma = (1-u)**0.5 /(2*A*u*t**2*V)
        elif mode == 'expcutoff':
            Sigma_0 = self.Mdisk/(2*np.pi*self.r_CR**2)/(1-2/np.e)
            Sigma = Sigma_0 * (self.RR/cst.au)**ind * np.exp(-(self.RR/self.r_CR)**(2+ind))
        elif mode == 'tapered_cutoff':
            Sigma_0 = self.Mdisk/(2*np.pi*self.r_CR**2) * (ind+3)
            Sigma = self.Mdisk/(2*np.pi) * (ind+3) * \
                self.r_CR**(-ind-3) * (self.RR/cst.au)**ind
        H = np.sqrt(cst.kB * self.Tdisk /(self.mean_mol_weight * cst.amu)) / np.sqrt(cst.G*self.Mstar/self.RR**3)
        self.rho = Sigma/np.sqrt(2*np.pi)/H * np.exp(-0.5*self.zz**2/H**2)
        self.vr = np.full_like(self.rho, 0)
        self.vt = np.full_like(self.rho, 0)
        self.vp = np.sqrt(cst.G*self.Mstar/self.RR)

    def maskfunc(self, model):
        return model.rho < self.rho

def save_kinemo_hdf5_spherical(model, save_path):
    from evtk.hl import gridToVTK
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gridToVTK(save_path, model.ri_ax/cst.au, model.ti_ax, model.pi_ax,
              cellData = {"den" :model.rho, "ur" :model.vr, "uth" :model.vth, "uph" :model.vph})

def save_kinemo_hdf5_certesian(model, save_path):
    from evtk.hl import gridToVTK
    from scipy.interpolate import interpn # , RectBivariateSpline, RegularGridInterpolator
    L = model.rc_ax[-1]
    xi = np.linspace(-L/10,L/10,200)
    yi = np.linspace(-L/10,L/10,200)
    zi = np.linspace(0, L/10, 100)
    xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
    xc = tools.make_array_center(xi)
    yc = tools.make_array_center(yi)
    zc = tools.make_array_center(zi)
    xxc, yyc, zzc = np.meshgrid(xc, yc, zc, indexing='ij')
    rr_cert = np.sqrt(xxc**2 + yyc**2 + zzc**2)
    tt_cert = np.arccos(zzc/rr_cert)
    pp_cert = np.arctan2(yyc, xxc)
    def interper(val):
        return interpn((model.rc_ax, model.tc_ax, model.pc_ax), val,
                       np.stack([rr_cert, tt_cert, pp_cert], axis=-1), bounds_error=False, fill_value=np.nan)
    den_cert = interper(model.rho)
    vr_cert = interper(model.vr)
    vt_cert = interper(model.vt)
    vp_cert = interper(model.vp)
    uux = vr_cert * np.sin(tt_cert) * np.cos(pp_cert) + vt_cert * np.cos(tt_cert) * np.cos(pp_cert) - vp_cert  * np.sin(pp_cert)
    uuy = vr_cert * np.sin(tt_cert) * np.sin(pp_cert) + vt_cert * np.cos(tt_cert) * np.sin(pp_cert) + vp_cert * np.cos(pp_cert)
    uuz = vr_cert * np.cos(tt_cert) - vt_cert * np.sin(tt_cert)
    os.makedir(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gridToVTK(save_path, xi/cst.au, yi/cst.au, zi/cst.au,
              cellData = {"den" :den_cert, "ux" :uux, "uy" :uuy, "uz" :uuz})
