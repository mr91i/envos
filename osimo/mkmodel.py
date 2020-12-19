#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
from osimo import tools, cubicsolver, tsc, log
from osimo import nconst as nc
logger = log.set_logger(__name__, ini=True)
#######################

###############################################################################################
###############################################################################################

def build_with_inp(inp):
    global logger
    logger = log.set_logger(__name__)
    grid = Grid(inp.ri_ax, inp.ti_ax, inp.pi_ax)
    ppar = PhysicalParameters(T=inp.Tenv, CR_au=inp.CR_au, M_Msun=inp.M_Msun, t_yr=inp.t_yr,
                 Omega=inp.Omega, j0=inp.j0, Mdot_Msyr=inp.Mdot_Msyr, meanmolw=inp.meanmolw)

    builder = KinematicModelBuilder()
    builder.set_grid(grid)
    builder.set_physpar(ppar)
    builder.build(inenv=inp.inenv, disk=inp.disk, outenv=inp.outenv)
    model = builder.get_model()
    return model


class KinematicModelBuilder:
    def __init__(self):
        self.kmodel = ModelBase
        self.inenv = None
        self.outenv = None
        self.disk = None

    def set_grid(self, Grid):
        self.grid = Grid

    def set_ppar(self, PhysicalParameters):
        self.ppar = PhysicalParameters

    def add_inenv(self, Inenv):
        self.inenv = Inenv

    def add_outenv(self, Outenv):
        self.outenv = Outenv

    def add_disk(self, Disk):
        self.disk = Disk

    def make_model(self):
        region_index = np.zeros_like(self.rr)
        if self.disk is not None:
            region_index[ self.disk.rho > self.inenv.rho ] += 1
        if self.outenv is not None:
            region_index[ self.rr > self.outenv.rho ] += 10
        def _put_value_with_index(i, key):
            return {0:getattr(self.inenv, key), 1:getattr(self.disk, key), 10:getattr(self.outenv, key)}[i]
        put_value_with_index = np.frompyfunc( _put_value_with_index, 2, 1)
        self.kmodel.rho = put_value_with_index(region_index, "rho")
        self.kmodel.vr = put_value_with_index(region_index, "vr")
        self.kmodel.vt = put_value_with_index(region_index, "vt")
        self.kmodel.vp = put_value_with_index(region_index, "vp")

    def build(self, grid=None, ppar=None, inenv="CM", disk="", outenv=""):
        ### Set grid
        if (self.grid is None) and (grid is None):
            raise Exception("grid is not set.")
        grid = grid if grid is not None else self.grid

        ### Set ppar
        if (self.ppar is None) and (ppar is None):
            raise Exception("grid is not set.")
        ppar = self.ppar

        ### Set models
        # Set inenv
        if not isinstance(inenv, str):
            self.add_inenv(inenv)
        elif inenv_model == "CM":
            cm = CassenMoosmanInnerEnvelope(grid, ppar.Mdot, ppar.CR, ppar.M)
            self.add_inenv(cm)
        elif inenv_model == "Simple":
            sb = SimpleBalisticInnerEnvelope(grid, ppar.Mdot, ppar.CR, ppar.M)
            self.add_inenv(sb)
        else:
            raise Exception("No Envelope.")

        # Set outenv
        if self.grid.rc_ax[-1] > self.ppar.rinlim_tsc:
            if not isinstance(outenv, str):
                self.add_outenv(inenv)
            if outenv_model == "TSC":
                tsc = TerebeyOuterEnvelope(ppar.t, ppar.cs, ppar.Omega)
                self.add_outenv(tsc)
        # set disk
        if not isinstance(disk, str):
            self.add_outenv(disk)
        elif disk == "exptail":
            disk = ExptailDisk(grid, ppar.Ms, ppar.CR, Td=30, fracMd=0.1, meanmolw=2.3, index=-1.5)
            self.add_outenv(disk)

        ### Make kmodel
        self.make_model()

    def get_model(self):
        return self.kmodel

    def save_model_pickle(self, save_path=None):
        save_path = save_path if save_path is not None else "./kmodel.pkl"
        os.mkedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        pd.to_pickle(self, save_path)
        logger.info(f'Saved : {save_path}\n')

####################################################################################################

class PhysicalParameters:
    def __init__(self, T=None, CR_au=None, M_Msun=None, t_yr=None, Omega=None, j0=None, Mdot_Msyr=None, meanmolw=None):
        self.Tenv = None
        self.cs = None
        self.Mdot = None
        self.CR = None
        self.Ms = None
        self.t = None
        self.Omega = None
        self.maxangmom = None
        self.meanmolw =  None

        self.set_param_set(T, CR, Ms, t, Omega, j0, Mdot, meanmolw)
        self.log_param_set()

    #def set_param_set(self, T, CR_au, M_Msun, t_yr, Omega, j0, Mdot_Msyr, meanmolw):  # choose 3
    def set_param_set(self, T, CR, Ms, t, Omega, j0, Mdot, meanmolw):  # choose 3
        if sum((a is not None) for a in (T, CR_au, M_Msun, t_yr, Omega, j0, Mdot_Msyr)) != 3:
            raise Exception("Too many given parameters.")
        m0 = 0.975
        try:
            if T is not None:
                self.T = T
                self.cs = np.sqrt(nc.kB*self.T/(self.meanmolw * nc.amu))
                self.Mdot = cs**3 * m0 / nc.G
            elif Mdot is not None:
                self.Mdot = Mdot
                self.T = (Mdot*nc.G/m0)**(2/3) * meanmolw * nc.amu / nc.kB
                self.cs = np.sqrt(nc.kB*T/(meanmolw * nc.amu))

            if Ms is not None:
                self.Ms = Ms
                self.t = M / Mdot
            elif t is not None:
                self.t = t
                self.Ms = Mdot * t

            if CR is not None:
                self.CR = CR
                self.maxangmom = np.sqrt(CR * nc.G * self.Ms)
                self.Omega = self.maxangmom / (0.5*self.cs*m0*self.t)**2
            elif j0 is not None:
                self.maxangmom = j0
                self.CR = j0**2 / (nc.G * self.Ms)
                self.Omega = j0 / (0.5*self.cs*m0*self.t)**2
            elif Omega is not None:
                self.Omega = Omega
                self.maxangmom = (0.5*self.cs*m0*self.t)**2 * Omega
                self.CR = self.maxangmom**2 / (nc.G * self.Ms)
        except Exception as e:
            raise Exception(e, "Something wrong in parameter equations.")

        rinlim_tsc = cs*Omega**2*t**3


    def log_param_set(self):
        def logp(valname, unit_name, val, unit_val=1):
            val = paramset[key]/unit_val
            logger.info(key.ljust(10)+f'is {val:10.2g} '+unit_name.ljust(10))

        logger.info('Model Parameters:')
        logp("Tenv", 'K', self.T)
        logp("cs", 'km/s', self.cs, nc.kms)
        logp('t', 'yr', self.t, nc.yr)
        logp('Ms', 'Msun', self.Ms, nc.Msun)
        logp('Omega', 's^-1', self.Omega)
        logp('Mdot', "Msun/yr", self.Mdot, nc.Msun/nc.yr)
        logp('j0', 'au*km/s', self.maxangmom, nc.kms*nc.au)
        logp('j0', 'pc*km/s', self.maxangmom, nc.kms*nc.pc)
        logp('CR', "au", self.CR, nc.au)
        logp('CB', "au", self.CR/2, nc.au)
        logp('meanmolw', "", self.meanmolw)
        logp('cavangle', self.cavangle_deg, 'deg')
        logp('Omega*t', '', self.Omega*self.t)
        logp('rinlim_tsc', 'au', self.cs*Omega**2*t**3, nc.au)
        logp('rinlim_tsc', 'cs*t', Omega**2*t**2)

####################################################################################################

class Grid:
    def __init__(self, ri_ax, ti_ax, pi_ax):
        self.ri_ax = ri_ax
        self.ti_ax = ti_ax
        self.pi_ax = pi_ax
        self.rc_ax = 0.5*(self.ri_ax[0:-1] + self.ri_ax[1:])
        self.tc_ax = 0.5*(self.ti_ax[0:-1] + self.ti_ax[1:])
        self.pc_ax = 0.5*(self.pi_ax[0:-1] + self.pi_ax[1:])
        self.rr, self.tt, self.pp = np.meshgrid(self.rc_ax, self.tc_ax, self.pc_ax, indexing='ij')
        self.RR = self.rr * np.sin(self.tt)
        self.zz = self.rr * np.cos(self.tt)
        self.mm = np.round(np.cos(self.tt), 15)
        self.ss = np.where(self.mm == 1, 1e-100, np.sqrt(1-self.mm**2))

####################################################################################################

class ModelBase:
    def maskfunc(self, model):
        return np.full_like(self.rr, True)

    def set_grid(self, grid):
        for k, v in grip.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.uR = self.vr * self.ss + self.vt * self.mm
        self.uz = self.vr * self.mm - self.vt * self.ss

    def check_include_nan(self):
        if np.isnan([self.rho, self.vr, self.vt, self.vp, self.zeta, self.mu0]).any():
            raise Exception("Bad values.", self.rho, self.vr, self.vt, self.vp, self.zeta, self.mu0)


class CassenMoosmanInnerEnvelope(ModelBase):
    def __init__(self, grid, Mdot, CR, M, cavangle_deg=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.Mdot = Mdot
        self.CR = CR
        self.Ms = M
        self.cavangle_deg = cavangle_deg
        self.set_grid(grid)
        self.calc_kinematic_structure()

    def calc_kinematic_structure(self):
        csol = np.frompyfunc(sol_with_cubic, 2, 1)
        zeta = self.CR / self.rr
        self.mu0 = csol(self.mm, zeta).astype(np.float64)
        sin0 = np.sqrt(1 - self.mu0**2)
        mu_over_mu0 = 1 - zeta*(1 - self.mu0**2)
        v0 = np.sqrt(nc.G*self.Mstar / self.rr)
        self.vr = - v0 * np.sqrt(1 + mu_over_mu0)
        self.vt = v0 * zeta*sin0**2*self.mu0/self.ss * np.sqrt(1 + mu_over_mu0)
        self.vp = v0 * sin0**2/self.ss * np.sqrt(zeta)
        rho = - self.Mdot/(4 * np.pi * self.rr**2 * vr * (1 + zeta*(3*mu0**2-1) ))
        self.rho = rho * self.cav_mask()

    def cav_mask(self):
        return np.where(mu0 <= np.cos(np.radians(self.cavangle_deg))))

    @staticmethod
    def _sol_with_cubic(m, zeta):
        allsols = np.round(cubicsolver.solve(zeta, 0, 1-zeta, -m).real, 8)
        sol = [ sol for sol in allsols if 0 <= sol <= 1 ]
        return sols[0] if len(sols) != 0 else np.nan

class SimpleBallisticInnerEnvelope(ModelBase):
    def __init__(self, grid, Mdot, CR, M, cavangle_deg=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.mu0 = None
        self.Mdot = Mdot
        self.CR = CR
        self.Ms = M
        self.cavangle_deg = cavangle_deg
        self.set_grid(grid)
        self.calc_kinematic_structure()

    def calc_kinematic_structure(self):
        vff = np.sqrt(2*nc.G*self.Mstar/self.rr)
        CB = self.CR / 2
        rho_prof = self.Mdot/(4*np.pi*self.rr**2*vff)
        self.rho = rho_prof * np.where(self.rr >= CB) * self.cav_mask()
        self.vr = - vff * np.sqrt( (1-CB/self.rr).clip(0) )
        self.vt = np.zeros_like(self.rr)
        self.vp = vff/np.sqrt(self.rr/CB)
        self.mu0 = self.mm

    def cav_mask(self):
        return np.where(mu0 <= np.cos(np.radians(self.cavangle_deg)))

##################################################

class TerebeyOuterEnvelope(ModelBase):
    def __init__(self, grid, t, cs, Omega, cavangle_deg=0):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.t = t
        self.cs = cs
        self.Omega = Omega
        self.cavangle_deg = cavangle_deg
        self.rin_lim = self.cs * self.Omega**2 * self.t**3 * 1
        self.set_grid(grid)
        self.calc_kinematic_structure()

    def calc_kinematic_structure(self):
        tsc = tsc.solve_TSC(self.t, self.cs, self.Omega, run=True, r_crit=self.rin_lim)
        self.rho = tsc.frho(self.rr, self.tt) * self.cav_mask()
        self.vr, self.vt, self.vp = tsc.fvelo(self.rr, self.tt)

    def cav_mask(self):
        return np.where(self.tt <= np.radians(self.cavangle_deg))

    @staticmethod
    def rinlim_func(model):
        return  model.cs * model.Omega**2 * model.t**3 * 0.1

    def maskfunc(self, model):
        return model.rr <  self.rinlim_func(model)

##################################################

class Disk(ModelBase):
    def calc_kinematic_structure_from_Sigma(self, Sigma):
        OmegaK = np.sqrt(nc.G*self.Ms/self.RR**3)
        H = self.cs / OmegaK
        self.rho = Sigma/(np.sqrt(2*np.pi)*H) * np.exp(-0.5*(self.zz/H)**2)
        self.vr = np.zeros_like(self.rho)
        self.vt = np.zeros_like(self.rho)
        self.vp = OmegaK*self.RR

class CMviscDisk(Disk):
    def __init__(self, grid, Ms, CR, Td, fracMd, meanmolw):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Ms = Ms
        self.CR = CR
        self.Mdisk = frac_Md * self.Mstar
        self.cs = np.sqrt(nc.kB * Td /(meanmolw * nc.amu))
        self.set_grid(grid)
        Sigma = self.get_Sigma()
        self.calc_kinematic_structure_from_Sigma(Sigma)

    def get_Sigma(self):
        logger.error("Still Constructing!")
        exit()
        u = self.RR/self.CR
        P = (3*0.01*self.cs**2) / np.sqrt(nc.G*self.Ms/self.CR**3) * self.Mstar**6/self.Mfin**5/self.Mdot/self.CR**2
        P_rd2 = 3 * 0.1 * self.Tdisk/self.Tenv * np.sqrt(self.Mfin/self.Mstar) * np.sqrt(nc.G * self.Mfin/self.r_CR)/self.cs_disk
        a3 = 0.2757347731  # = (2^10/3^11)^(0.25)
        ue = np.sqrt(3*P)*(1-56/51*a3/P**0.25)
        y = np.where(u < 1, 2*np.sqrt(1-u.clip(max=1)) + 4/3/np.sqrt(u) *
                     (1-(1+0.5*u)*np.sqrt(1-u.clip(max=1))), 4/3/np.sqrt(u))
        y = np.where(u <= ue, y - 4/3/np.sqrt( ue.clip(1e-30) ), 0)
        Sigma = 0.5/P * y * self.Ms/(np.pi*self.CR**2)
        return Sigma


class ExptailDisk(Disk):
    def __init__(self, grid, Ms, Rd, Td=30, fracMd=0.1, meanmolw=2.3, index=-1.5):
        self.rho = None
        self.vr = None
        self.vt = None
        self.vp = None
        self.Ms = Ms
        self.cs = np.sqrt(nc.kB * Td /(meanmolw * nc.amu))
        self.set_grid(grid)
        Mdisk = frac_Md * self.Mstar
        Sigma = self.get_Sigma(Mdisk, Rd, index)
        self.calc_kinematic_structure_from_Sigma(Sigma)

    def get_Sigma(self, Mdisk, Rd, ind):
        Sigma0 = Mdisk/(2*np.pi*Rd**2)/(1-2/np.e)
        return Sigma0 * (self.RR/nc.au)**ind * np.exp(-(self.RR/self.Rd)**(2+ind))


####################################################################################################
## Functions                                                                                       #
####################################################################################################


def read_model_pkl(read_path):
    return pd.read_pickle(read_path)

def save_kmodel_hdf5_spherical(model, save_path):
    from evtk.hl import gridToVTK
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    gridToVTK(save_path, model.ri_ax/nc.au, model.ti_ax, model.pi_ax,
              cellData = {"den" :model.rho, "ur" :model.vr, "uth" :model.vth, "uph" :model.vph})

def save_kmodel_hdf5_certesian(model, save_path):
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
    gridToVTK(save_path, xi/nc.au, yi/nc.au, zi/nc.au,
              cellData = {"den" :den_cert, "ux" :uux, "uy" :uuy, "uz" :uuz})
