#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import logging
from envos import tools, cubicsolver, tsc, log, config
from envos import nconst as nc
logger = log.set_logger(__name__, ini=True)

class ModelBase:
    def read_grid(self, grid):
        for k, v in grid.__dict__.items():
            setattr(self, k, v)

    def set_cylindrical_velocity(self):
        self.vR = self.vr * self.sin + self.vt * self.mu
        self.vz = self.vr * self.mu - self.vt * self.sin

    def check_include_nan(self):
        if np.isnan([self.rho, self.vr, self.vt, self.vp, self.zeta, self.mu0]).any():
            raise Exception("Bad values.", self.rho, self.vr, self.vt, self.vp, self.zeta, self.mu0)

class Grid:
    def __init__(self, rau_lim=None, theta_lim=None, phi_lim=(0, 2*np.pi),
                 axes=None, nr=60, ntheta=180, nphi=1,
                 dr_to_r0=None, aspect=None, logr=True):
        if axes is not None:
            self.ri_ax = axes[0]
            self.ti_ax = axes[1]
            self.pi_ax = axes[2]
        else:
            if dr_to_r0 is not None:
                nr = int(2.3/dr_to_r0 * np.log10(rau_lim[1]/rau_lim[0]) )
            if aspect is not None:
                ntheta = self._get_square_ntheta(rau_lim, nr, assratio=aspect)

            if logr:
                self.ri_ax = np.logspace(*np.log10(rau_lim), nr+1) * nc.au
            else:
                self.ri_ax = np.linspace(*rau_lim, nr+1) * nc.au
            self.ti_ax = np.linspace(*theta_lim, ntheta+1)
            self.pi_ax = np.linspace(*phi_lim, nphi+1)

        self.set_params()

    def set_params(self):
        self.rc_ax = 0.5*(self.ri_ax[0:-1] + self.ri_ax[1:])
        self.tc_ax = 0.5*(self.ti_ax[0:-1] + self.ti_ax[1:])
        self.pc_ax = 0.5*(self.pi_ax[0:-1] + self.pi_ax[1:])
        self.rr, self.tt, self.pp = np.meshgrid(self.rc_ax, self.tc_ax, self.pc_ax, indexing='ij')
        self.R = self.rr * np.sin(self.tt)
        self.z = self.rr * np.cos(self.tt)
        self.mu = np.round(np.cos(self.tt), 15)
        self.sin = np.where(self.mu == 1, 1e-100, np.sqrt(1-self.mu**2))

    @staticmethod
    def _get_square_ntheta(r_range, nr, assratio=1):
        dlogr = np.log10(r_range[1]/r_range[0])/nr
        dr_over_r = 10**dlogr -1
        return int(round(0.5*np.pi/dr_over_r /assratio ))

class KinematicModel(ModelBase):
    def __init__(self, filepath=None, grid=None):
        self.inenv = None
        self.outenv = None
        self.disk = None
        self.grid = grid
        if filepath is not None:
            read_model(filepath, base=self)


    def set_grid(self, rau_lim=None, theta_lim=(0, np.pi/2),
                 phi_lim=(0, 2*np.pi), axes=None, nr=60, ntheta=180, nphi=1,
                 dr_to_r0=None, aspect=None, logr=True, grid=None):
        if grid is not None:
            self.grid = grid
        else:
            args = ["rau_lim", "theta_lim", "phi_lim", "axes", "nr",
                    "ntheta", "nphi", "dr_to_r0", "aspect", "logr"]
            loc = locals()
            self.grid = Grid(**dict((a, loc[a]) for a in args))
        self.read_grid(self.grid)

    def set_physical_params(self, *, T=None, CR_au=None, M_Msun=None, t_yr=None,
            Omega=None, j0=None, Mdot_Msyr=None, meanmolw=2.3, cavangle_deg=0):
        lst = (T, CR_au, M_Msun, t_yr, Omega, j0, Mdot_Msyr)
        if sum(a is not None for a in lst) != 3:
            raise Exception("Too many given parameters.")

        m0 = 0.975
        self.meanmolw = meanmolw
        self.cavangle_deg = cavangle_deg

        if T is not None:
            cs = np.sqrt(nc.kB*T/(meanmolw * nc.amu))
            Mdot = cs**3 * m0 / nc.G
        elif Mdot_Msyr is not None:
            Mdot = Mdot_Msyr * nc.Msun / nc.year
            T = (Mdot*nc.G/m0)**(2/3) * meanmolw * nc.amu / nc.kB
            cs = np.sqrt(nc.kB*T/(meanmolw * nc.amu))

        if M_Msun is not None:
            M = M_Msun * nc.Msun
            t = M / Mdot
        elif t_yr is not None:
            t = t_yr * nc.yr
            M = Mdot * t

        if CR_au is not None:
            CR = CR_au * nc.au
            maxangmom = np.sqrt(CR * nc.G * M)
            Omega = maxangmom / (0.5*cs*m0*t)**2
        elif j0 is not None:
            maxangmom = j0
            CR = j0**2 / (nc.G * M)
            Omega = j0 / (0.5*cs*m0*t)**2
        elif Omega is not None:
            Omega = Omega
            maxangmom = (0.5*cs*m0*t)**2 * Omega
            CR = maxangmom**2 / (nc.G * M)

        for k in ("T", "cs", "CR", "M", "t", "Omega", "maxangmom", "Mdot"):
            setattr(self, k, locals()[k])

        self.rinlim_tsc = cs*Omega**2*t**3

        logger.info('Model Parameters:')
        self._logp("Tenv", 'K', T)
        self._logp("cs", 'km/s', cs, nc.kms)
        self._logp('t', 'yr', t, nc.yr)
        self._logp('Ms', 'Msun', M, nc.Msun)
        self._logp('Omega', 's^-1', Omega)
        self._logp('Mdot', "Msun/yr", Mdot, nc.Msun/nc.yr)
        self._logp('j0', 'au*km/s', maxangmom, nc.kms*nc.au)
        self._logp('j0', 'pc*km/s', maxangmom, nc.kms*nc.pc)
        self._logp('CR', "au", CR, nc.au)
        self._logp('CB', "au", CR/2, nc.au)
        self._logp('meanmolw', "", meanmolw)
        self._logp('cavangle', "deg", cavangle_deg)
        self._logp('Omega*t', '', Omega*t)
        self._logp('rinlim_tsc', 'au', cs*Omega**2*t**3, nc.au)
        self._logp('rinlim_tsc', 'cs*t', Omega**2*t**2)

    @staticmethod
    def _logp(valname, unitname, val, unitval=1):
        logger.info(valname.ljust(10)+f'is {val/unitval:10.2g} '+unitname.ljust(10))

    def build(self, grid=None, inenv="CM", disk=None, outenv=None):
        ### Set grid
        if (self.grid is None) and (grid is None):
            raise Exception("grid is not set.")
        grid = grid or self.grid

        ### Set models
        # Set inenv
        if not isinstance(inenv, str):
            self.inenv = inenv
        elif inenv == "CM":
            self.inenv = CassenMoosmanInnerEnvelope(grid, self.Mdot, self.CR, self.M)
        elif inenv == "Simple":
            self.inenv = SimpleBalisticInnerEnvelope(grid, self.Mdot, self.CR, self.M)
        else:
            raise Exception("No Envelope.")

        # Set outenv
        if self.grid.rc_ax[-1] > self.rinlim_tsc:
            if hasattr(outenv, "rho"):
                self.outenv = outenv
            elif outenv == "TSC":
                self.outenv = TerebeyOuterEnvelope(grid, self.t, self.cs, self.Omega)
            elif outenv is not None:
                 raise Exception("Unknown outenv type")

        # Set disk
        if hasattr(self, "rho"):
            self.disk = disk
        elif disk == "exptail":
            self.disk = ExptailDisk(grid, self.Ms, self.CR, Td=30, fracMd=0.1, meanmolw=self.meanmolw, index=-1.5)
        elif disk is not None:
            raise Exception("Unknown disk type")

        ### Make kmodel
        conds = [np.ones_like(self.rr, dtype=bool)]
        regs = [self.inenv]
        if self.outenv is not None:
            #conds += [self.grid.rr > self.outenv.rin_lim ]
            conds += [np.logical_and(self.outenv.rho > self.inenv.rho, self.grid.rr > self.outenv.rin_lim)]
            regs += [self.outenv]
        if self.disk is not None:
            conds += [self.disk.rho > self.inenv.rho ]
            regs += [self.disk]

        self.rho = np.select(conds[::-1], [r.rho for r in regs[::-1]])
        self.vr = np.select(conds[::-1], [r.vr for r in regs[::-1]])
        self.vt = np.select(conds[::-1], [r.vt for r in regs[::-1]])
        self.vp = np.select(conds[::-1], [r.vp for r in regs[::-1]])
        self.set_cylindrical_velocity()

    def save(self, filename="kmodel.pkl", filepath=None):
        if filepath is None:
            filepath = os.path.join(config.dp_run, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.to_pickle(self, filepath)
        logger.info(f'Saved : {filepath}\n')

####################################################################################################

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
        self.read_grid(grid)
        self.calc_kinematic_structure()
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self):
        csol = np.frompyfunc(self._sol_with_cubic, 2, 1)
        zeta = self.CR / self.rr
        self.mu0 = csol(self.mu, zeta).astype(np.float64)
        sin0 = np.sqrt(1 - self.mu0**2)
        mu_over_mu0 = 1 - zeta*(1 - self.mu0**2)
        v0 = np.sqrt(nc.G*self.Ms / self.rr)
        self.vr = - v0 * np.sqrt(1 + mu_over_mu0)
        self.vt = v0 * zeta*sin0**2*self.mu0/self.sin * np.sqrt(1 + mu_over_mu0)
        self.vp = v0 * sin0**2/self.sin * np.sqrt(zeta)
        rho = - self.Mdot/(4 * np.pi * self.rr**2 * self.vr * (1 + zeta*(3*self.mu0**2-1) ))
        self.rho = rho * self.cav_mask()

    def cav_mask(self):
        return np.where(self.mu0 <= np.cos(np.radians(self.cavangle_deg)), 1, 0)

    @staticmethod
    def _sol_with_cubic(m, zeta):
        allsols = np.round(cubicsolver.solve(zeta, 0, 1-zeta, -m).real, 8)
        sols = [ sol for sol in allsols if 0 <= sol <= 1 ]
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
        self.read_grid(grid)
        self.calc_kinematic_structure()
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self):
        vff = np.sqrt(2*nc.G*self.Mstar/self.rr)
        CB = self.CR / 2
        rho_prof = self.Mdot/(4*np.pi*self.rr**2*vff)
        self.rho = rho_prof * np.where(self.rr >= CB) * self.cav_mask()
        self.vr = - vff * np.sqrt( (1-CB/self.rr).clip(0) )
        self.vt = np.zeros_like(self.rr)
        self.vp = vff/np.sqrt(self.rr/CB)
        self.mu0 = self.mu

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
        #self.rin_lim = self.cs * self.Omega**2 * self.t**3 * 0.01
        self.read_grid(grid)
        self.calc_kinematic_structure()
        self.set_cylindrical_velocity()

    def calc_kinematic_structure(self):
        res = tsc.get_tsc(self.rc_ax, self.tc_ax, self.t, self.cs, self.Omega, mode="read")
        self.rho, self.vr, self.vt, self.vp = [v[:,:,np.newaxis] for v in res["vars"]]
        self.Delta = res["Delta"]
        P2 = 1 - 3/2 * self.sin**2
        self.rin_lim = self.cs * self.Omega**2 * self.t**3 / (1 + self.Delta * P2) * 0.4

    def cav_mask(self):
        return np.where(self.tt <= np.radians(self.cavangle_deg), 1, 0)

    @staticmethod
    def rinlim_func(model):
        return  model.cs * model.Omega**2 * model.t**3 * 0.1

    def maskfunc(self, model):
        return model.rr <  self.rinlim_func(model)

##################################################

class Disk(ModelBase):
    def calc_kinematic_structure_from_Sigma(self, Sigma):
        OmegaK = np.sqrt(nc.G*self.Ms/self.R**3)
        H = self.cs / OmegaK
        self.rho = Sigma/(np.sqrt(2*np.pi)*H) * np.exp(-0.5*(self.z/H)**2)
        self.vr = np.zeros_like(self.rho)
        self.vt = np.zeros_like(self.rho)
        self.vp = OmegaK*self.R

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
        self.read_grid(grid)
        Sigma = self.get_Sigma()
        self.calc_kinematic_structure_from_Sigma(Sigma)

    def get_Sigma(self):
        logger.error("Still Constructing!")
        exit()
        u = self.R/self.CR
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
        self.read_grid(grid)
        Mdisk = frac_Md * self.Mstar
        Sigma = self.get_Sigma(Mdisk, Rd, index)
        self.calc_kinematic_structure_from_Sigma(Sigma)
        self.set_cylindrical_velocity()

    def get_Sigma(self, Mdisk, Rd, ind):
        Sigma0 = Mdisk/(2*np.pi*Rd**2)/(1-2/np.e)
        return Sigma0 * (self.R/nc.au)**ind * np.exp(-(self.R/self.Rd)**(2+ind))

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
            for k,v in cls.__dict__.items():
                setattr(base, k, v)

def save_kmodel_hdf5_spherical(model, filename="flow.vtk", filepath=None):
    if filepath is None:
        filepath = os.path.join(config.dp_run, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    from evtk.hl import gridToVTK
    gridToVTK(filepath, model.ri_ax/nc.au, model.ti_ax, model.pi_ax,
              cellData = {"den" :model.rho, "ur" :model.vr, "uth" :model.vt, "uph" :model.vp})

def save_kmodel_hdf5_certesian(model, xi, yi, zi, filename="flow.vtk", filepath=None):
    if filepath is None:
        filepath = os.path.join(config.dp_run, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    from evtk.hl import gridToVTK
    from scipy.interpolate import interpn # , RectBivariateSpline, RegularGridInterpolator
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
    os.makedir(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    gridToVTK(filepath, xi/nc.au, yi/nc.au, zi/nc.au,
              cellData = {"den" :den_cert, "ux" :uux, "uy" :uuy, "uz" :uuz})
