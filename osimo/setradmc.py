#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import logging
import itertools
import numpy as np
import pandas as pd
import radmc3dPy.analyze as rmca
import radmc3dPy.data as rmcd
from scipy import interpolate, integrate
from osimo import tools, log
from osimo import nconst as nc
logger = log.set_logger(__name__, ini=True)

####################################################################################################
## Wrapper                                                                                         #
####################################################################################################

def set_radmc_input():
    self.radmc_controller.set_radmc_dir()
    self.radmc_controller.set_radmc_input(self.kmodel)
    self.radmc_controller.exe_radmc()

def set_radmc_with_inp(inp):
    global logger
    logger = log.set_logger(__name__, inp.fpath_log)
    rc = RadmcController()
    rc.set_radmc_dir(inp.dpath_eadmc)
    rc.set_KinematicModel(inp.kmodel_pkl)
    rc.set_radmc_input()
    TKmodel = rc.get_model()
    return TKmodel

####################################################################################################
## Class                                                                                           #
####################################################################################################
class Molecule:
    def __init__(self, name, abundance):
        self.name = name
        self.abun = abundance
        self.mass = rmca.readMol(fname=f"{dpath_radmc}/molecule_{name}.inp")
        self.region =

class RadmcController:
    #def __init__(self, inp):
    def __init__(self, dpath_radmc, dpath_radmc_storage):
        self.dpath_radmc = dpath_radmc
        self.dpath_radmc_storage = dpath_radmc_storage
        self.kmo = None
        self.nphot = None
        self.n_thread = None
        self.scattering_mode_max = None
        self.f_dg = None
        self.opac = None
        self.Lstar = None
        self.temp_mode = None
        self.calc_line = None
        self.mfrac_H2 = None
        self.T_const = None
        self.Rstar = None

    def set_radmc_dir(self, dpath):
        self.dpath_radmc = dpath
        if os.path.isdir(dpath):
            logger.info(f"Already exist {self.dpath_radmc}. Reuse this directory.")
        else:
            os.makedirs(dpath, exist_ok=True)

    def set_KinematicModel(self, kmodel):
        if isinstance(kmodel, str) and os.path.isfile(kmodel):
            self.kmo = pd.read_pickle(self.kmodel_pkl)
        if Kmodel is not None:
            self.kmo = kmodel
        else:
            raise Exception

    def set_parameters(self, nphoto=1e6, n_thread=1, scattering_mode_max=0, f_dg=0.01,
        opac="silicate", Lstar=nc.Lsun, mfrac_H2=0.74, T_const=10, Rstar=nc.Rsun)
        for k,v in locals().__dict__.items():
            if k != "self":
                setattr(self, k, v)

    def set_line(self, molname, molabun, iline):
        self.calc_line = True
        self.mol_name = molname
        self.mol_abun = molabun
        self.mol_iline = iline

    def set_radmc_input(self):
        cwd = os.getcwd()
        os.chdir(self.dpath_radmc)

        kmo = self.kmo
        rc, tc, pc = kmo.rc_ax, kmo.tc_ax, kmo.pc_ax
        ri, ti, pi = kmo.ri_ax, kmo.ti_ax, kmo.pi_ax
        rr, tt, pp = np.meshgrid(rc, tc, pc, indexing='ij')
        Nr, Nt, Np = rc.size, tc.size, pc.size
        ntot = rr.size
        rhog, vr, vt, vp = kmo.rho, kmo.vr, kmo.vt, kmo.vp
        if np.max(rhog) == 0:
            raise Exception('Zero density')
        rhod = rhog * self.f_dg
        n_mol = rhog / (2*nc.amu/self.mfrac_H2) * self.mol_abun
        n_mol = np.where(rr < self.mol_rlim, n_mol, 0)

        lam1, lam2, lam3, lam4 = 0.1, 7, 25, 1e4
        n12, n23, n34 = 20, 100, 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        lam = np.concatenate([lam12, lam23, lam34])
        nlam = lam.size
        Tstar = (self.Rstar/nc.Rsun)**(-0.5) * (self.Lstar/nc.Lsun)**0.25 * nc.Tsun

        ## Setting Radmc Parameters Below
        def join_list(l, sep='\n'):
            return sep.join(map(lambda x: f'{x:13.6e}', l))

        # set grid
        save_file('amr_grid.inp',
                  ['1', '0', '100', '0', '1 1 0', f'{Nr:d} {Nt:d} {Np:d}',
                   join_list(ri, sep=''), join_list(ti, sep=''), join_list(pi, sep='')])
        # set wavelength
        save_file('wavelength_micron.inp',
                  [f'{nlam:d}', join_list(lam)])
        # set a star
        save_file('stars.inp',
                  ['2', f'1 {nlam}', f'{self.Rstar:13.6e} 0 0 0 0',
                   join_list(lam), f'{-Tstar:13.6e}'])
        # set dust density
        save_file('dust_density.inp',
                  ['1', f'{ntot:d}', '1',
                   join_list(rhod.ravel(order='F')) ])
        # set_dust_opacity
        opacs = self.opac if isinstance(self.opac, list) else [self.opac]
        text_lines = ['2', f'{len(opacs)}', '===========']
        for op in opacs:
            self._copy_from_storage(f"dustkappa_{op}.inp")
            text_lines += ['1', '0', f'{op}', '----------']
        save_file('dustopac.inp', text_lines)
        # set_input
        param_dict={'nphot': int(self.nphot),
                    'scattering_mode_max': self.scattering_mode_max,
                    'iranfreqmode': 1,
                    'mc_scat_maxtauabs': 5.0,
                    'tgas_eq_tdust': 1}
        save_file('radmc3d.inp', [f'{k} = {v}'for k,v in param_dict.items()] )

        if self.calc_line:
            # set molcular number density
            save_file(f'numberdens_{self.mol_name}.inp',
                      ['1', f'{ntot:d}',
                       join_list(n_mol.ravel(order='F')) ])
            # set_mol_lines
            self._copy_from_storage(f"molecule_{self.mol_name}.inp")
            save_file('lines.inp',
                      ['2', '1', # number of molecules
                        f'{self.mol_name}  leiden 0 0 0']) # molname1 inpstyle1 iduma1 idumb1 ncol1
            # set_gas_velocity
            save_file('gas_velocity.inp',
                      ['1', f'{ntot:d}',
                       *[f'{_vr:13.6e} {_vt:13.6e} {_vp:13.6e}' for _vr, _vt, _vp
                         in zip(vr.ravel(order='F'), vt.ravel(order='F'), vp.ravel(order='F')) ] ])
            if self.temp_mode == 'mctherm':
                # remove gas_temperature.inp 6 dust_temperature.inp
                remove_file('gas_temperature.inp')
                remove_file('dust_temperature.dat')
            elif self.temp_mode == 'const':
                self._set_constant_temperature(rr, self.T_const)

            elif self.temp_mode == 'lambda':
                self._set_userdefined_temperature(inp.T_func, rr, tt, pp)
            else:
                raise Exception(f"Unknown temp_mode: {temp_mode}")

        os.chdir(cwd)
        ### END ###

    def _copy_from_storage(self, filename):
        if not os.path.isdir(self.dpath_radmc):
            raise Exception(f"Not found {self.dpath_radmc} (now in {os.getcwd()})")
        if not os.path.isdir(self.dpath_radmc_storage):
            raise Exception("Not found ", self.dpath_radmc_storage)
        fp1 = f"{self.dpath_radmc_storage}/{filename}"
        fp2 = f"{self.dpath_radmc}/{filename}"
        if not os.path.isfile(fp1):
            raise Exception
        if os.path.isfile(fp2):
            logger.debug(f"{fp2} already exist. Did nothing.")
            return
        shutil.copy2(fp1, fp2)

    def _set_constant_temperature(self, meshgrid, T_const):
        if Tenv is not None:
            T = T_const
            v_fwhm = np.sqrt(T*(16*np.log(2)*nc.kB)/self.mol_mass)
        #elif v_fwhm is not None:
        #    T = m_mol * v_fwhm**2 /(16 * np.log(2) * nc.kB )
        logger.info(f'Constant temperature is  {T}')
        logger.info(f'v_FWHM_kms is  {v_fwhm/nc.kms}')
        self.set_temperature(np.full_like(meshgrid, T), ntot)

    def set_userdefined_tfunc(self, func):
        self.tfunc = func

    def _set_userdefined_temperature(self):
        T = self.tfunc(self)
        T = T.clip(min=0.1, max=10000)
        self._set_temperature(T, ntot)

    def _set_temperature(self, temp, ntot):
        # set gas & dust temperature:
        #  Format_number, Nr of total cells, tgas
        save_file('gas_temperature.inp',
                  ['1', f'{ntot:d}', join_list(temp.ravel(order='F')) ])
        save_file('dust_temperature.inp',
                  ['1', f'{ntot:d}', '1', join_list(temp.ravel(order='F')) ])

    def exe_mctherm(self):
        if not os.path.isdir(self.dpath_radmc):
            raise Exception
        tools.exe(f'cd {self.dpath_radmc} ; radmc3d mctherm setthreads {self.n_thread} ')

    def read_radmc_data(self):
        cwd = os.getcwd()
        os.chdir(dpath_radmc)
        self.data = rmca.readData(ddens=use_ddens, dtemp=use_dtemp, gdens=use_gdens, gtemp=use_gtemp, gvel=use_gvel, ispec=ispec)
        os.chdir(cwd)
        return self.data

    def get_model(self):
        return ThermalKinematicModel(dpath_radmc=self.dpath_radmc, ispec=self.mol_name, f_dg=self.f_dg)

class ThermalKinematicModel:
    def __init__(self, dpath_radmc=None, ispec=None, f_dg=None, tgas_eq_tdust=True):
        ## getting data
        cwd = os.getcwd()
        os.chdir(dpath_radmc)
        rmcdata = rmcd.radmc3dData()

        self.r_ax = rmcdata.grid.x
        self.t_ax = rmcdata.grid.y
        self.rrc, self.ttc = np.meshgrid(self.r_ax, self.t_ax, indexing='ij')
        self.RR = self.rrc * np.sin(self.ttc)
        self.zz = self.rrc * np.cos(self.ttc)

        rmcdata.readDustDens()
        rmcdata.readDustTemp()
        rmcdata.readGasTemp()
        rmcdata.readGasVel()
        rmcdata.readGasDens(ispec=ispec)
        self.rhodust = self.get_value(rmcdata, "rhodust")
        self.dtemp = self.get_value(rmcdata, "dusttemp")
        self.gtemp = self.get_value(rmcdata, "gastemp") if not tgas_eq_tdust else self.dtemp
        self.vr = self.get_value(rmcdata, "gasvel", index=0)
        self.vt = self.get_value(rmcdata, "gasvel", index=1)
        self.vp = self.get_value(rmcdata, "gasvel", index=2)
        self.ndens_mol = self.get_value(rmcdata, "ndens_mol")
        self.rhogas = self.rhodust/f_dg
        os.chdir(cwd)

    def get_value(self, rmcdata, key, index=0):
        val = getattr(rmcdata, key)
        if len(val) != 0:
            logger.debug(f"Setting {key}")
            return val[:,:,:,index]
        else:
            logger.info(f"Tried to set {key} but not found.")
            return None


####################################################################################################
## Functions                                                                                       #
####################################################################################################

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f'Removed: {file_path}')

def save_file(file_path, text_lines):
    # os.path.dirname(os.path.abspath(file_path))
    with open(file_path, 'w+') as f:
        f.write('\n'.join(text_lines))
        logger.info(f'Saved:  {f.name}')


def del_mctherm_files(dpath):
    remove_file(f'{dpath}/radmc3d.out')


class Stream:
    def __init__(self, r_ax, t_ax, vr, vt):
        self.r_ax = r_ax
        self.t_ax = t_ax
        self.slines = []
        self.val_list = []

    def set_pos0list(self, r0_list, theta0_list):
        self.pos0list = [(r0, th0) for r0 in r0_list for th0 in theta0_list]

    def set_vfield(self, r_ax, t_ax, vr, vt):
        self.vr_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vr, bounds_error=False, fill_value=None)
        self.vt_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vt, bounds_error=False, fill_value=None)

    def add_value(self, name, val, unit):
        valfunc = interpolate.RegularGridInterpolator((self.r_ax, self.t_ax), val, bounds_error=False, fill_value=None)
        self.val_list.append([ name, valfunc, unit])

    def calc_streamlines(self, dpath):
        sl = Streamline(self.r_ax, self.t_ax, self.vr_field, self.vt_field, self.t_span)
        for name, valfunc, unit in self.val_list:
            sl.add_value(name, valfunc, unit)

        for _pos0 in self.pos0_list:
            sl.calc_streamline(_pos0)
            sl.save_data(dpath)

class Streamline:
    def __init__(self, r_ax, th_ax, vrf, vtf, t_span=(0, 1e30)):
        self.name = "{pos0[0]/nc.au:.2f}_{np.deg2rad(pos0[1]):.2f}"
        self.vr_field = vrf
        self.vt_field = vtf
        self.vnames = []
        self.values = []
        self.vunits = []
        self.t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), 600)
        self.set_vfield(r_ax, t_ax, vr, vt)
        #self.calc_streamline()

    def add_value(self, name, interpfunc, unit):
        self.vnames.append(name)
        self.vfuncs.append(interpfunc)
        self.vunits.append(unit)

    def calc_streamline(self, pos0):
       if pos0[0] > self.r_ax[-1]:
            looger.info(f'Too large position:r0 = {pos0[0]/nc.au} au. r0 must be less than {r_ax[-1]/nc.au} au. I use r0 = {r_ax[-1]/nc.au} au instead of r0 = {pos0[0]/nc.au} au')
            pos0 = [self.r_ax[-1], pos0[1]]

        def func(t, pos, hit_flag=0):
            if hit_midplane(t, pos) < 0:
                hit_flag = 1
            vr = self.vr_field(pos)
            vt = self.vt_field(pos)
            return np.array([vr, vt/pos[0]])

        def hit_midplane(t, pos):
            return np.pi/2 - pos[1]
        hit_midplane.terminal = True

        pos = integrate.solve_ivp(func, (self.t_eval[0], self.t_eval[-1]), pos0, method='RK45',
                                  events=hit_midplane, t_eval=self.t_eval[1:-1], rtol=1e-8)
        pos.R = pos.y[0] * np.sin(pos.y[1])
        pos.z = pos.y[0] * np.cos(pos.y[1])
        self.pos = pos
        self.values = [f(pos) for f in self.vfuncs]

    def save_data(self, dpath):
        vlist = [f"{v}["u"]" for v, u in zip(self.vnames, self.sunits)]
        header = " ".join('t[s]', "R[cm]", "z[cm]", *vlist)
        stream_data = np.stack((self.pos.t, self.pos.R, self.pos.z, *self.values), axis=-1)
        np.savetxt(f'{dpath}/stream_{self.name}.txt', stream_data, header=header)

