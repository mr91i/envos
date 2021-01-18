#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import logging
import itertools
import numpy as np
import pandas as pd
import radmc3dPy.analyze as rmca
from scipy import interpolate, integrate
from envos import tools, log
from envos import nconst as nc
from envos import config
from envos.model import ThermalKinematicModel

logger = log.set_logger(__name__, ini=True)

####################################################################################################
## Wrapper                                                                                         #
####################################################################################################

# def set_radmc_with_inp(inp):
#     global logger
#     logger = log.set_logger(__name__, inp.fpath_log)
#     rc = RadmcController()
#     rc.set_radmc_dir(inp.dpath_eadmc)
#     rc.set_model(inp.kmodel_pkl)
#     rc.set_radmc_input()
#     TKmodel = rc.get_model()
#     return TKmodel

####################################################################################################
## Class                                                                                           #
####################################################################################################
class RadmcController:
    #def __init__(self, inp):
    def __init__(self, model=None, dpath_run=None, dpath_radmc=None, dpath_storage=None):
        self.dpath_run = dpath_run or config.dp_run # global?
        self.dpath_radmc = dpath_radmc or config.dp_radmc # local
        self.dpath_storage = dpath_storage or config.dp_storage # local
        self.set_dirs(self.dpath_radmc, self.dpath_run, self.dpath_storage)

        self.model = None
        self.nphot = None
        self.n_thread = None
        self.scattering_mode_max = None
        self.f_dg = None
        self.opac = None
        self.Lstar = None
        self.temp_mode = None
        self.calc_line = calc_line
        self.mfrac_H2 = None
        self.T_const = None
        self.Rstar = None

        if model is not None:
            self.set_model(model)

    def set_dirs(self, radmc=None, run=None, storage=None):
        if radmc is not None:
            self.dpath_radmc = radmc
            os.makedirs(radmc, exist_ok=True)
        if run is not None:
            self.dpath_run = run
            os.makedirs(run, exist_ok=True)
        if storage is not None:
            self.dpath_storage = storage
            os.makedirs(storage, exist_ok=True)

    def set_model(self, model):
        if isinstance(model, str) and os.path.isfile(model):
            self.model = pd.read_pickle(self.model_pkl)
        if model is not None:
            self.model = model
        else:
            raise Exception

    def set_parameters(self, nphot=1e6, n_thread=1,
        scattering_mode_max=0,
        mc_scat_maxtauabs=5.0,
        f_dg=0.01,
        opac="silicate",
        Lstar_Lsun=1,
        mfrac_H2=0.74,
        T_const=10,
        Rstar_Rsun=1,
        temp_mode="mctherm",
        molname=None,
        molabun=None,
        iline=None,
        mol_rlim=1000):

        args = locals()
        args.pop("self")
        for k,v in args.items():
            setattr(self, k, v)
        if iline is not None:
            self.calc_line = True


    def set_radmc_input(self):
        cwd = os.getcwd()
        os.chdir(self.dpath_radmc)

        md = self.model
        rc, tc, pc = md.rc_ax, md.tc_ax, md.pc_ax
        ri, ti, pi = md.ri_ax, md.ti_ax, md.pi_ax
        rr, tt, pp = np.meshgrid(rc, tc, pc, indexing='ij')
        Nr, Nt, Np = rc.size, tc.size, pc.size
        ntot = rr.size
        rhog, vr, vt, vp = md.rho, md.vr, md.vt, md.vp
        if np.max(rhog) == 0:
            raise Exception('Zero density')
        rhod = rhog * self.f_dg
        n_mol = rhog / (2*nc.amu/self.mfrac_H2) * self.molabun
        n_mol = np.where(rr < self.mol_rlim, n_mol, 0)

        lam1, lam2, lam3, lam4 = 0.1, 7, 25, 1e4
        n12, n23, n34 = 20, 100, 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        lam = np.concatenate([lam12, lam23, lam34])
        nlam = lam.size
        Tstar = (self.Rstar_Rsun)**(-0.5) * (self.Lstar_Lsun)**0.25 * nc.Tsun

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
                  ['2', f'1 {nlam}', f'{self.Rstar_Rsun*nc.Rsun:13.6e} 0 0 0 0',
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
                    'mc_scat_maxtauabs': self.mc_scat_maxtauabs,
                    'tgas_eq_tdust': 1}
        save_file('radmc3d.inp', [f'{k} = {v}'for k,v in param_dict.items()] )

        if self.calc_line:
            # set molcular number density
            save_file(f'numberdens_{self.molname}.inp',
                      ['1', f'{ntot:d}',
                       join_list(n_mol.ravel(order='F')) ])

            # set_mol_lines
            self._copy_from_storage(f"molecule_{self.molname}.inp")
            save_file('lines.inp',
                      ['2', '1', # number of molecules
                        f'{self.molname}  leiden 0 0 0']) # molname1 inpstyle1 iduma1 idumb1 ncol1

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

        if not os.path.isdir(self.dpath_storage):
            raise Exception("Not found ", self.dpath_storage)

        fp1 = f"{self.dpath_storage}/{filename}"
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
            raise Exception("Dir is not found")
        logger.info(f"cd {self.dpath_radmc} ; radmc3d mctherm setthreads {self.n_thread}")
        tools.exe(f'cd {self.dpath_radmc} ; radmc3d mctherm setthreads {self.n_thread} ')

    def get_model(self):
        return ThermalKinematicModel(dpath_radmc=self.dpath_radmc, ispec=self.molname, f_dg=self.f_dg)

####################################################################################################
## Functions                                                                                       #
####################################################################################################

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f'Removed: {file_path}')

def save_file(file_path, text_lines):
    print("saving:", os.getcwd(), file_path)
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # os.path.dirname(os.path.abspath(file_path))
    with open(file_path, 'w+') as f:
        f.write('\n'.join(text_lines))
        logger.info(f'Saved:  {f.name}')


def del_mctherm_files(dpath):
    remove_file(f'{dpath}/radmc3d.out')


