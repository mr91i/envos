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
from pmodes import cst, tools
print("Setradmc: Set logger using ", __name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#######################

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

## Wrapper
def set_radmc_input():
        self.radmc_controller.set_radmc_dir()
        self.radmc_controller.set_radmc_input(self.Kinemo)
        self.radmc_controller.exe_radmc()
####

class RadmcController:
    def __init__(self, inp):

        ## Read inp & Initialize
        self.dpath_radmc = inp.dpath_radmc
        self.kinemo_pkl = inp.kinemo_pkl
        self.dpath_radmc_storage = inp.dpath_radmc_storage


        self.nphot = inp.nphot
        self.f_dg = inp.f_dg
        self.mol_abun = inp.mol_abun
        self.mol_name = inp.mol_name
        self.mol_mass = inp.mol_mass
        self.mfrac_H2 = inp.mfrac_H2
        self.mol_rlim = inp.mol_rlim
        self.T_const = inp.T_const
        self.opac = inp.opac
        self.temp_mode = inp.temp_mode
        self.calc_line = inp.calc_line
        self.scattering_mode_max = inp.scattering_mode_max
        self.Rstar = inp.Rstar
        self.Lstar = inp.Lstar
        self.n_thread = inp.n_thread

    def initialize(self, kinemo):
        self.set_radmc_dir()
        self.set_KinematicModel(kinemo)
        self.set_radmc_input()

    def set_radmc_dir(self):
        if os.path.isdir(self.dpath_radmc):
            logger.info(f"Already exist {self.dpath_radmc}. Reuse this directory.")
        else:
            os.makedirs(self.dpath_radmc, exist_ok=True)

    def copy_from_storage(self, filename):
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

    def set_KinematicModel(self, Kinemo):
        if Kinemo is not None:
            self.kmo = Kinemo
        elif os.path.isfile(self.kinemo_pkl):
            self.kmo = pd.read_pickle(self.kinemo_pkl)
        else:
            raise Exception

    def set_radmc_input_without_mctherm(self):
        cwd = os.getcwd()
        os.chdir(self.dpath_radmc)
        kmo = self.kmo
        rc, tc, pc = kmo.rc_ax, kmo.tc_ax, kmo.pc_ax
        rr, tt, pp = np.meshgrid(rc, tc, pc, indexing='ij')
        if temp_mode == 'const':
            self.set_constant_temperature(rr, self.T_const)
        elif temp_mode == 'lambda':
            self.set_userdefined_temperature(inp.T_func, rr, tt, pp)

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
        n_mol = rhog / (2*cst.amu/self.mfrac_H2) * self.mol_abun
        n_mol = np.where(rr < self.mol_rlim, n_mol, 0)

        lam1, lam2, lam3, lam4 = 0.1, 7, 25, 1e4
        n12, n23, n34 = 20, 100, 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        lam = np.concatenate([lam12, lam23, lam34])
        nlam = lam.size

        Tstar = (self.Rstar/cst.Rsun)**(-0.5) * (self.Lstar/cst.Lsun)**0.25 * cst.Tsun

        ## Setting Radmc Parameters Below
        def join_list(l, sep='\n'):
            return sep.join(map(lambda x: f'{x:13.6e}', l))

        # set grid
        # Format number, grid style, Coordinate type (100=spherical), gridinfo
        # (incl x, incl y, incl z),  grid size
        # ri[0] ... , thetai[:] ..., phii[:] ...
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

        if self.calc_line:
            # set molcular number density
            save_file(f'numberdens_{self.mol_name}.inp',
                      ['1', f'{ntot:d}',
                       join_list(n_mol.ravel(order='F')) ])

            # set_mol_lines
            self.copy_from_storage(f"molecule_{self.mol_name}.inp")
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
                self.set_constant_temperature(rr, self.T_const)

            elif self.temp_mode == 'lambda':
                self.set_userdefined_temperature(inp.T_func, rr, tt, pp)
            else:
                raise Exception(f"Unknown temp_mode: {temp_mode}")


        # set_dust_opacity
        opacs = self.opac if isinstance(self.opac, list) else [self.opac]
        text_lines = ['2', f'{len(opacs)}', '===========']
        for op in opacs:
            self.copy_from_storage(f"dustkappa_{op}.inp")
            text_lines += ['1', '0', f'{op}', '----------']
        save_file('dustopac.inp', text_lines)

        # set_input
        param_dict={'nphot': int(self.nphot),
                    'scattering_mode_max': self.scattering_mode_max,
                    'iranfreqmode': 1,
                    'mc_scat_maxtauabs': 5.0,
                    'tgas_eq_tdust': 1}
        save_file('radmc3d.inp', [f'{k} = {v}'for k,v in param_dict.items()] )

        os.chdir(cwd)
        ### END ###

    #def set_constant_temperature(self, meshgrid, Tenv, v_fwhm):
    def set_constant_temperature(self, meshgrid, T_const):
        if Tenv is not None:
            T = T_const
            v_fwhm = np.sqrt(T*(16*np.log(2)*cst.kB)/self.mol_mass)
        #elif v_fwhm is not None:
        #    T = m_mol * v_fwhm**2 /(16 * np.log(2) * cst.kB )
        logger.info(f'Constant temperature is  {T}')
        logger.info(f'v_FWHM_kms is  {v_fwhm/cst.kms}')
        self.set_temperature(np.full_like(meshgrid, T), ntot)

    def set_userdefined_temperature(self, func, rr, tt, pp):
        T = func(rr, tt, pp)
        T = T.clip(min=0.1, max=10000)
        self.set_temperature(T, ntot)

    def set_temperature(self, temp, ntot):
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


# def get_thermal_structure(data):



#########################################################################################

##################################################################################################################
#class RadmcData:
#class PhysicalModel(ModelBase):
#    def __init__(self, inp=None):
#        super().__init__(name)
#        self.grid = Grid(inp)
#        self.kinemo = Kinemo(inp, self.grid)
#        self.radmc_controller = RadmcController(dpath_radmc, inp)
#        self.thermo = Thermo(inl, self.grid#)
#
#    def calc_KinematicStr(self):
#        self.kinemo.calc#()
#        self.kinem_str = self.kinemo.get_structure()
#
#    def calc_ThermalStr(#self#):
#        self.radmc_controller.initialize(self.Kinemo)
#        self.radmc_controller.exe_radmc()
#        self.therm_str = self.radmc_controller.get_structure()

class ThermalKinematicModel:
#    def __init__(self, dpath_radmc=None, dpath_fig=None,
#                 use_ddens=True, use_gdens=True, use_gtemp=True,
#                 use_dtemp=True, use_gvel=True,
#                 ispec=None, mol_abun=0, opac='', autoplot=True, plot_tau=False, fn_model=None):
    def __init__(self, dpath_radmc=None, ispec=None, f_dg=None, tgas_eq_tdust=True):

        # tools.set_arguments(self, locals() )
        #D = pd.read_pickle(dpath_radmc+'/'+fn_model)
        #self.D = namedtuple('Model', D.keys())(*D.values())
        # D = pd.read_pickle(dpath_radmc+'/'+fn_model)

        ## getting data
        cwd = os.getcwd()
        os.chdir(dpath_radmc)
        print("now in ", os.getcwd())
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
            print(f"Try to set {key} but not found")
            return None

        # self.calc_gas_trajectries( )

    def calc_gas_trajectries(self, r0, theta0, t_span):
        pos0_list = [(r0, th0) for r0 in r0_list   for th0 in theta0_list]
        pos_list = [trace_streamline_meridional2d(self.r_ax, self.t_ax, self.vr, self.vt, t_span, pos0)
                     for pos0 in pos0_list]
        self.pos_list = pos_list
        dens_func = interpolate.RegularGridInterpolator((self.r_ax, self.t_ax), self.rhogas, bounds_error=False, fill_value=None)
        temp_func = interpolate.RegularGridInterpolator((self.r_ax, self.t_ax), self.gtemp, bounds_error=False, fill_value=None)
        for i, pos in enumerate(pos_list):
            stream_pos = pos.y.T
            dens_stream = dens_func(stream_pos)
            temp_stream = temp_func(stream_pos)
            stream_data = np.stack((pos.t, pos.R, pos.z, dens_stream, temp_stream), axis=-1)
            np.savetxt(f'{dpath_fig}/stream_{i:d}.txt', stream_data, header='t[s] R[cm] z[cm] rho_gas[g cm^-3] T[K]')


def trace_sreamline_meridional2d(r_ax, t_ax, vr, vt, t_span, pos0):
    # There are some choice for coordination, but i gave up genelarixation for the code simplicity.
    # input axis : rth
    # velocity   : rth
    # return pos : rth

    vr_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vr, bounds_error=False, fill_value=None)
    vt_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vt, bounds_error=False, fill_value=None)

    def func(t, pos, hit_flag=0):
        if pos[0] > r_ax[-1]:
            raise Exception(f'Too large position. r must be less than {r_ax[-1]/cst.au} au.')

        if hit_midplane(t, pos) < 0:
            hit_flag = 1
        r, th = pos[0], pos[1]
        vr = vr_field((r, th))
        vt = vt_field((r, th))
        return np.array([vr, vt/r])

    def hit_midplane(t, pos):
        return np.pi/2 - pos[1]
    hit_midplane.terminal = True

    t_trace = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), 600)
    if pos0[0] > r_ax[-1]:
        print(f'Too large position:r0 = {pos0[0]/cst.au} au. r0 must be less than {r_ax[-1]/cst.au} au. I use r0 = {r_ax[-1]/cst.au} au instead of r0 = {pos0[0]/cst.au} au')
        pos0 = [r_ax[-1], pos0[1]]

    #pos = integrate.solve_ivp(func, t_span, pos0, method='BDF', events=hit_midplane, t_eval=t_trace[1:-1])
    pos = integrate.solve_ivp(func, t_span, pos0, method='RK45', events=hit_midplane, rtol=1e-8)
    pos.R = pos.y[0] * np.sin(pos.y[1])
    pos.z = pos.y[0] * np.cos(pos.y[1])
    return pos

