#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d, interp1d, griddata
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca
import subprocess
from scipy import interpolate, integrate

from pmodes.header import inp, dpath_home, dpath_radmc, dpath_fig
from pmodes import cst, tools
import pmodes.myplot as mp
# msg = tools.Message(__file__)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#######################

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f'Removed: {file_path}')

def save_file(file_path, text_lines):
    with open(file_path, 'w+') as f:
        f.write('\n'.join(text_lines))
        logger.info(f'Saved:  {f.name}')

def exe_radmc_therm():
    tools.exe('cd %s ; radmc3d mctherm setthreads 16 ' % dpath_radmc )

def del_mctherm_files():
    remove_file(dpath_radmc+'/radmc3d.out')


def set_radmc_parameters(dpath_radmc,
                 nphot=1000000,
                 r_in=1*cst.au, r_out=1000*cst.au, nr=128,
                 theta_in=0, theta_out=0.5*np.pi, ntheta=64,
                 phi_in=0, phi_out=0, nphi=1,
                 fdg=0.01, mol_abun=1e-7, mol_name='c18o',
                 m_mol=28*cst.amu, mol_radius=None, v_fwhm=0.5*cst.kms, Tenv=None,
                 opac='silicate',
                 temp_mode='mctherm', line=True, turb=True, lowreso=False, subgrid=True,
                 autoset=True, fn_model_pkl=None, fn_radmcset_pkl=None,
                 scattering_mode_max=0, temp_lambda=None,
                 Mstar=cst.Msun, Rstar=cst.Rsun, Lstar=cst.Lsun, **kwargs):

    cwd = os.getcwd()
    os.chdir(dpath_radmc)

    kin = pd.read_pickle(fn_model_pkl)
    kin = namedtuple('Model', kin.keys())(*kin.values())

    rc = kin.r_ax
    thetac = kin.th_ax
    phic = kin.ph_ax

    ri = kin.ri
    thetai = kin.thetai
    phii = kin.phii
    rr, tt, pp = np.meshgrid(rc, thetac, phic, indexing='ij')

    nr = rc.size
    ntheta = thetac.size
    nphi = phic.size
    ntot = rr.size

    rhog = kin.rho
    if np.max(rhog) == 0:
        raise Exception('Zero density')
    vr = kin.ur
    vth = kin.uth
    vph = kin.uph
    rhod = rhog * fdg

    n_mol = rhog / (2.34*cst.amu) * mol_abun
    n_mol = np.where(rr < mol_radius, n_mol, 0)

    if temp_mode == 'const':
        if Tenv is not None:
            T = Tenv
            v_fwhm = np.sqrt(T*(16*np.log(2)*cst.kB)/m_mol)
        else:
            v_fwhm = v_fwhm
            T = m_mol * v_fwhm**2 /(16 * np.log(2) * cst.kB )
        logger.info(f'Constant temperature is  {T}')
        logger.info(f'v_FWHM_kms is  {v_fwhm/cst.kms}')
        tgas = T * np.ones_like(rhog)
        tdust = tags

    if temp_mode == 'lambda':
        tgas = temp_lambda(rr/cst.au)* (rr/(1000*cst.au))**(-1.5)
        tgas = tgas.clip(min=0.1, max=10000)
        tdust = tgas

    lam1 = 0.1
    lam2 = 7
    lam3 = 25
    lam4 = 1e4
    n12 = 20
    n23 = 100
    n34 = 30
    lam12 = np.logspace(np.log10(lam1), np.log10(lam2), n12, endpoint=False)
    lam23 = np.logspace(np.log10(lam2), np.log10(lam3), n23, endpoint=False)
    lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
    lam = np.concatenate([lam12, lam23, lam34])
    nlam = lam.size

    Tstar = (Lstar/cst.Lsun)**0.25 * cst.Tsun


    ## Setting Radmc Parameters Below

    def join_list(l, sep='\n'):
        return sep.join(map(lambda x: f'{x:13.6e}', l))

    # set grid
    # Format number, grid style, Coordinate type (100=spherical), gridinfo
    # (incl x, incl y, incl z),  grid size
    # ri[0] ... , thetai[:] ..., phii[:] ...
    save_file('amr_grid.inp',
              ['1', '0', '100', '0', '1 1 0', f'{nr:d} {ntheta:d} {nphi:d}',
               join_list(ri, sep=''), join_list(thetai, sep=''), join_list(phii, sep='')])

    # set wavelength
    save_file('wavelength_micron.inp',
              [f'{nlam:d}', join_list(lam)])

    # set a star
    save_file('stars.inp',
              ['2', f'1 {nlam}', f'{Rstar:13.6e} {Mstar:13.6e} 0 0 0',
               join_list(lam), f'{-Tstar:13.6e}'])

    # set dust density
    save_file('dust_density.inp',
              ['1', f'{ntot:d}', '1',
               join_list(rhod.ravel(order='F')) ])

    if line:
        # set molcular number density
        save_file(f'numberdens_{mol_name}.inp',
                  ['1', f'{ntot:d}',
                   join_list(n_mol.ravel(order='F')) ])

        # set_mol_lines
        save_file('lines.inp',
                  ['2', '1', # number of molecules
                    f'{mol_name}  leiden 0 0 0']) # molname1 inpstyle1 iduma1 idumb1 ncol1

        # set_gas_velocity
        save_file('gas_velocity.inp',
                  ['1',  # Format number
                   f'{ntot:d}', # Nr of cells
                   *[f'{vr[ir, it, ip]:13.6e} {vr[ir, it, ip]:13.6e} {vr[ir, it, ip]:13.6e}\n'
                     for ip, it, ir in itertools.product(range(nphi), range(ntheta), range(nr))]
                   ])

    if temp_mode == 'mctherm':
        # remove gas_temperature.inp  6 dust_temperature.inp
        remove_file('gas_temperature.inp')
        remove_file('dust_temperature.dat')

    elif (self.temp_mode == 'const') or (self.temp_mode == 'lambda'):
        # set gas temperature:
        #  Format_number, Nr of total cells, tgas
        save_file('gas_temperature.inp',
                  ['1', f'{ntot:d}', join_list(tgas.ravel(order='F')) ])

        # set dust temperature:
        #  Format_number, Nr of total cells, Format_number , tgas
        save_file('dust_temperature.inp',
                  ['1', f'{ntot:d}', '1', join_list(tdust.ravel(order='F')) ])
    else:
        raise Exception

    # set_dust_opacity
    opacs = opac if isinstance(opac,list) else [opac]
    text_lines = ['2', # Format number of this file
                  f'{len(opacs)}', # Number of dust species
                  '===========']
    for op in opacs:
        text_lines += ['1', # Way in which this dust species is read
                       '0', # 0=Thermal grain
                       f'{op}',
                       '----------'] # Extension of name of dustkappa_***.inp file
    save_file('dustopac.inp', text_lines)

    # set_input
    param_dict={'nphot': int(nphot),
                'scattering_mode_max': scattering_mode_max,
                'iranfreqmode': 1,
                'mc_scat_maxtauabs': 5.0,
                'tgas_eq_tdust': 1}
    save_file('radmc3d.inp', [f'{k} = {v}'for k,v in param_dict.items()] )

    os.chdir(cwd)
#########################################################################################

##################################################################################################################
#class RadmcData:
class PhysicalModel:
    def __init__(self, dpath_radmc=None, dpath_fig=None,
                 use_ddens=True, use_gdens=True, use_gtemp=True,
                 use_dtemp=True, use_gvel=True,
                 ispec=None, mol_abun=0, opac='', autoplot=True, plot_tau=False, fn_model=None):

        tools.set_arguments(self, locals() )
        D = pd.read_pickle(dpath_radmc+'/'+fn_model)
        self.D = namedtuple('Model', D.keys())(*D.values())

        ## getting data
        cwd = os.getcwd()
        os.chdir(dpath_radmc)
        data = rmca.readData(ddens=use_ddens, dtemp=use_dtemp, gdens=use_gdens, gtemp=use_gtemp, gvel=use_gvel, ispec=ispec)
        os.chdir(cwd)

        ## redefining variables
        self.r_ax = data.grid.x
        self.th_ax = data.grid.y
        self.xauc = data.grid.x/cst.au
        self.rrc, self.ttc = np.meshgrid(self.xauc, data.grid.y, indexing='ij')
        self.RR = self.rrc * np.sin(self.ttc)
        self.zz = self.rrc * np.cos(self.ttc)

        self.rhodust = data.rhodust[:,:,0,0]
        self.rhogas = self.rhodust/inp.radmc.fdg

        if use_gtemp:
            print(data.dusttemp.shape)
            try:
                self.gtemp = data.gastemp[:,:,0,0]
            except:
                self.gtemp = data.dusttemp[:,:,0,0]

        if use_gdens and ispec:
            self.ndens_mol = data.ndens_mol[:,:,0,0]

        if use_gvel:
            self.vr = data.gasvel[:,:,0,0]
            self.vt = data.gasvel[:,:,0,1]
            self.vp = data.gasvel[:,:,0,2]

        if use_gdens and use_gtemp:
            # Chemical lifetime
            self.t_dest = self.cch_lifetime( self.ndens_mol, self.mol_abun, self.gtemp)
            self.t_dyn = 5.023e6 * np.sqrt( self.rrc**3 /0.18 ) ## sqrt(au^3/GMsun) = 5.023e6

        if plot_tau:
            self.tau = self.calc_tau_surface(tau=1e-4, npix=100, sizeau=500, incl=70, phi=0, posang=0, n_thread=10, lamb=1249)
            print(self.tau.shape,np.min(self.tau), np.max(self.tau))

        self.calc_gas_trajectries()

    @staticmethod
    def cch_lifetime(nmol, nabun, T):
        k = 1.2e-11*np.exp(-998/T)
        return 1/( (nmol/nabun+1e-100)* k)

    def calc_tau_surface(self, tau=1, npix=100, sizeau=500, incl=90, phi=0, posang=85, n_thread=10, iline=None, lamb=None):
        common = f'incl {incl} phi {phi} setthreads {n_thread:d} '
        wl = f'iline {iline} ' if iline is not None else f'lambda {lamb} '
        cmd = f'radmc3d tausurf {tau} npix {npix} sizeau {sizeau} fluxcons ' + common + wl
        print(cmd)
        os.chdir(dpath_radmc)
        subprocess.call(cmd, shell=True)
        data = rmci.readImage()
        self.imx = data.x
        self.imy = data.y
        print(vars(data))
        return data.image[:,:,0].T.clip(0)

    def calc_gas_trajectries(self):

        r_ew = self.D.cs * self.D.t
        print(f'r_ew is {r_ew/cst.au}')
        pos0_list = [(r_ew, np.pi/180*80), (r_ew, np.pi/180*70), (r_ew, np.pi/180*60), (r_ew, np.pi/180*50)]

        pos_list = [trace_particle_2d_meridional(self.r_ax, self.th_ax, self.vr, self.vt, (0.1*cst.yr, 1e7*cst.yr), pos0)
                     for pos0 in pos0_list]
        self.pos_list = pos_list

        dens_func = interpolate.RegularGridInterpolator((self.r_ax, self.th_ax), self.rhogas, bounds_error=False, fill_value=None)
        temp_func = interpolate.RegularGridInterpolator((self.r_ax, self.th_ax), self.gtemp, bounds_error=False, fill_value=None)

        for i, pos in enumerate(pos_list):
            stream_pos = pos.y.T
            dens_stream = dens_func(stream_pos)
            temp_stream = temp_func(stream_pos)
            stream_data = np.stack((pos.t, pos.R, pos.z, dens_stream, temp_stream), axis=-1)
            np.savetxt(f'{dpath_fig}/stream_{i:d}.txt', stream_data, header='t[s] R[cm] z[cm] rho_gas[g cm^-3] T[K]')

def trace_particle_2d_meridional(r_ax, th_ax, vr, vth, t_span, pos0):
    # There are some choice for coordination, but i gave up genelarixation for the code simplicity.
    # input axis : rth
    # velocity   : rth
    # return pos : rth

    vr_field = interpolate.RegularGridInterpolator((r_ax, th_ax), vr, bounds_error=False, fill_value=None)
    vth_field = interpolate.RegularGridInterpolator((r_ax, th_ax), vth, bounds_error=False, fill_value=None)

    def func(t, pos, hit_flag=0):
        if pos[0] > r_ax[-1]:
            raise Exception(f'Too large position. r must be less than {r_ax[-1]/cst.au} au.')

        if hit_midplane(t, pos) < 0:
            hit_flag = 1
        r, th = pos[0], pos[1]
        vr = vr_field((r, th))
        vth = vth_field((r, th))
        return np.array([vr, vth/r])

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

if __name__ == '__main__':
    calc_temp = 1
    if calc_temp:
        #srmc = SetRadmc(dpath_radmc, **vars(inp.radmc))
        set_radmc_parameters(dpath_radmc, **vars(inp.radmc))
        # or you can input parmeters directly.
        if inp.radmc.temp_mode=='mctherm':
            exe_radmc_therm()
        else:
            del_mctherm_files()

    if inp.radmc.plot:
        rmc_data = RadmcData(dpath_radmc=dpath_radmc, dpath_fig=dpath_fig, ispec=inp.radmc.mol_name, mol_abun=inp.radmc.mol_abun, opac=inp.radmc.opac, fn_model=inp.radmc.fn_model_pkl)
        plot_radmc_data(rmc_data)


