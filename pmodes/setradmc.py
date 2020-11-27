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
from scipy import interpolate

from pmodes.header import inp, dn_home, dn_radmc, dn_fig
from pmodes import cst, mytools
import pmodes.myplot as mp
from pmodes.mkmodel import trace_particle_2d_meridional
# msg = mytools.Message(__file__)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#######################

def main():
    calc_temp = 1
    if calc_temp:
        #srmc = SetRadmc(dn_radmc, **vars(inp.radmc))
        set_radmc_parameters(dn_radmc, **vars(inp.radmc))
        # or you can input parmeters directly.
        if inp.radmc.temp_mode=="mctherm":
            exe_radmc_therm()
        else:
            del_mctherm_files()

    if inp.radmc.plot:
        rmc_data = RadmcData(dn_radmc=dn_radmc, dn_fig=dn_fig, ispec=inp.radmc.mol_name, mol_abun=inp.radmc.mol_abun, opac=inp.radmc.opac, fn_model=inp.radmc.fn_model_pkl)
        plot_radmc_data(rmc_data)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Removed: {file_path}")

def save_file(file_path, text_lines):
    with open(file_path, 'w+') as f:
        f.write("\n".join(text_lines))
        logger.info(f"Saved:  {f.name}")

def exe_radmc_therm():
    mytools.exe('cd %s ; radmc3d mctherm setthreads 16 ' % dn_radmc )

def del_mctherm_files():
    remove_file(dn_radmc+"/radmc3d.out")

def set_radmc_parameters(dpath_radmc,
                 nphot=1000000,
                 r_in=1*cst.au, r_out=1000*cst.au, nr=128,
                 theta_in=0, theta_out=0.5*np.pi, ntheta=64,
                 phi_in=0, phi_out=0, nphi=1,
                 fdg=0.01, mol_abun=1e-7, mol_name='c18o',
                 m_mol=28*cst.amu, v_fwhm=0.5*cst.kms, Tenv=None,
                 opac="silicate",
                 temp_mode='mctherm', line=True, turb=True, lowreso=False, subgrid=True,
                 autoset=True, fn_model_pkl=None, fn_radmcset_pkl=None,
                 scattering_mode_max=0, temp_lambda=None,
                 Mstar=cst.Msun, Rstar=cst.Rsun, Lstar=cst.Lsun, **kwargs):
    cwd = os.getcwd()
    os.chdir(dpath_radmc)

    D = pd.read_pickle(dpath_radmc+"/"+fn_model_pkl)
    D = namedtuple("Model", D.keys())(*D.values())

    if subgrid:
        ri = np.logspace(np.log10(r_in), np.log10(r_out), nr+1)
        thetai = np.linspace(theta_in, theta_out, ntheta+1)
        phii = np.linspace(phi_in, phi_out, nphi+1)
        rc = 0.5 * (ri[0:nr] + ri[1:nr+1])
        thetac = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta+1])
        phic = 0.5 * (phii[0:nphi] + phii[1:nphi+1])
    else:
        rc = D.r_ax
        thetac = D.th_ax
        phic = D.ph_ax
        ri = rc

    nr = rc.size
    ntheta = thetac.size
    nphi = phic.size

    rr, tt, pp = np.meshgrid(rc, thetac, phic, indexing='ij')
    ntot = rr.size

    def interp_value(v):
        return _interpolator2d(v, D.r_ax, D.th_ax, rr, tt, logx=True, logy=False, logv=True)

    rhog = interp_value(D.rho_tot[:, :, 0])
    if np.max(rhog) == 0:
        print("!! WARNING !! : Zero density")
        #raise Exception("Zero density")
    rhod = rhog * fdg
    vr = interp_value(D.ur[:, :, 0])
    vth = interp_value(D.uth[:, :, 0])
    vph = interp_value(D.uph[:, :, 0])
    vturb = np.zeros_like(vr)

    n_mol = rhog / (2.34*cst.amu) * mol_abun
    n_mol = np.where(rr<1000*cst.au, n_mol, 0)
    if temp_mode == 'const':
        if Tenv is not None:
            T = Tenv
            v_fwhm = np.sqrt(T*(16*np.log(2)*cst.kB)/m_mol)
        else:
            v_fwhm = v_fwhm
            T = m_mol * v_fwhm**2 /(16 * np.log( 2) * cst.kB )
        logger.info(f"Constant temperature is  {T}")
        logger.info(f"v_FWHM_kms is  {v_fwhm/cst.kms}")
        tgas = T * np.ones_like(rhog)
        tdust = tags

    if temp_mode == 'lambda':
        #rr, tt = np.meshgrid(D.r_ax, D.th_ax)
        #print(rr.shape, rhog.shape)
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

    def join_list(l):
        return '\n'.join(map(lambda x: f'{x:13.6e}', l))

    # set grid
    save_file('amr_grid.inp',
              ["1", # iformat: AMR grid style  (0=regular grid, no AMR)
               "0",
               "100", # Coordinate system: spherical
               "0", # gridinfo
               "1 1 0", # Include r,theta coordinates
               f'{nr:d} {ntheta:d} {nphi:d}', # Size of grid
               join_list(ri),
               join_list(thetai),
               join_list(phii)])

    # set wavelength
    save_file('wavelength_micron.inp',
              [f'{nlam:d}',
               join_list(lam)])

    save_file('stars.inp',
              ['2', f"1 {nlam}",
               " ".join(map(lambda  x: f'{x:13.6e}', [Rstar, Mstar, 0, 0, 0])),
               join_list(lam),
               f"{-Tstar:13.6e}"])

    # set dust density
    save_file('dust_density.inp',
              ['1',
               f'{ntot:d}',
               '1',
               join_list(rhod.ravel(order='F')) ])


    if line:
        # set molcular number density
        save_file(f'numberdens_{mol_name}.inp',
                  ['1',
                   f'{ntot:d}',
                   join_list(n_mol.ravel(order='F')) ])

        # set_mol_lines
        save_file('lines.inp',
                  ['2',
                   '1', # number of molecules
                    f'{mol_name}  leiden 0 0 0']) # molname1 inpstyle1 iduma1 idumb1 ncol1

        # set_gas_velocity
        save_file('gas_velocity.inp',
                  ['1',  # Format number
                   f'{ntot:d}', # Nr of cells
                   *[f"{vr[ir, it, ip]:13.6e} {vr[ir, it, ip]:13.6e} {vr[ir, it, ip]:13.6e}\n"
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
                  "==========="]
    for op in opacs:
        text_lines += ['1', # Way in which this dust species is read
                       '0', # 0=Thermal grain
                       f'{op}',
                       "----------"] # Extension of name of dustkappa_***.inp file

    save_file('dustopac.inp', text_lines)

#    with open('dustopac.inp', 'w+') as f:
#        opacs = opac if isinstance(opac,list) else [opac]
#        f.write('2      Format number of this file\n')
#        f.write(f'{len(opacs)}     Nr of dust species\n')
#        f.write('========================================================\n')
#        for op in opacs:
#            f.write('1      Way in which this dust species is read\n')
#            f.write('0      0=Thermal grain\n')
#            f.write(f'{op}     Extension of name of dustkappa_***.inp file\n')
#            f.write('--------------------------------------------------------\n')
#        logger.info(f"Saved:  {f.name}")

    # set_input
    param_dict={"nphot": int(nphot),
                "scattering_mode_max": scattering_mode_max,
                "iranfreqmode": 1,
                "mc_scat_maxtauabs": 5.0,
                "tgas_eq_tdust": 1
                }

    with open('radmc3d.inp', 'w+') as f:
        f.write("\n".join([f'{k} = {v}'for k,v in param_dict.items()]))
        logger.info(f"Saved:  {f.name}")

    os.chdir(cwd)
#########################################################################################

class SetRadmc:
    def __init__(self, dpath_radmc,
                 nphot=1000000,
                 r_in=1*cst.au, r_out=1000*cst.au, nr=128,
                 theta_in=0, theta_out=0.5*np.pi, ntheta=64,
                 phi_in=0, phi_out=0, nphi=1,
                 fdg=0.01, mol_abun=1e-7, mol_name='c18o',
                 m_mol=28*cst.amu, v_fwhm=0.5*cst.kms, Tenv=None,
                 opac="silicate",
                 temp_mode='mctherm', line=True, turb=True, lowreso=False, subgrid=True,
                 autoset=True, fn_model_pkl=None, fn_radmcset_pkl=None,
                 scattering_mode_max=0, temp_lambda=None,
                 Mstar=cst.Msun, Rstar=cst.Rsun, Lstar=cst.Lsun, **kwargs):

        logger.info("radmc directry is %s"%dpath_radmc)
        mytools.set_arguments(self, locals(), printer=logger.info)

        self.nlam = None
        self.lam = None
        self.rhog = None
        self.rhod = None
        self.vr = None
        self.vth = None
        self.vph = None
        self.vturb = None
        self.Tstar = None

        if kwargs != {}:
            logger.info(f"There is unused args : {kwargs}")

        if autoset:
            self.set_all()

    def set_all(self):
        self.read_model_data()
        self.set_grid()
        self.set_physical_values()
        self.set_photon_wavelength()
        self.set_stars()
        if self.line:
            self.set_mol_lines()
            self.set_mol_ndens()
            self.set_gas_velocity()
        if (self.temp_mode == 'const') or (self.temp_mode == 'lambda'):
            self.set_gas_temperature()
            self.set_dust_temperature()
        elif self.temp_mode == 'mctherm':
            self.remove_gas_temperature()
            self.remove_dust_temperature()

        self.set_dust_density()
        self.set_dust_opacity()
        self.set_input()
        self.save_pickle()

    def read_model_data(self):
        D = pd.read_pickle(self.dpath_radmc+"/"+self.fn_model_pkl)
        self.D = namedtuple("Model", D.keys())(*D.values())

    def set_grid(self):
        if self.subgrid:
            #nr, ntheta, nphi = [128, 64, 1] if self.lowreso else [512, 256, 1]
            ri = np.logspace(np.log10(self.r_in), np.log10(self.r_out), self.nr+1)
            thetai = np.linspace(self.theta_in, self.theta_out, self.ntheta+1)
            phii = np.linspace(self.phi_in, self.phi_out, self.nphi+1)
            self.rc = 0.5 * (ri[0:self.nr] + ri[1:self.nr+1])
            self.thetac = 0.5 * (thetai[0:self.ntheta] + thetai[1:self.ntheta+1])
            self.phic = 0.5 * (phii[0:self.nphi] + phii[1:self.nphi+1])
        else:
            rc = self.D.r_ax
            thetac = self.D.th_ax
            phic = self.D.ph_ax
            ri = rc

        self.nr = self.rc.size
        self.ntheta = self.thetac.size
        self.nphi = self.phic.size

        self.rr, self.tt, self.pp = np.meshgrid(
            self.rc, self.thetac, self.phic, indexing='ij')
        self.ntot = self.rr.size
        with open(self.dpath_radmc+'/amr_grid.inp', 'w+') as f:
            f.write('1\n') # iformat: AMR grid style  (0=regular grid, no AMR)
            f.write('0\n')
            f.write('100\n') # Coordinate system: spherical
            f.write('0\n') # gridinfo
            f.write('1 1 0\n') # Include r,theta coordinates
            #f.write('1 1 1\n') # Include r,theta coordinates
            f.write('%d %d %d\n' % (self.nr, self.ntheta, self.nphi)) # Size of grid
            for value in ri:
                f.write('%13.6e\n' % (value)) # X coordinates (cell walls)
            for value in thetai:
                f.write('%13.6e\n' % (value)) # Y coordinates (cell walls)
            for value in phii:
                f.write('%13.6e\n' % (value)) # Z coordinates (cell walls)

        pstar = np.array([0., 0., 0.])
        with open(self.dpath_radmc+'/stars.inp', 'w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n' % (self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n' % \
                    (self.Rstar, self.Mstar, pstar[0], pstar[1], pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n' % (value))
            f.write('\n%13.6e\n' % (-self.Tstar))
            logger.info(f"Saved:  {f.name}")


    def set_physical_values(self):
        logger.info("Interpolating...")
        def interp_value(v):
            return _interpolator2d(v, self.D.r_ax, self.D.th_ax, self.rr, self.tt, logx=True, logy=False, logv=True)

        #test = _interpolator3d(self.D.ur, self.D.r_ax, self.D.th_ax, self.D.ph_ax, self.rr, self.tt, self.pp)
        #print(test)
        #exit()

        self.rhog = interp_value(self.D.rho_tot[:, :, 0])
        if np.max(self.rhog) == 0:
            print("!! WARNING !! : Zero density")
            #raise Exception("Zero density")
        self.rhod = self.rhog * self.fdg
        self.vr = interp_value(self.D.ur[:, :, 0])
        self.vth = interp_value(self.D.uth[:, :, 0])
        self.vph = interp_value(self.D.uph[:, :, 0])
        self.vturb = np.zeros_like(self.vr)

        self.n_mol = self.rhog / (2.34*cst.amu) * self.mol_abun
        self.n_mol = np.where(self.rr<1000*cst.au, self.n_mol, 0)
        if self.temp_mode == 'const':
            if self.Tenv is not None:
                T = self.Tenv
                v_fwhm = np.sqrt(T * (16 * np.log( 2) * cst.kB )/self.m_mol)
            else:
                v_fwhm = self.v_fwhm
                T = self.m_mol * v_fwhm**2 /(16 * np.log( 2) * cst.kB )
            logger.info(f"Constant temperature is  {T}")
            logger.info(f"v_FWHM_kms is  {v_fwhm/cst.kms}")
            self.tgas = T * np.ones_like(self.rhog)

        if self.temp_mode == 'lambda':
            #rr, tt = np.meshgrid(self.D.r_ax, self.D.th_ax)
            #print(self.rr.shape, self.rhog.shape)
            self.tgas = self.temp_lambda(self.rr/cst.au)
#10 * (self.rr/(1000*cst.au))**(-1.5)
            self.tgas = self.tgas.clip(min=0.1, max=10000)

        self.Tstar = (self.Lstar/cst.Lsun)**0.25 * cst.Tsun

    def set_photon_wavelength(self):
        lam1 = 0.1
        lam2 = 7
        lam3 = 25
        lam4 = 1e4
        n12 = 20
        n23 = 100
        n34 = 30
        lam12 = np.logspace(np.log10(lam1), np.log10(lam2),
                            n12, endpoint=False)
        lam23 = np.logspace(np.log10(lam2), np.log10(lam3),
                            n23, endpoint=False)
        lam34 = np.logspace(np.log10(lam3), np.log10(lam4), n34, endpoint=True)
        self.lam = np.concatenate([lam12, lam23, lam34])
        self.nlam = self.lam.size
        with open(self.dpath_radmc+'/wavelength_micron.inp', 'w+') as f:
            f.write('%d\n' % (self.nlam))
            for value in self.lam:
                f.write('%13.6e\n' % (value))
            logger.info(f"Saved:  {f.name}")

    def set_stars(self):
        pstar = np.array([0., 0., 0.])
        with open(self.dpath_radmc+'/stars.inp', 'w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n' % (self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n' % \
                    (self.Rstar, self.Mstar, pstar[0], pstar[1], pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n' % (value))
            f.write('\n%13.6e\n' % (-self.Tstar))
            logger.info(f"Saved:  {f.name}")

    def set_mol_ndens(self):
        with open(self.dpath_radmc+'/numberdens_%s.inp' % (self.mol_name), 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.n_mol.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')
            logger.info(f"Saved:  {f.name}")

    # Write the lines.inp control file
    def set_mol_lines(self):
        with open(self.dpath_radmc+'/lines.inp', 'w') as f:
            f.write('2\n')
            f.write('1\n')  # number of molecules
            # molname1 inpstyle1 iduma1 idumb1 ncol1
            f.write('%s  leiden 0 0 0\n' % (self.mol_name))
            logger.info(f"Saved:  {f.name}")

    # Write the gas velocity field
    def set_gas_velocity(self):
        with open(self.dpath_radmc+'/gas_velocity.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            for ip, it, ir in itertools.product(range(self.nphi), range(self.ntheta), range(self.nr)):
                f.write('%13.6e %13.6e %13.6e\n' % (self.vr[ir, it, ip],
                                                    self.vth[ir, it, ip],
                                                    self.vph[ir, it, ip]))
            logger.info(f"Saved:  {f.name}")

    def set_gas_turbulence(self):
        with open(self.dpath_radmc+'/microturbulence.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)    # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.vturb.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            #f.write('\n')
            logger.info(f"Saved:  {f.name}")


    def set_gas_temperature(self):
        with open(self.dpath_radmc+'/gas_temperature.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.tgas.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            #f.write('\n')
            logger.info(f"Saved:  {f.name}")

    def remove_gas_temperature(self):
        fpath = self.dpath_radmc+'/gas_temperature.inp'
        if os.path.exists(fpath):
            logger.info(f"remove gas_temperature.inp: {fpath}")
            os.remove(fpath)

    def set_dust_temperature(self):
        with open(self.dpath_radmc+'/dust_temperature.dat', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)
            f.write('1\n')                       # Format number
            # Create a 1-D view, fortran-style indexing
            data = self.tgas.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')
            logger.info(f"Saved:  {f.name}")

    def remove_dust_temperature(self):
        fpath = self.dpath_radmc+'/dust_temperature.dat'
        if os.path.exists(fpath):
            logger.info(f"remove dust_temperature.dat': {fpath}")
            os.remove(fpath)

    def set_dust_density(self):
        with open(self.dpath_radmc+'/dust_density.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)    # Nr of cells
            f.write('1\n')                       # Nr of dust species
            # Create a 1-D view, fortran-style indexing
            data = self.rhod.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')
            logger.info(f"Saved:  {f.name}")

    def set_dust_opacity(self):
        with open(self.dpath_radmc+'/dustopac.inp', 'w+') as f:
            opacs = self.opac if isinstance(self.opac,list) else [self.opac]
            f.write('2      Format number of this file\n')
            f.write('{}     Nr of dust species\n'.format(len(opacs)))
            f.write('========================================================\n')
            for op in opacs:
                f.write('1      Way in which this dust species is read\n')
                f.write('0      0=Thermal grain\n')
                f.write('{}     Extension of name of dustkappa_***.inp file\n'.format(op))
                f.write('--------------------------------------------------------\n')
            logger.info(f"Saved:  {f.name}")

    def set_input(self):
        params=[["nphot", int(self.nphot) ],
                ["scattering_mode_max", self.scattering_mode_max],
                ["iranfreqmode", 1],
                ["mc_scat_maxtauabs", 5.0],
                ["tgas_eq_tdust",1],
                #["nphot_scat", 1000000],
                #["camera_maxdphi", 0.0000001],
                #["camera_spher_cavity_relres", 0.005], # 0.05
                #["camera_min_dangle", 0.005], # 0.05
                #["camera_min_drr",0.0003], # 0.003
                #["camera_refine_criterion",1.0],
                #["nphot_spec", 100000],
                #"iseed", -5415],
                ]

        with open(self.dpath_radmc+'/radmc3d.inp', 'w+') as f:
            for k,v in params:
                 f.write(f'{k} = {v}\n')

            logger.info(f"Saved:  {f.name}")


    def save_pickle(self):
        del self.D, self.temp_lambda
#        import dill
#        dill.dump(self, open(self.dpath_radmc+'/'+self.fn_radmcset_pkl+".dill",'wb'))
#        return

        savefile = self.dpath_radmc+'/'+self.fn_radmcset_pkl
        pd.to_pickle(self, savefile, protocol=2)
        logger.info('Saved:  %s\n' % savefile)


def _interpolator2d(value, x_ori, y_ori, x_new, y_new, logx=False, logy=False, logv=False):
    xo = np.log10(x_ori) if logx else x_ori
    xn = np.log10(x_new) if logx else x_new
    yo = np.log10(y_ori) if logy else y_ori
    yn = np.log10(y_new) if logy else y_new
    vo = np.log10(np.abs(value)) if logv else value
    fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
    ret0 = fv(xn, yn)
    if logv:
        if (np.sign(value)!=1).any():
            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
            sgn = np.sign(fv_sgn(xn, yn))
            ret = np.where(sgn!=0, sgn*10**ret0, 0)
        else:
            ret = 10**ret0
    else:
        ret = ret0
    return np.nan_to_num(ret0)

def _interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new, logx=False, logy=False, logz=False, logv=False):
    if len(z_ori) == 1:
        value = _interpolator2d(value, x_ori, y_ori, xx_new, yy_new, logx=False, logy=False, logv=False)
        return value
#        return

    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
    def points(*xyz):
        return [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                       for r in self.xau ] for v in self.vkms]
        #return np.array(list(itertools.product(xyz[0], xyz[1], xyz[2])))
    xo = np.log10(x_ori) if logx else x_ori
    yo = np.log10(y_ori) if logy else y_ori
    zo = np.log10(z_ori) if logz else z_ori
    xn = np.log10(x_new) if logx else xx_new
    yn = np.log10(y_new) if logy else yy_new
    zn = np.log10(z_new) if logz else zz_new
    vo = np.log10(np.abs(value)) if logv else value
    print(np.stack([xn, yn, zn], axis=-1), xo, yo, zo )


    ret0 = RegularGridInterpolator((xo, yo, zo), vo, bounds_error=False, fill_value=-1 )( np.stack([xn, yn, zn], axis=-1))
    print(ret0, np.max(ret0) )
    exit()
    #fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
    ret0 = fv(xn, yn)
    if logv:
        if (np.sign(value)!=1).any():
            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
            sgn = np.sign(fv_sgn(xn, yn))
            ret = np.where(sgn!=0, sgn*10**ret0, 0)
        else:
            ret = 10**ret0
    else:
        ret = ret0
    return np.nan_to_num(ret0)


def _interpolator3d_bu(value, xyz_ori, xyz_new, logx=False, logy=False, logz=False, logv=False, pm=False):

    from scipy.interpolate import RegularGridInterpolator

    def points(xyz):
        return np.array(list(itertools.product(xyz[0], xyz[1], xyz[2])))

    if logx:
        xyz_ori[0] = np.log10(xyz_ori[0])
        xyz_new[0] = np.log10(xyz_new[0])

    if logy:
        xyz_ori[1] = np.log10(xyz_ori[1])
        xyz_new[1] = np.log10(xyz_new[1])

    if logz:
        xyz_ori[2] = np.log10(xyz_ori[2])
        xyz_new[2] = np.log10(xyz_new[2])

    if logv:
        sign = np.sign(value)
        value = np.log10(np.abs(value))

#    f = interp2d( x_ori , y_ori , value.T , fill_value = 0 )
#   print(  np.reshape(xyz_ori) )
    # , value.shape, np.stack(xyz_new, axis=1).shape )

#   print(xyz_ori)
#   print(   points(xyz_ori)    )
#   print(   points(xyz_new)   )
#   print(  value   )
#   f = griddata( points(xyz_ori)  , value.flatten() , points(xyz_new) )
    print(xyz_ori, xyz_new)
    f = RegularGridInterpolator(xyz_ori, value)

    print(f(points(xyz_new)))
#   fv = np.vectorize(f)
#   print(fv( [16],[0]))
#   print(   xyz_new[0], xyz_new[1] , xyz_new[2]   )
#   print( fv( xyz_new[0], xyz_new[1] , xyz_new[2]  ) )
    return f
 #   fv = np.vectorize(f)
#    ret =  fv( x_new  , y_new )
    if logz:
        if pm:
            f2 = interp2d(x_ori, y_ori, sign.T, fill_value=np.nan)
            fv2 = np.vectorize(f2)
            sgn = fv2(x_new, y_new)
            return np.where(sgn != 0, np.sign(sgn)*10**ret, 0)
        return 10**ret  # np.where( sgn!=0, np.sign(sgn)*10**ret, 0 )
    else:
        return ret



##################################################################################################################
#def readRadmcData(dn_radmc=None, use_ddens=True, use_dtemp=True,
#                use_gdens=True, use_gtemp=True, use_gvel=True, ispec=None, setting=None): # modified version of radmc3dPy.analyze.readData
#
#    res = rmca.radmc3dData()
#    res.grid = rmca.radmc3dGrid()
#    res.grid.readSpatialGrid(fname=dn_radmc+"/amr_grid.inp")
#
#    if use_ddens:
#        res.readDustDens(binary=False, fname=dn_radmc+"/dust_density.inp")
#
#    if use_dtemp:
#        res.readDustTemp(binary=False, fname=dn_radmc+"/dust_temperature.dat")
#
#    if use_gvel:
#        res.readGasVel(binary=False, fname=dn_radmc+"/gas_velocity.inp")
#
#    if use_gtemp:
#        res.readGasTemp(binary=False, fname=dn_radmc+"/gas_temperature.inp")
#
#    if use_gdens:
#        if not ispec:
#            print('ERROR\nNo gas species is specified!')
#            return 0
#        else:
#            res.ndens_mol = res._scalarfieldReader(fname=dn_radmc+"/numberdens_"+ispec+".inp", binary=False)
#    logger.info(f"RadmcData is created. Variables are : ",  {".join([ k for k, v in res.__dict__.items() if not isinstance(v, int)]) }")
#
#
#    return res

class RadmcData:
    def __init__(self, dn_radmc=None, dn_fig=None,
                 use_ddens=True, use_gdens=True, use_gtemp=True,
                 use_dtemp=True, use_gvel=True,
                 ispec=None, mol_abun=0, opac="", autoplot=True, plot_tau=False, fn_model=None):

        mytools.set_arguments(self, locals() )
        D = pd.read_pickle(dn_radmc+"/"+fn_model)
        self.D = namedtuple("Model", D.keys())(*D.values())

        ## getting data
        cwd = os.getcwd()
        os.chdir(dn_radmc)
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

       # if autoplot:
       #     self.plotall()

#    def readRadmcData(self): # modified version of radmc3dPy.analyze.readData
#        grid = rmc.radmc3dGrid.readSpatialGrid(fname=self.dn_radmc+"/amr_grid.inp", old=False)
#        res = rmca.radmc3dData()
#        #res.grid = rmca.radmc3dGrid(grid=)
#        if self.use_ddens:
#            res.readDustDens(binary=False, fname=self.dn_radmc+"/dust_density.inp")
#
#        if self.use_dtemp:
#            res.readDustTemp(binary=False, fname=self.dn_radmc+"/dust_temperature.dat")
#
#        if self.use_gvel:
#            res.readGasVel(binary=False, fname=self.dn_radmc+"/gas_velocity.inp")
#
#        if self.use_gtemp:
#            res.readGasTemp(binary=False, fname=self.dn_radmc+"/gas_temperature.inp")
#
#        if self.use_gdens:
#            if not self.ispec:
#                print('ERROR\nNo gas species is specified!')
#                return 0
#            else:
#                res.ndens_mol = res._scalarfieldReader(fname=self.dn_radmc+"/numberdens_"+self.ispec+".inp", binary=False)
#
#        logger.info("RadmcData is created. Variables are : ", ", ".join([ k for k, v in res.__dict__.items() if not isinstance(v, int)]) )
#
#        return res

    @staticmethod
    def cch_lifetime(nmol, nabun, T):
        k = 1.2e-11*np.exp(-998/T)
        return 1/( (nmol/nabun+1e-100)* k)

    def calc_tau_surface(self, tau=1, npix=100, sizeau=500, incl=90, phi=0, posang=85, n_thread=10, iline=None, lamb=None):
        common = f"incl {incl} phi {phi} setthreads {n_thread:d} "
        wl = f"iline {iline} " if iline is not None else f"lambda {lamb} "
        cmd = f"radmc3d tausurf {tau} npix {npix} sizeau {sizeau} fluxcons " + common + wl
        print(cmd)
        import os
        os.chdir(dn_radmc)
        subprocess.call(cmd, shell=True)
        data = rmci.readImage()
        self.imx = data.x
        self.imy = data.y
        print(vars(data))
        return data.image[:,:,0].T.clip(0)
#        fig = plt.figure()
#        c   = plb.contourf( data.x/cst.au , data.y/cst.au , data.image[:,:,0].T.clip(0)/cst.au, levels=np.linspace(0.0, 30, 20+1) )
#        cb = plb.colorbar(c)
#        plt.savefig(dn_fig+"tausurf.pdf")

#    def plotall(self):


    def calc_gas_trajectries(self):

        r_ew = self.D.cs * self.D.t
        print(f"r_ew is {r_ew/cst.au}")
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
            np.savetxt(f"{dn_fig}/stream_{i:d}.txt", stream_data, header="t[s] R[cm] z[cm] rho_gas[g cm^-3] T[K]")

#def read_radmc_data():


def plot_radmc_data(rmc_data):
   xlim = [rmc_data.xauc[0], rmc_data.xauc[-1]]

   pl1d = mp.Plotter(rmc_data.dn_fig, x=rmc_data.xauc,
                  logx=True, leg=False,
                  xl='Radius [au]', xlim=xlim,
                  fn_wrapper=lambda s:'rmc_%s_1d'%s)

   pl2d = mp.Plotter(rmc_data.dn_fig, x=rmc_data.RR, y=rmc_data.zz,
                  logx=False, logy=False, logcb=True, leg=False,
                  xl='Radius [au]', yl='Height [au]', xlim=[0, 500], ylim=[0, 500],
                  fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)

   def plot_plofs(d, fname, lim=None, log=True, lb=None):
       pl1d.plot(d[:,-1], fname, ylim=lim, logy=log, yl=lb)
       pl2d.map(d, fname, ctlim=lim, logcb=log, cbl=lb)

   if rmc_data.use_gdens:
       nmin = rmc_data.ndens_mol.min()
       nmax = rmc_data.ndens_mol.max()
       maxlim =  10**(0.5+round(np.log10(nmax)))
       plot_plofs(rmc_data.ndens_mol, "nden", lim=[maxlim*1e-3, maxlim], lb=r"Number density [cm$^{-3}$]")

   if rmc_data.use_gtemp:
       pl1d.plot(rmc_data.gtemp[:,-1], "temp", ylim=[1,1000], logy=True, yl='Temperature [K]')
       pl1d.plot(rmc_data.gtemp[:, 0], "temp_pol", ylim=[1,1000], logy=True, yl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_in", ctlim=[0,200], xlim=[0,100],ylim=[0,100], logcb=False, cbl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_out", ctlim=[0,100], logcb=False, cbl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_L", ctlim=[0,40], xlim=[0,7000],ylim=[0,7000], logcb=False, cbl='Temperature [K]')
       pl2d_log = mp.Plotter(rmc_data.dn_fig, x=rmc_data.RR, y=rmc_data.zz,
                  logx=True, logy=True, logcb=True, leg=False,
                  xl='log Radius [au]', yl='log Height [au]', xlim=[1, 1000], ylim=[1, 1000],
                  fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)

       pl2d_log.map(rmc_data.gtemp, "temp_log", ctlim=[10**0.5, 10**2.5], cbl='log Temperature [K]')

   if rmc_data.use_gvel:
       lb_v = r"Velocity [km s$^{-1}$]"
       plot_plofs(-rmc_data.vr/1e5, "gvelr", lim=[0,5], log=False, lb=lb_v)
       plot_plofs(rmc_data.vt/1e5, "gvelt", lim=[-5,5], log=False, lb=lb_v)
       plot_plofs(np.abs(rmc_data.vp)/1e5, "gvelp", lim=[0,5], log=False, lb=lb_v)

   if rmc_data.use_gdens and rmc_data.use_gtemp:
       plot_plofs(rmc_data.t_dest/rmc_data.t_dyn, "tche", lim=[1e-3,1e3], lb="CCH Lifetime/Dynamical Timescale")

   if rmc_data.opac!="":
       with open(f"{dn_radmc}/dustkappa_{rmc_data.opac}.inp", mode='r') as f:
           read_data = f.readlines()
           mode = int(read_data[0])
       if mode==2:
           lam, kappa_abs, kappa_sca = np.loadtxt(f"{dn_radmc}/dustkappa_{rmc_data.opac}.inp", skiprows=2).T
       elif mode==3:
           lam, kappa_abs, kappa_sca, _ = np.loadtxt(f"{dn_radmc}/dustkappa_{rmc_data.opac}.inp", skiprows=3).T

       mp.Plotter(rmc_data.dn_fig).plot([["ext",kappa_abs+kappa_sca],["abs", kappa_abs],["sca",kappa_sca]], "dustopac",
           x=lam, xlim=[0.03,3e4], ylim=[1e-4,1e6], logx=True, logy=True,
           xl=r'Wavelength [$\mu$m]', yl=r"Dust Extinction Opacity [cm$^2$ g$^{-1}$]",
           ls=["--"], c=["k"], lw=[3,2,2])

   exit()
   if 1:
       pl2d.map(D.rho, 'rho_L', ctlim=[1e-20, 1e-16], xlim=[0, 5000], ylim=[0, 5000], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec, save=False)
       pl2d.ax.plot(rmc_data.pos_list[0].R/cst.au, rmc_data.pos_list[0].z/cst.au, c="orangered", lw=1.5, marker="o")
       pl2d.save("rho_L_pt")

   if rmc_data.plot_tau:
       pl1d = mp.Plotter(rmc_data.dn_fig, x=rmc_data.imx/cst.au,
                      logx=True, leg=False,
                      xl='Radius [au]', xlim=xlim,
                      fn_wrapper=lambda s:'rmc_%s_1d'%s)

       pl2d = mp.Plotter(rmc_data.dn_fig, x=rmc_data.imx/cst.au, y=rmc_data.imy/cst.au,
                      logx=False, logy=False, logcb=True, leg=False,
                      xl='Radius [au]', yl='Height [au]', xlim=[-500/2, 500/2], ylim=[-500/2, 500/2],
                      fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)
       plot_plofs(rmc_data.tau/cst.au, "tau", lim=[1e-2, 1000], lb=r"tau")




if __name__ == '__main__':
    main()

