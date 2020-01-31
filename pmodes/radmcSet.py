#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,  absolute_import, division
import sys
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d, interp1d, griddata
import cst
from input_params import inp, dn_home, dn_radmc

def main():
    SetRadmc(dn_radmc, **vars(inp.radmc)).do_all()
    # or you can input parmeters directly.

class SetRadmc:
    def __init__(self, dpath_radmc,
                 nphot=1000000, 
                 r_in=1*cst.au, r_out=1000*cst.au, nr=128,
                 theta_in=0, theta_out=0.5*np.pi, ntheta=64,
                 phi_in=0, phi_out=0, nphi=1,
                 fdg=0.01, mol_abun=1e-7, mol_name='c18o',
                 tgas='tdust', line=True, turb=True, lowreso=False, subgrid=True, 
                 autoset=True, fn_model_pkl=None, fn_radmcset_pkl=None, 
                 Mstar=cst.Msun, Rstar=cst.Rsun, Lstar=cst.Lstar, **args):

        for k, v in locals().items():
            setattr(self, k, v)

        self.nlam = None
        self.lam = None
        self.rhog = None
        self.rhod = None
        self.tgas = None
        self.vr = None
        self.vth = None
        self.vph = None
        self.vturb = None
        self.Tstar = None
        if args != {}:
            raise Exception("There is unused args :", args)

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
            self.set_gas_velocity()
        if self.tgas == 'const':
            self.set_gas_temperature()
            self.set_dust_temperature()
        self.set_dust_density()
        self.set_dust_opacity()
        self.set_input()
        self.save_pickle()

    def read_model_data(self):
        D = pd.read_pickle(self.dpath_radmc+"/"+self.fn_model_pkl)
        self.D = namedtuple("Model", D.keys())(*D.values())

    def set_grid(self):
        if self.subgrid:
            nr, ntheta, nphi = [128, 64, 1] if self.lowreso else [512, 256, 1]
            ri = np.logspace(np.log10(self.r_in), np.log10(self.r_out), nr+1)
            thetai = np.linspace(self.theta_in, self.theta_out, ntheta+1)
            phii = np.linspace(self.phi_in, self.phi_out, nphi+1)
            rc = 0.5 * (ri[0:nr] + ri[1:nr+1])
            thetac = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta+1])
            phic = 0.5 * (phii[0:nphi] + phii[1:nphi+1])
        else:
            rc = self.D.r_ax
            thetac = self.D.th_ax
            phic = self.D.ph_ax
            ri = rc 

        self.nr = rc.size
        self.ntheta = thetac.size
        self.nphi = phic.size
        self.rr, self.tt, self.pp = np.meshgrid(
            rc, thetac, phic, indexing='ij')
        self.ntot = self.rr.size
        with open(self.dpath_radmc+'amr_grid.inp', 'w+') as f:
            f.write('1\n') # iformat: AMR grid style  (0=regular grid, no AMR)
            f.write('0\n')
            f.write('100\n') # Coordinate system: spherical
            f.write('0\n') # gridinfo
            f.write('1 1 0\n') # Include r,theta coordinates
            f.write('%d %d %d\n' % (self.nr, self.ntheta, self.nphi)) # Size of grid
            for value in ri:
                f.write('%13.6e\n' % (value)) # X coordinates (cell walls)
            for value in thetai:
                f.write('%13.6e\n' % (value)) # Y coordinates (cell walls)
            for value in phii:
                f.write('%13.6e\n' % (value)) # Z coordinates (cell walls)


    def set_physical_values(self):
        self.rhog = _interpolator2d(
            D.den_tot[:, :, 0], D.r_ax, D.th_ax, self.rr, self.tt, logx=True, logz=True)
        self.rhod = self.rhog * self.fdg

        self.vr = _interpolator2d(
            D.ur[:, :, 0], D.r_ax, D.th_ax, self.rr, self.tt, logx=True, logz=True)
        self.vth = _interpolator2d(
            D.uth[:, :, 0], D.r_ax, D.th_ax, self.rr, self.tt, logx=True, logz=True, pm=True)
        self.vph = _interpolator2d(
            D.uph[:, :, 0], D.r_ax, D.th_ax, self.rr, self.tt, logx=True, logz=True)
        self.vturb = np.zeros_like(self.vr)

        self.n_mol = self.rhog * self.mol_abun / (2.34*cst.amu)
        if self.tgas == 'const':
            tgas = 76*np.ones_like(self.rhog)

        self.Tstar = (self.Lstar/cst.Lsun)**0.25 * cst.Tsun

        #self.Mstar = self.Mstar  # cst.ms
        #self.Rstar = self.Rstar
        #self.Tstar = self.Tstar  # 1.2877 * cst.Tsun ## gives 2.75 Ls

#   @classmethod

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
        with open(self.dpath_radmc+'wavelength_micron.inp', 'w+') as f:
            f.write('%d\n' % (self.nlam))
            for value in self.lam:
                f.write('%13.6e\n' % (value))

    def set_stars(self):
        pstar = np.array([0., 0., 0.])
        with open(self.dpath_radmc+'stars.inp', 'w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n' % (self.nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n' % (self.Rstar, self.Mstar,
                                                                pstar[0], pstar[1], pstar[2]))
            for value in self.lam:
                f.write('%13.6e\n' % (value))
            f.write('\n%13.6e\n' % (-self.Tstar))

    def set_mol_ndens(self):
        with open(self.dpath_radmc+'numberdens_%s.inp' % (self.mol_name), 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.n_mol.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')

    # Write the lines.inp control file
    def set_mol_lines(self):
        with open(self.dpath_radmc+'lines.inp', 'w') as f:
            f.write('2\n')
            f.write('1\n')  # number of molecules
            # molname1 inpstyle1 iduma1 idumb1 ncol1
            f.write('%s  leiden 0 0 0\n' % (self.mol_name))

    # Write the gas velocity field
    def set_gas_velocity(self):
        with open(self.dpath_radmc+'gas_velocity.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            for ip, it, ir in itertools.product(range(self.nphi), range(self.ntheta), range(self.nr)):
                f.write('%13.6e %13.6e %13.6e\n' % (self.vr[ir, it, ip],
                                                    self.vth[ir, it, ip],
                                                    self.vph[ir, it, ip]))

    def set_gas_turbulence(self):
        with open(self.dpath_radmc+'microturbulence.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)    # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.vturb.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')

    def set_gas_temperature(self):
        with open(self.dpath_radmc+'gas_temperature.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.tgas.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')

    def set_dust_temperature(self):
        if self.tgas == 'const':
            with open(self.dpath_radmc+'dust_temperature.dat', 'w+') as f:
                f.write('1\n')                       # Format number
                f.write('%d\n' % self.ntot)
                f.write('1\n')                       # Format number
                # Create a 1-D view, fortran-style indexing
                data = self.tgas.ravel(order='F')
                data.tofile(f, sep='\n', format='%13.6e')
                f.write('\n')

    def set_dust_density(self):
        with open(self.dpath_radmc+'dust_density.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)    # Nr of cells
            f.write('1\n')                       # Nr of dust species
            # Create a 1-D view, fortran-style indexing
            data = self.rhod.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')

    def set_dust_opacity(self):
        with open(self.dpath_radmc+'dustopac.inp', 'w+') as f:
            opacs = ['silicate']
            f.write('2      Format number of this file\n')
            f.write('{}     Nr of dust species\n'.format(len(opacs)))
            f.write('========================================================\n')
            for op in opacs:
                f.write('1      Way in which this dust species is read\n')
                f.write('0      0=Thermal grain\n')
                f.write('{}     Extension of name of dustkappa_***.inp file\n'.format(op))
                f.write('--------------------------------------------------------\n')

    def set_input(self):
        with open(self.dpath_radmc+'radmc3d.inp', 'w+') as f:
            f.write('nphot = %d\n' % (self.nphot))
            f.write('scattering_mode_max = 0\n')  # 1: with scattering
            f.write('iranfreqmode = 1\n')
            f.write('mc_scat_maxtauabs = 5.d0\n')
            f.write('tgas_eq_tdust = %d' % (0 if self.tgas == 'tdust' else 1))

    def save_pickle(self):
        savefile = self.dpath_radmc+self.fn_radmcset_pkl
        pd.to_pickle(self, savefile, protocol=2)
        print('Saved : %s\n' % savefile)


def _interpolator2d(value, x_ori, y_ori, x_new, y_new, logx=False, logy=False, logz=False, pm=False):

    if logx:
        x_ori = np.log10(x_ori)
        x_new = np.log10(x_new)

    if logy:
        y_ori = np.log10(y_ori)
        y_new = np.log10(y_new)

    if logz:
        sign = np.sign(value)
        value = np.log10(np.abs(value))

    f = interp2d(x_ori, y_ori, value.T, fill_value=0)
    fv = np.vectorize(f)
    ret = fv(x_new, y_new)
    if logz:
        if pm:
            f2 = interp2d(x_ori, y_ori, sign.T, fill_value=np.nan)
            fv2 = np.vectorize(f2)
            sgn = fv2(x_new, y_new)
            return np.where(sgn != 0, np.sign(sgn)*10**ret, 0)
        return np.nan_to_num(10**ret)
    else:
        return np.nan_to_num(ret)


def _interpolator3d(value, xyz_ori, xyz_new, logx=False, logy=False, logz=False, logv=False, pm=False):

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


if __name__ == '__main__':
    main()
