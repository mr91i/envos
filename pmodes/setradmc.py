#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function,  absolute_import, division
import os
import itertools
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d, interp1d, griddata
import cst
from header import inp, dn_home, dn_radmc, dn_fig
import mytools
import radmc3dPy.analyze as rmca
import myplot as mp


msg = mytools.Message(__file__)
#######################

def main():
    SetRadmc(dn_radmc, **vars(inp.radmc))
    # or you can input parmeters directly.
    if inp.radmc.temp_mode=="mctherm":
        exe_radmc_therm()
    else:
        del_mctherm_files()

    if inp.radmc.plot:
        rdata = RadmcData(dn_radmc=dn_radmc, dn_fig=dn_fig, ispec=inp.radmc.mol_name, mol_abun=inp.radmc.mol_abun)

def exe_radmc_therm():
    mytools.exe('cd %s ; radmc3d mctherm setthreads 16' % dn_radmc )

def del_mctherm_files():
    if os.path.exists(dn_radmc+"/radmc3d.out"):
        mytools.exe('rm {}/radmc3d.out'.format(dn_radmc))

class SetRadmc:
    def __init__(self, dpath_radmc,
                 nphot=1000000,
                 r_in=1*cst.au, r_out=1000*cst.au, nr=128,
                 theta_in=0, theta_out=0.5*np.pi, ntheta=64,
                 phi_in=0, phi_out=0, nphi=1,
                 fdg=0.01, mol_abun=1e-7, mol_name='c18o',
                 m_mol=28*cst.amu, v_fwhm=0.5*cst.kms, Tenv=None,
                 temp_mode='mctherm', line=True, turb=True, lowreso=False, subgrid=True,
                 autoset=True, fn_model_pkl=None, fn_radmcset_pkl=None,
                 Mstar=cst.Msun, Rstar=cst.Rsun, Lstar=cst.Lsun, **kwargs):

        msg("radmc directry is %s"%dpath_radmc)
        msg("Set radmc parameters")
        for k, v in locals().items():
            if (k != 'self') and (k!="kwargs"):
                print(k,v)
                setattr(self, k, v)
                msg(k.ljust(20)+"is {:20}".format(v if v is not None else "None"))

        self.nlam = None
        self.lam = None

        self.rhog = None
        self.rhod = None
        self.vr = None
        self.vth = None
        self.vph = None
        self.vturb = None
        self.Tstar = None
        #if args != {}:
        #    raise Exception("There is unused args :", args)
        if kwargs != {}:
            msg("There is unused args :", kwargs)

        #exit()

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
        if self.temp_mode == 'const':
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


    def set_physical_values(self):
        msg("Interpolating...")
        def interp_value(v):
            return _interpolator2d(v, self.D.r_ax, self.D.th_ax, self.rr, self.tt, logx=True, logy=False, logv=True)

        #test = _interpolator3d(self.D.ur, self.D.r_ax, self.D.th_ax, self.D.ph_ax, self.rr, self.tt, self.pp)
        #print(test)
        #exit()

        self.rhog = interp_value(self.D.rho_tot[:, :, 0])
        if np.max(self.rhog) == 0:
            raise Exception("Zero density")
        self.rhod = self.rhog * self.fdg
        self.vr = interp_value(self.D.ur[:, :, 0])
        self.vth = interp_value(self.D.uth[:, :, 0])
        self.vph = interp_value(self.D.uph[:, :, 0])
        self.vturb = np.zeros_like(self.vr)

        self.n_mol = self.rhog / (2.34*cst.amu) * self.mol_abun
        if self.temp_mode == 'const':
            if self.Tenv is not None:
                T = self.Tenv
                v_fwhm = np.sqrt(T * (16 * np.log( 2) * cst.kB )/self.m_mol)
            else:
                v_fwhm = self.v_fwhm
                T = self.m_mol * v_fwhm**2 /(16 * np.log( 2) * cst.kB )
            msg("Constant temperature is ", T)
            msg("v_FWHM_kms is ", v_fwhm/cst.kms)
            self.tgas = T * np.ones_like(self.rhog)

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
            msg("Saved: ",f.name)

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
            msg("Saved: ",f.name)

    def set_mol_ndens(self):
        with open(self.dpath_radmc+'/numberdens_%s.inp' % (self.mol_name), 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.n_mol.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            f.write('\n')
            msg("Saved: ",f.name)

    # Write the lines.inp control file
    def set_mol_lines(self):
        with open(self.dpath_radmc+'/lines.inp', 'w') as f:
            f.write('2\n')
            f.write('1\n')  # number of molecules
            # molname1 inpstyle1 iduma1 idumb1 ncol1
            f.write('%s  leiden 0 0 0\n' % (self.mol_name))
            msg("Saved: ",f.name)

    # Write the gas velocity field
    def set_gas_velocity(self):
        with open(self.dpath_radmc+'/gas_velocity.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            for ip, it, ir in itertools.product(range(self.nphi), range(self.ntheta), range(self.nr)):
                f.write('%13.6e %13.6e %13.6e\n' % (self.vr[ir, it, ip],
                                                    self.vth[ir, it, ip],
                                                    self.vph[ir, it, ip]))
            msg("Saved: ",f.name)

    def set_gas_turbulence(self):
        with open(self.dpath_radmc+'/microturbulence.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)    # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.vturb.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            #f.write('\n')
            msg("Saved: ",f.name)


    def set_gas_temperature(self):
        with open(self.dpath_radmc+'/gas_temperature.inp', 'w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n' % self.ntot)            # Nr of cells
            # Create a 1-D view, fortran-style indexing
            data = self.tgas.ravel(order='F')
            data.tofile(f, sep='\n', format='%13.6e')
            #f.write('\n')
            msg("Saved: ",f.name)

    def remove_gas_temperature(self):
        fpath = self.dpath_radmc+'/gas_temperature.inp'
        if os.path.exists(fpath):
            msg("remove gas_temperature.inp:", fpath)
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
            msg("Saved: ",f.name)

    def remove_dust_temperature(self):
        fpath = self.dpath_radmc+'/dust_temperature.dat'
        if os.path.exists(fpath):
            msg("remove dust_temperature.dat':", fpath)
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
            msg("Saved: ",f.name)

    def set_dust_opacity(self):
        with open(self.dpath_radmc+'/dustopac.inp', 'w+') as f:
            opacs = ['silicate']
            f.write('2      Format number of this file\n')
            f.write('{}     Nr of dust species\n'.format(len(opacs)))
            f.write('========================================================\n')
            for op in opacs:
                f.write('1      Way in which this dust species is read\n')
                f.write('0      0=Thermal grain\n')
                f.write('{}     Extension of name of dustkappa_***.inp file\n'.format(op))
                f.write('--------------------------------------------------------\n')
            msg("Saved: ",f.name)

    def set_input(self):
        params=[["nphot", 1000000 ], #self.nphot],
                ["scattering_mode_max", 0],
                ["iranfreqmode", 1],
                ["mc_scat_maxtauabs", 5.0],
                ["tgas_eq_tdust",1],
                #["camera_maxdphi", 0.0000001],
                #["camera_spher_cavity_relres", 0.005], # 0.05
                #["camera_min_dangle", 0.005], # 0.05
                #["camera_min_drr",0.0003], # 0.003
                #["camera_refine_criterion",1.0],
                #["nphot_spec", 100000],
                #["iseed", -5415],
                ]

        with open(self.dpath_radmc+'/radmc3d.inp', 'w+') as f:
            for k,v in params:
                 f.write('{} = {}\n'.format(k, v))

#            f.write('nphot = %d\n' % self.nphot)
#            f.write('scattering_mode_max = 0\n')  # 1: with scattering
#            f.write('iranfreqmode = 1\n')
#            f.write('mc_scat_maxtauabs = 5.d0\n')
#            f.write('tgas_eq_tdust = 1\n') # %d')# % int(self.temp_mode == '') )
#            f.write('camera_maxdphi = 0.05\n')
#            f.write('iseed = -5415\n')


            #f.write('nphot_spec = 200000\n')

            msg("Saved: ",f.name)


    def save_pickle(self):
        del self.D
        savefile = self.dpath_radmc+'/'+self.fn_radmcset_pkl
        pd.to_pickle(self, savefile, protocol=2)
        msg('Saved:  %s\n' % savefile)



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
def readRadmcData(dn_radmc=None, use_ddens=True, use_dtemp=True,
                use_gdens=True, use_gtemp=True, use_gvel=True, ispec=None): # modified version of radmc3dPy.analyze.readData

    res = rmca.radmc3dData()
    res.grid = rmca.radmc3dGrid()
    res.grid.readSpatialGrid(fname=dn_radmc+"/amr_grid.inp")

    if use_ddens:
        res.readDustDens(binary=False, fname=dn_radmc+"/dust_density.inp")

    if use_dtemp:
        res.readDustTemp(binary=False, fname=dn_radmc+"/dust_temperature.dat")

    if use_gvel:
        res.readGasVel(binary=False, fname=dn_radmc+"/gas_velocity.inp")

    if use_gtemp:
        res.readGasTemp(binary=False, fname=dn_radmc+"/gas_temperature.inp")

    if use_gdens:
        if not ispec:
            print('ERROR\nNo gas species is specified!')
            return 0
        else:
            res.ndens_mol = res._scalarfieldReader(fname=dn_radmc+"/numberdens_"+ispec+".inp", binary=False)
    msg("RadmcData is created. Variables are : ", ", ".join([ k for k, v in res.__dict__.items() if not isinstance(v, int)]) )

    return res

class RadmcData:
    def __init__(self, dn_radmc=None, dn_fig=None, use_ddens=True, use_gdens=True, use_gtemp=True,
                 use_dtemp=True, use_gvel=True, ispec=None, mol_abun=0, autoplot=True):
        for k, v in locals().items():
            if k is not 'self':
                setattr(self, k, v)

        data = self.readRadmcData()

        self.xauc = data.grid.x/cst.au
        self.rrc, self.ttc = np.meshgrid(self.xauc, data.grid.y, indexing='xy')
        self.RR = self.rrc * np.sin(self.ttc)
        self.zz = self.rrc * np.cos(self.ttc)

        if use_gtemp:
            try:
                self.gtemp = data.gastemp[:,:,0,0].T
            except:
                self.gtemp = data.dusttemp[:,:,0,0].T

        if use_gdens and ispec:
            self.ndens_mol = data.ndens_mol[:,:,0,0].T

        if use_gvel:
            self.vr = data.gasvel[:,:,0,0].T
            self.vt = data.gasvel[:,:,0,1].T
            self.vp = data.gasvel[:,:,0,2].T

        if use_gdens and use_gtemp:
            # Chemical lifetime
            self.t_dest = self.cch_lifetime( self.ndens_mol, self.mol_abun, self.gtemp)
            self.t_dyn = 5.023e6 * np.sqrt( self.rrc**3 /0.18 ) ## sqrt(au^3/GMsun) = 5.023e6

        if autoplot:
            self.plotall()

    def readRadmcData(self): # modified version of radmc3dPy.analyze.readData

        res = rmca.radmc3dData()
        res.grid = rmca.radmc3dGrid()
        res.grid.readSpatialGrid(fname=self.dn_radmc+"/amr_grid.inp")

        if self.use_ddens:
            res.readDustDens(binary=False, fname=self.dn_radmc+"/dust_density.inp")

        if self.use_dtemp:
            res.readDustTemp(binary=False, fname=self.dn_radmc+"/dust_temperature.dat")

        if self.use_gvel:
            res.readGasVel(binary=False, fname=self.dn_radmc+"/gas_velocity.inp")

        if self.use_gtemp:
            res.readGasTemp(binary=False, fname=self.dn_radmc+"/gas_temperature.inp")

        if self.use_gdens:
            if not self.ispec:
                print('ERROR\nNo gas species is specified!')
                return 0
            else:
                res.ndens_mol = res._scalarfieldReader(fname=self.dn_radmc+"/numberdens_"+self.ispec+".inp", binary=False)

        msg("RadmcData is created. Variables are : ", ", ".join([ k for k, v in res.__dict__.items() if not isinstance(v, int)]) )

        return res

    @staticmethod
    def cch_lifetime(nmol, nabun, T):
        k = 1.2e-11*np.exp(-998/T)
        return 1/( (nmol/nabun+1e-100)* k)

    def plotall(self):

        pl1d = mp.Plotter(self.dn_fig, x=self.xauc,
                       logx=True, leg=False,
                       xl='Radius [au]', xlim=[10, 10000],
                       fn_wrapper=lambda s:'rmc_%s_1d'%s)

        pl2d = mp.Plotter(self.dn_fig, x=self.RR, y=self.zz,
                       logx=False, logy=False, logcb=True, leg=False,
                       xl='Radius [au]', yl='Height [au]', xlim=[0, 500], ylim=[0, 500],
                       fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)

        def plot_plofs(d, fname, lim=None, log=True, lb=None):
            pl1d.plot(d[-1,:], fname, ylim=lim, logy=log, yl=lb)
            pl2d.map(d, fname, cblim=lim, logcb=log, cbl=lb)

        if self.use_gdens:
            nmin = self.ndens_mol.min()
            nmax = self.ndens_mol.max()
            maxlim =  10**(0.5+round(np.log10(nmax)))
            plot_plofs(self.ndens_mol, "nden", lim=[maxlim*1e-3, maxlim], lb=r"Number density [cm$^{-3}$]")

        if self.use_gtemp:
            plot_plofs(self.gtemp, "temp", lim=[1,1000], lb='Temperature [K]')

        if self.use_gtemp:
            plot_plofs(self.gtemp, "temp", lim=[1,1000], lb='Temperature [K]')

        if self.use_gvel:
            lb_v = r"Velocity [km s$^{-1}$]"
            plot_plofs(-self.vr/1e5, "gvelr", lim=[0,5], log=False, lb=lb_v)
            plot_plofs(self.vt/1e5, "gvelt", lim=[-5,5], log=False, lb=lb_v)
            plot_plofs(np.abs(self.vp)/1e5, "gvelp", lim=[0,5], log=False, lb=lb_v)

        if self.use_gdens and self.use_gtemp:
            plot_plofs( self.t_dest/self.t_dyn, "tche", lim=[1e-3,1e3], lb="CCH Lifetime/Dynamical Timescale")


if __name__ == '__main__':
    main()

