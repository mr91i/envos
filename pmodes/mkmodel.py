#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy import optimize
#
from header import inp, dn_home, dn_radmc, dn_fig
import myplot as mp
import cst
import mytools
import cubicsolver

msg = mytools.Message(__file__, debug=False)
#######################

def main():
    data = EnvelopeDiskModel(**vars(inp.model))

    if inp.model.plot:
        Plots(data, dn_fig=dn_fig)

class EnvelopeDiskModel:
    '''
    Calculator for infalling rotating envelope and Kepler rotating disk.
    Input parameter are envelope temperautre, centrifugal radius, and  Mass of star.

    '''
    def __init__(self,
        ire_model=1,
        r_in=1*cst.au, r_out=1000*cst.au, nr=601,
        theta_in=0, theta_out=0.5*np.pi, ntheta=91,
        phi_in=0, phi_out=0, nphi=1,
        Tenv_K=10, rCR_au=100, Mstar_Msun=1, t=None, Omega=None, j0=None, Mdot=None,
        cavity_angle=0, Tdisk=70, disk_star_fraction=0.01,
        rho0=None, rho_index=None,
        simple_density=False, disk=True, counterclockwise_rotation=False,
        fn_model_pkl=None, submodel=False,
        **args):

        mytools.set_arguments(self, locals(), printer=msg)

        # Non Variable Paramters Through Calculation
        self.model = self.read_inp_ire_model(ire_model)
        if 0:
            ri = np.logspace(np.log10(r_in), np.log10(r_out), nr+1)
            thetai = np.linspace(theta_in, theta_out, ntheta+1)
            phii = np.linspace(phi_in, phi_out, nphi+1)
            self.r_ax = 0.5 * (ri[0:nr] + ri[1:nr+1])
            self.th_ax = 0.5 * (thetai[0:ntheta] + thetai[1:ntheta+1])
            self.ph_ax = 0.5 * (phii[0:nphi] + phii[1:nphi+1])
        else:
            self.r_ax = np.logspace(np.log10(r_in), np.log10(r_out), nr)
            self.th_ax = np.linspace(theta_in, theta_out, ntheta)
            self.ph_ax = np.linspace(phi_in, phi_out, nphi) if nphi > 1 else [0]

        self.mu = np.round(np.cos(self.th_ax), 15)
        self.sin = np.where(self.mu == 1, 1e-100, np.sqrt(1 - self.mu**2))
        self.Mfin = 0.7 * cst.Msun

        self.Tenv, self.cs, self.Mdot, self.r_CR, self.Mstar, self.t, self.Omega, self.j0 = self.set_params(
            T=Tenv_K, CR=rCR_au*cst.au, M=Mstar_Msun*cst.Msun, t=t, Omega=Omega, j0=j0, Mdot=Mdot)
        self.GM = cst.G * self.Mstar
        self.r_CB = self.r_CR * 0.5  # 50 * cst.au
        self.r_in_lim = self.cs * self.Omega**2 * self.t**3
        self.Req = self.r_CB * 2 * np.sin(cavity_angle / 180 * np.pi)**2

        self.Mdisk = disk_star_fraction * self.Mstar
        self.cs_disk = np.sqrt(cst.kB * self.Tdisk / cst.mn)
        self.mu_cav = np.cos(self.cavity_angle/180*np.pi)

        if args != {}:
            msg("There is unused args :", args, debug=1)
#            raise Exception("There is unused args :", args)

        self.print_params()
        self.calc()


    @staticmethod
    def read_inp_ire_model(inp_ire_model):
        if isinstance(inp_ire_model, int):
            return {1: "CM", 2: "Simple"}[inp_ire_model]
        else:
            return inp_ire_model

    def print_params(self):
        def print_format(name, val, unit):
            msg(name.ljust(10)+'is {:10.2g} '.format(val)+unit.ljust(10))

        def print_str(name, s):
            msg(name.ljust(10)+'is '+s.rjust(10))

        print("")
        msg('Model Parameters:')
        print_str('Model', self.model)
        print_format('T', self.Tenv, 'K')
        print_format('cs', self.cs/cst.kms, 'km/s')
        print_format('t', self.t/cst.yr, 'yr')
        print_format('M', self.Mstar/cst.Msun, 'Msun')
        print_format('Omega', self.Omega, 's^-1')
        print_format('dM/dt', self.Mdot/(cst.Msun/cst.yr), 'M/yr')
        print_format('r_lim', self.r_in_lim/cst.au, 'au')
        print_format('j0', self.j0/(cst.kms*cst.au), 'au*km/s')
        print_format('j0', self.j0/(cst.kms*cst.pc), 'pc*km/s')
        print_format('r_CB', self.r_CB/cst.au, 'au')
        print_format('r_CR', self.r_CR/cst.au, 'au')
        print_format('Req', self.Req/cst.au, 'au')
        print_format('c.a.', self.cavity_angle, '')
        print('')

    def calc(self):
        vals_prt = {}

        for p in self.ph_ax:
            vals_rt = {}
            for r in self.r_ax:
                vals = self.calc_physical_structure_for_theta2(r,p)
                self._stack_dict(vals, vals_rt)
            self._stack_dict(vals_rt, vals_prt)

        for k, v_prt in vals_prt.items():
            setattr(self, k, np.array(v_prt).transpose(1,2,0))

        self.save_pickle()
        #self.save_hdf5()
        return self


    def calc_physical_structure_for_theta2(self, r, phi):
        RR, zz = r*self.sin, r*self.mu
        rho, ur, uth, uph, zeta, mu0, uR, uz = self.calc_Kinematics(
            r, model=self.model)
        rho_disk = self.put_Disk_sph(r, mode='CM_visc')
        b_disk = rho_disk > np.nan_to_num(rho)
        v_Kep = np.sqrt(self.GM / RR)
        rho_tot = rho + rho_disk
        vr_tot = np.where(b_disk, 0, ur)
        vphi_tot = np.where(b_disk, v_Kep, uph)
        vth_tot = np.where(b_disk, 0, uth)
        rr = np.full_like(self.th_ax, r)
        tt = self.th_ax
        zeta = np.full_like(RR,zeta)
        if self.submodel is not None:
            rho_sub, ur_sub, uth_sub, uph_sub, zeta_sub, mu0_sub, uR_sub, uz_sub = self.calc_Kinematics(
                r, model=self.submodel)
        ret = locals()
        del ret["self"]
        return ret

    @staticmethod
    def _stack_dict(dict_vals, dict_stacked):
        for k, v in dict_vals.items():
            if not k in dict_stacked:
                dict_stacked[k] = []
            if not isinstance(v, (list, np.ndarray)):
                v = [v]
            dict_stacked[k].append(v)
        dict_vals.clear()


    def calc_Kinematics(self, r, model='CM'):
        solver = {'CM': self.get_Kinematics_CM,
                  'Simple':self.get_Kinematics_SimpleBalistic}[model]
        rho, ur, uth, uph, zeta, mu0 = solver(r)
        if np.any(np.isnan([rho, ur, uth, uph, zeta, mu0])):
            raise Exception("Bad values.")

        if self.simple_density:
            rho = self.get_Kinematics_SimpleBalistic(r)[0]

        if self.counterclockwise_rotation:
            uph *= -1

        uR = ur * self.sin + uth * self.mu
        uz = ur * self.mu - uth * self.sin
        return rho, ur, uth, uph, zeta, mu0, uR, uz

    def get_Kinematics_CM(self, r):
        zeta = self.j0**2 / (self.GM * r)
        mu0 = self.get_mu0(zeta, method='cubic')
        sin0 = np.sqrt(1 - mu0**2)
        v0 = np.sqrt(self.GM / r)
        mu_over_mu0 = 1 - zeta*(1 - mu0**2)
        ur = - v0 * np.sqrt(1 + mu_over_mu0)
        uth = v0 * zeta*sin0**2*mu0/self.sin * np.sqrt(1 + mu_over_mu0)
        uph = v0 * sin0**2/self.sin * np.sqrt(zeta)
        rho = - self.Mdot / (4 * np.pi * r**2 * ur * (1 + zeta*(3*mu0**2-1) ) )
        mask = 1.0
        if self.cavity_angle is not None:
            mask = np.where(mu0 < self.mu_cav, 1, 0)
        return rho*mask, ur, uth, uph, zeta, mu0

    @staticmethod
    def sol_with_roots(m,zeta):
        sols = [ round(sol, 10) for sol in np.roots([zeta, 0, 1-zeta, -m]).real if 0 <= round(sol, 10) <= 1 ]
        return sols[0]

    @staticmethod
    def sol_with_cubic(m,zeta):
        sols = [ round(sol, 8) for sol in cubicsolver.solve(zeta, 0, 1-zeta, -m).real if 0 <= round(sol, 8) <= 1 ]
        try:
            return sols[0]
        except:
            sols_exc = cubicsolver.solve(zeta, 0, 1-zeta, -m).real
            print("val is {:.20f}".format(sols_exc[0]))
            raise Exception("No solution.", sols_exc, 0 <= round(sols_exc[0], 10) <= 1 )

    def get_mu0(self, zeta, method='roots'):
        solver = {"roots":self.sol_with_roots, "cubic":self.sol_with_cubic}[method]
        return np.array([solver(m, zeta) for m in self.mu])

    def get_Kinematics_SimpleBalistic(sellf, r, fillv=0):
        vff = np.sqrt(2 * self.GM / r)
        x = r/self.r_CB
        def value_env(val):
            b_env = np.logical_and(x >= 1, self.mu <= self.mu_cav)
            return np.where(b_env, val, fillv)
        rho_prof = self.Mdot/(4*np.pi*r**2 * vff))
            if (self.rho0 is not None) or (self.rho_index is not None) else self.rho0*x**self.rho_index
        rho = value_env(rho_prof)
        ur = - vff * np.sqrt(value_env(1-1/x))
        uth = value_env(0)
        uph = value_env(vff/np.sqrt(x))
        zeta = 0
        mu0 = self.mu
        return rho, ur, uth, uph, zeta, mu0

    def put_Disk_sph(self, r, CM=False, mode='exponential_cutoff', ind=-1):
        if not self.disk:
            return np.zeros_like(self.th_ax)
        R = r*self.sin
        z = r*self.mu
        if mode == 'CM_visc':
            # Viscous case
            u = R/self.r_CR
#           P = self.Mfin / self.Mdot / ( self.r_CR**2 / (3*0.01*self.csd**2) * np.sqrt(self.GM /self.r_CR_fin**3) )
            P = (3*0.01*self.cs_disk**2) / np.sqrt(self.GM / self.r_CR**3) * \
                self.Mstar**6/self.Mfin**5/self.Mdot/self.r_CR**2
            P_rd2 = 3 * 0.1 * self.Tdisk/self.Tenv * \
                np.sqrt(self.Mfin/self.Mstar) * \
                np.sqrt(cst.G * self.Mfin/self.r_CR)/self.cs
            a3 = 0.2757347731  # (2^10/3^11)^(0.25)
            ue = np.sqrt(3*P)*(1-56/51*a3/P**0.25)
            y = np.where(u < 1, 2*np.sqrt(1-u.clip(max=1)) + 4/3/np.sqrt(u) *
                         (1-(1+0.5*u)*np.sqrt(1-u.clip(max=1))), 4/3/np.sqrt(u))
            y = np.where(u <= ue, y - 4/3/np.sqrt( ue.clip(1e-30) ), 0)
            Sigma = 0.5/P * y * self.Mstar/(np.pi*self.r_CR**2)
#           print(P, P_rd2)
        elif mode == 'S+94':
            pass
#           A = 4*a/(m0 * Omega0 * cst.Rsun)
#           u = R/R_CR
#           V =
#           Sigma = (1-u)**0.5 /(2*A*u*t**2*V)

        elif mode == 'exponential_cutoff':
            Sigma_0 = self.Mdisk/(2*np.pi*self.r_CR**2)/(1-2/np.e)
            Sigma = Sigma_0 * (R/cst.au)**ind * np.exp(-(R/self.r_CR)**(2+ind))
        elif mode == 'tapered_cutoff':
            Sigma_0 = self.Mdisk/(2*np.pi*R_**2) * (ind+3)
            Sigma = self.Mdisk/(2*np.pi) * (ind+3) * \
                self.r_CR**(-ind-3) * (R/cst.au)**ind
        H = np.sqrt(cst.kB * self.Tdisk / cst.mn) / np.sqrt(cst.G*self.Mstar/R**3)
        rho = Sigma/np.sqrt(2*np.pi)/H * np.exp(- 0.5*z**2/H**2)
        return rho

    def save_pickle(self):
        savefile = dn_radmc + '/' + self.fn_model_pkl
        pd.to_pickle(self.__dict__, savefile, protocol=2)
        msg('Saved : %s\n' % savefile)

    def save_hdf5(self):
        #import h5py
        #import V1sHdf5
        #import evtk
        from evtk.hl import gridToVTK, pointsToVTK
        L = self.r_ax[-1]
        xi = np.linspace(-L/10,L/10,200)
        yi = np.linspace(-L/10,L/10,200)
        zi = np.linspace(0,L/10,100)
        xc = mytools.make_array_center(xi)
        yc = mytools.make_array_center(yi)
        zc = mytools.make_array_center(zi)
        xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
        xxc, yyc, zzc = np.meshgrid(xc, yc, zc, indexing='ij')
        rr_cert = np.sqrt(xxc**2 + yyc**2 + zzc**2)
        tt_cert = np.arccos(zzc/rr_cert) #np.arctan(np.sqrt(xxc**2 + yyc**2)/zzc)#(zzc/rr_cert)

        #pp_cert = np.arcsin(yyc/np.sqrt(xxc**2 + yyc**2)) + np.where( xxc >0 , 0,  ) #cos(xxc/np.sqrt(xxc**2 + yyc**2))
        pp_cert =  np.arctan2(yyc, xxc) #np.where(xxc>0, np.arcsin(yyc/np.sqrt(xxc**2 + yyc**2)) , -np.arcsin(yyc/np.sqrt(xxc**2 + yyc**2))) #cos(xxc/np.sqrt(xxc**2 + yyc**2))
        print( rr_cert, tt_cert, pp_cert)

        #print(self.rho.transpose(0,1,2).shape, self.rho.shape, self.r_ax.shape, self.th_ax.shape, self.ph_ax.shape,  rr_cert.shape, tt_cert.shape, pp_cert.shape)
        def interper(val):
            return interpolator3d(val, self.r_ax, self.th_ax, self.ph_ax, rr_cert, tt_cert, pp_cert, logx=False, logy=False, logz=False, logv=False)
        den_cert = interper(self.rho)
        #print(den_cert.shape)
        ur_cert = interper(self.ur)
        uth_cert = interper(self.uth)
        uph_cert = interper(self.uph)

        uux = ur_cert * np.sin(tt_cert) * np.cos(pp_cert) + uth_cert * np.cos(tt_cert) * np.cos(pp_cert) - uph_cert  * np.sin(pp_cert)
        uuy = ur_cert * np.sin(tt_cert) * np.sin(pp_cert) + uth_cert * np.cos(tt_cert) * np.sin(pp_cert) + uph_cert * np.cos(pp_cert)
        uuz = ur_cert * np.cos(tt_cert) - uth_cert * np.sin(tt_cert)


        ri = mytools.make_array_interface(self.r_ax)
        ti = mytools.make_array_interface(self.th_ax)
        pi = mytools.make_array_interface(self.ph_ax)


        import itertools
        #rtp = np.array(list(itertools.product(self.r_ax, self.th_ax, self.ph_ax)) ).T
        rr, tt, pp = np.meshgrid(self.r_ax,self.th_ax, self.ph_ax, indexing='ij')
        rri, tti, ppi = np.meshgrid(ri,ti,pi, indexing='ij')
        print(self.rho.shape, rri.shape)
        #pointsToVTK(dn_radmc + '/' +"model_sph.vtk", rr.ravel(), tt.ravel(), pp.ravel(), data = {"den" :self.rho.ravel() })
        #gridToVTK(dn_radmc + '/' +"model_sph.vtk", rri, tti, ppi, cellData = {"den" :self.rho}) #, "ur" :self.ur, "uth" :self.uth,"uph" :self.uph})
        #evtk.hl.imageToVTK(dn_radmc + '/' +"model.vtk", cellData = {"pressure" : pressure}, pointData = {"temp" : temp} )
        print(den_cert)
        gridToVTK(dn_radmc + '/' +"model.vtk", xi/cst.au, yi/cst.au, zi/cst.au, cellData = {"den" :den_cert, "ux" :uux, "uy" :uuy, "uz" :uuz})
        #gridToVTK(dn_radmc + '/' +"model.vtk", x_new, y_new, z_new, cellData = {"den" :den_cert})


        exit()


        with h5py.File( dn_radmc + '/' + 'model.h5', 'w') as f:
            f["x"] = xx
            f["y"] = yy
            f["z"] = zz
            f['x'].make_scale()
            f['y'].make_scale("yy")
            f['z'].make_scale("zz")
            #f["den"] = den_cert
            f.create_dataset("den", data=den_cert)
            f["den"].dims[0].attach_scale(f['x'])
            f["den"].dims[1].attach_scale(f['y'])
            f["den"].dims[2].attach_scale(f['z'])

#    def transform_to_cert():
#        x = np.linspace


        exit()

    def set_params(self, T=None, CR=None, M=None, t=None, Omega=None, j0=None, Mdot=None):  # choose 3
        #       eq1(cs,T),  eq2(Mdot,cs) ==> eq3(CR,M,j0), eq4(M,t), eq5(j0,Omega) : 5 vars 3 eqs
        if sum(a is None for a in locals().values()) != 4:
            raise Exception("Too many given parameters.")
        m0 = 0.975
#        cs = np.sqrt(cst.kB * T / cst.mn)
#        Mdot = cs**3 * m0 / cst.G
        try:
            if T:
                cs = np.sqrt(cst.kB * T / cst.mn)
                Mdot = cs**3 * m0 / cst.G
            elif Mdot:
                T = (Mdot * cst.G / m0)**(2/3) * cst.mn/cst.kB
                cs = np.sqrt(cst.kB * T / cst.mn)

            if M:
                t = M / Mdot
            elif t:
                M = Mdot * t

            if CR:
                j0 = np.sqrt(CR * cst.G * M)
            elif j0:
                CR = j0**2 / (cst.G * M)

            if j0:
                Omega = j0 / (0.5*cs*m0*t)**2
            elif Omega:
                j0 = (0.5*cs*m0*t)**2 * Omega
        except Exception as e:
            raise Exception(e, "Something wrong in parameter equations.")

        return T, cs, Mdot, CR, M, t, Omega, j0

    def stack(self, dict_vals, dict_stacked):
        for k, v in dict_vals.items():
            if not k in dict_stacked:
                dict_stacked[k] = []
            if not isinstance(v, (list, np.ndarray)):
                v = [v]
            dict_stacked[k].append(v)
        dict_vals.clear()

#
# Plotter
#


def Plots(D, r_lim=500, dn_fig=None):

    def slice_at_midplane(tt, *vals_rtp):
        iphi = 0
        if len(vals_rtp) >= 2:
            return np.array([val_rtp.take(iphi, 2)[tt.take(iphi, 2) == np.pi/2] for val_rtp in vals_rtp])
        else:
            return vals_rtp.take(iphi, 2)[tt == np.pi/2]

    ####
    stamp = inp.object_name
    ph_ax = D.ph_ax if len(D.ph_ax) != 1 else np.linspace(-np.pi, np.pi, 31)
    r_mg, th_mg, ph_mg = np.meshgrid(D.r_ax, D.th_ax, ph_ax, indexing='ij')
    R_mg, z_mg = r_mg * [np.sin(th_mg), np.cos(th_mg)]
    x_mg, y_mg = R_mg * [np.cos(ph_mg), np.sin(ph_mg)]

    plmap = mp.Plotter(dn_fig, x=R_mg.take(0, 2)/cst.au, y=z_mg.take(0, 2)/cst.au,
                       logx=False, logy=False, logcb=True, leg=False, square=True,
                       xl='Radius [au]', yl='Height [au]', xlim=[0, r_lim], ylim=[0, r_lim],
                       fn_wrapper=lambda s:'map_%s_%s'%(s, stamp),
                       decorator=lambda x: x.take(0,2))
    Vec = np.array([D.uR.take(0, 2), D.uz.take(0, 2)])
    # Density and velocity map
    plmap.map(D.rho, 'rho', cblim=[1e-20, 1e-16], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec)

    # Ratio between mu0 and mu : where these gas come from
    plmap.map(np.arccos(D.mu0)*180/np.pi, 'theta0', cblim=[0, 90], cbl=r'$\theta_0$ [degree]', div=10, Vector=Vec, n_sl=40, logcb=False)

    plmap.map(D.rho*r_mg**2, 'rhor2', cblim=[1e9, 1e14], cbl=r'log $\rho*r^2$ [g/cm]', div=10, n_sl=40)

    #from scipy.integrate import simps
    #integrated_value = np.where(r_mg < 300 * cst.au  , D.rho*r_mg**2*np.sin(th_mg), 0)
    #Mtot = 2*np.pi*simps( simps(integrated_value[:,:,0],D.th_ax, axis=1), D.r_ax)
    #print("Mtot is ", Mtot/cst.Msun)
    #plmap.map(D.zeta, 'zeta', cblim=[1e-2, 1e2], cbl=r'log $\zeta$ [g/cm$^{3}$]', div=6, n_sl=40)


    def zdeco_plane(z):
        if z.shape[2] == 1:
            return np.concatenate([z]*31, axis=2).take(-1,1)
        else:
            return z.take(-1,1)

    ux = D.ur * np.cos(ph_mg) - D.uph*np.sin(ph_mg)
    uy = D.ur * np.sin(ph_mg) + D.uph*np.cos(ph_mg)
    Vec = np.array([ux.take(-1, 1), uy.take(-1, 1)])

    plplane = mp.Plotter(dn_fig, x=x_mg.take(-1, 1)/cst.au, y=y_mg.take(-1, 1)/cst.au,
                       logx=False, logy=False, leg=False, square=True,
                       xl='x [au]', yl='y [au] (-:our direction)',
                       xlim=[-1000, 1000], ylim=[-1000, 1000],
                       fn_wrapper=lambda s:'plmap_%s_%s'%(s, stamp),
                       decorator=zdeco_plane)

    # Analyze radial profiles at the midplane
    # Slicing
    V_LS = x_mg/r_mg * D.uph - y_mg/r_mg*D.ur

    print("Vls is ", V_LS/1e5)
    plplane.map(V_LS/1e5, 'Vls', cblim=[-2.0, 2.0], cbl=r'$V_{\rm LS}$ [km s$^{-1}$]',
                   div=10, n_sl=20, logcb=False, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    plplane.map(D.rho, 'rho', cblim=[1e-18, 1e-16], cbl=r'log Density [g/cm$^{3}$]',
                   div=10, n_sl=20, logcb=True, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    rho0, uR0, uph0, rho_tot0 = slice_at_midplane(
        th_mg, D.rho, D.uR, D.uph, D.rho_tot)

    pl = mp.Plotter(dn_fig, x=D.r_ax/cst.au, leg=True, xlim=[0, 500], xl="Radius [au]")

    # Density as a function of distance from the center
    pl.plot([['nH2_env', rho0/cst.mn], ['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'rhos_%s' % stamp, ylim=[1e3, 1e9],  xlim=[10, 10000],
            lw=[3, 3, 6], c=[None, None, 'k'], ls=['--', ':', '-'],
            logxy=True, vl=[2*D.r_CB/cst.au])

    pl.plot(['nH2_env', rho0/cst.mn], #['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'nenv_%s' % stamp, ylim=[1e3, 1e9],  xlim=[10, 10000],
            lw=[3], logxy=True, vl=[2*D.r_CB/cst.au])
    pl.plot(['nH2_env', rho0/cst.mn], #['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'nenv_%s_lin' % stamp, ylim=[1e3, 1e9],  xlim=[0, 500],
            lw=[3], logy=True, vl=[2*D.r_CB/cst.au])

    # Make a 'balistic' orbit similar procedure to Oya+2014
    pl.plot([['-uR', -uR0/cst.kms], ['uph', uph0/cst.kms]],
            'v_%s' % stamp, ylim=[-1, 3], xlim=[0, 500], yl=r"Velocities [km s$^{-1}$]",
            lw=[2, 2, 4, 4], ls=['-', '-', '--', '--'])
    pl.plot([['-uR', -uR0/max(np.abs(uph0))], ['uph', uph0/max(np.abs(uph0))]],
            'vnorm_%s' % stamp, ylim=[0, 1.5], x=D.r_ax/D.r_CB, xlim=[0, 3],  yl=r"Velocities [$u_{\phi,\rm CR}$]",
            lw=[2, 2, 4, 4], ls=['-', '-', '--', '--'])

    if inp.model.submodel is not None:
        # see when and how much the results is different
        rho0_TSC, uR0_TSC, uph0_TSC = slice_at_midplane(
            re, th_mg, 'rho_sub', 'uR_sub', 'uph_sub')
        pl.plot({'log nH2 - 6': np.log10(rho0/cst.mn) - 6,
                 '-uR': -uR0/cst.kms,
                 'uph': uph0/cst.kms,
                 'log nH2_TSC - 6': np.log10(rho0_sub/cst.mn) - 6,
                 '-uR_TSC': -uR0_sub/cst.kms,
                 'uph_TSC': uph0_sub/cst.kms},
                'vrho_compare_%s' % stamp, x=D.r_ax/cst.au, xlim=[0, 500], ylim=[-2, 10],
                lw=[3, 3, 3, 6, 6, 6], c=['k', ], ls=['-', '-', '-', '--', '--', '--'],
                vl=[D.r_CB*2/cst.au, D.r_in_lim/cst.au])
    return

##########################################################################################################################

#def interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new, logx=False, logy=False, logz=False, logv=False):
def interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new, logx=False, logy=False, logz=False, logv=False):
#    if len(z_ori) == 1:
#        value = _interpolator2d(value, x_ori, y_ori, xx_new, yy_new, logx=False, logy=False, logv=False)
#        return value
#        return

    from scipy.interpolate import interpn, RectBivariateSpline, RegularGridInterpolator
#    def points(*xyz):
#        return [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
#                       for r in self.xau ] for v in self.vkms]
        #return np.array(list(itertools.product(xyz[0], xyz[1], xyz[2])))
#    xo = np.log10(x_ori) if logx else x_ori
#    yo = np.log10(y_ori) if logy else y_ori
#    zo = np.log10(z_ori) if logz else z_ori
#    xn = np.log10(x_new) if logx else xx_new
#    yn = np.log10(y_new) if logy else yy_new
#    zn = np.log10(z_new) if logz else zz_new
#    vo = np.log10(np.abs(value)) if logv else value
#    print(np.stack([xn, yn, zn], axis=-1), xo, yo, zo )
#    print(np.stack([xx_new, yy_new, zz_new], axis=-1).shape)
    print(value.shape)
    ret0 = interpn((x_ori, y_ori, z_ori), value, np.stack([xx_new, yy_new, zz_new], axis=-1), bounds_error=False, fill_value=np.nan )#( np.stack([xx_new, yy_new, zz_new], axis=-1))
    print(ret0.shape)
    return ret0
    #ret0 = RegularGridInterpolator((xo, yo, zo), vo, bounds_error=False, fill_value=-1 )( np.stack([xn, yn, zn], axis=-1))
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

###########
if __name__ == '__main__':
    main()
###########







