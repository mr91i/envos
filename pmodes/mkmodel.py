#!/usr/bin/env python
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

msg = mytools.Message(__file__)
#######################


def main():
    data = EnvelopeDiskModel(**vars(inp.model))
    
    #Plots(data, dn_fig=dn_fig)

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
        Tenv=10, r_CR=100*cst.au, Mstar=cst.Msun, t=None, Omega=None, j0=None,
        cavity_angle=0, Tdisk=70, disk_star_fraction=0.01, 
        simple_density = False, disk=True, 
        fn_model_pkl=None, submodel=False, 
        **args):

        for k, v in locals().items():
            if k is not 'self':
                setattr(self, k, v)

        # Non Variable aramters Through Calculation
        self.model = self.read_inp_ire_model(ire_model)
        self.r_ax = np.logspace(np.log10(r_in), np.log10(r_out), nr)
        self.th_ax = np.linspace(theta_in, theta_out, ntheta)
        self.ph_ax = np.linspace(phi_in, phi_out, nphi)

        self.mu = np.round(np.cos(self.th_ax), 15)
        self.sin = np.where(self.mu == 1, 1e-100, np.sqrt(1 - self.mu**2))
        self.Mfin = 0.7 * cst.Msun

        self.Tenv, self.cs, self.dMdt, self.r_CR, self.Mstar, self.t, self.Omega, self.j0 = self.set_params(
            T=Tenv, CR=r_CR, M=Mstar, t=t, Omega=Omega, j0=j0)
        self.GM = cst.G * self.Mstar
        self.r_CB = self.r_CR * 0.5  # 50 * cst.au
        self.r_in_lim = self.cs * self.Omega**2 * self.t**3
        self.Req = self.r_CB * 2 * np.sin(cavity_angle / 180 * np.pi)**2

        self.Mdisk = disk_star_fraction * self.Mstar
        self.cs_disk = np.sqrt(cst.kB * self.Tdisk / cst.mn)
        self.mu_cav = np.cos(self.cavity_angle/180*np.pi)

        if args != {}:
            raise Exception("There is unused args :", args)

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
        print_format('dM/dt', self.dMdt/(cst.Msun/cst.yr), 'M/yr')
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
                self.stack(vals, vals_rt)
            self.stack(vals_rt, vals_prt)

        for k, v_prt in vals_prt.items():
            setattr(self, k, np.array(v_prt).transpose(1,2,0))

        self.Save_pickle()
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
        if self.submodel is not None:
            rho_sub, ur_sub, uth_sub, uph_sub, zeta_sub, mu0_sub, uR_sub, uz_sub = self.calc_Kinematics(
                r, model=self.submodel)
        ret = locals()
        del ret["self"]
        return ret

    def stack(self, dict_vals, dict_stacked):    
        for k, v in dict_vals.items():
            if not k in dict_stacked:
                dict_stacked[k] = []
            if not isinstance(v, (list, np.ndarray)):
                v = [v]
            dict_stacked[k].append(v)
        dict_vals.clear()


    def calc_Kinematics(self, r, model='CM'):
        if model == 'CM':
            solver = self.get_Kinematics_CM
        elif model == 'Simple':
            solver = self.get_Kinematics_SimpleBalistic
        else:
            raise Exception("Unknown model: ", model)
        rho, ur, uth, uph, zeta, mu0 = solver(r)
        if self.simple_density:
            rho = self.get_Kinematics_SimpleBalistic(r)[0]
        uR = ur * self.sin + uth * self.mu
        uz = ur * self.mu - uth * self.sin
        return rho, ur, uth, uph, zeta, mu0, uR, uz

    def get_Kinematics_CM(self, r):
        zeta = self.j0**2 / (self.GM * r)
        mu0 = self.get_mu0(zeta, method='cubic')
        sin0 = np.sqrt(1 - mu0**2)
        v0 = np.sqrt(self.GM / r)
        # np.where( np.logical_and(mu0==0,mu==0) , 1-zeta, mu/mu0 )
        mu_to_mu0 = 1 - zeta*(1 - mu0**2)
        ur = - v0 * np.sqrt(1 + mu_to_mu0)
        uth = v0 * zeta*sin0**2*mu0/self.sin * np.sqrt(1 + mu_to_mu0)
        uph = v0 * sin0**2/self.sin * np.sqrt(zeta)
        rho = - self.dMdt / (4 * np.pi * r**2 * ur) / (1 + zeta*(3*mu0**2-1))
        if self.cavity_angle is not None:
            mask = np.where(mu0 < self.mu_cav, 1, 0)
        return rho*mask, ur*mask, uth*mask, uph*mask, zeta, mu0

    @staticmethod
    def sol1(m,zeta):
        sols = [ round(sol, 10) for sol in np.roots([zeta, 0, 1-zeta, -m]).real if 0 <= round(sol, 10) <= 1 ]
        return sols[0]

    @staticmethod
    def sol2(m,zeta):
        sols = [ round(sol, 10) for sol in cubicsolver.solve(zeta, 0, 1-zeta, -m).real if 0 <= round(sol, 10) <= 1 ]
        return sols[0]

    def get_mu0(self, zeta, method='roots'):
        solver = {"roots":self.sol1, "cubic":self.sol2}[method]
        return np.array([solver(m, zeta) for m in self.mu])

    def get_Kinematics_SimpleBalistic(self, r, p=-1.5, r0=None, rho0=None, dMdt=None, h=0.1, fillv=0):
        vff = np.sqrt(2 * self.GM / r)
        x = r/self.r_CB
        b_env = np.logical_and(r*self.sin >= self.r_CB, self.mu <= self.mu_cav)
        rho = np.where(b_env, self.dMdt/(4*np.pi*r**2 * vff), fillv)
        ur = - vff * np.sqrt(np.where(b_env, 1-1/x, fillv))
        uth = np.where(b_env, 0, fillv)
        uph = np.where(b_env, vff/np.sqrt(x), fillv)
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
#           P = self.Mfin / self.dMdt / ( self.r_CR**2 / (3*0.01*self.csd**2) * np.sqrt(self.GM /self.r_CR_fin**3) )
            P = (3*0.01*self.cs_disk**2) / np.sqrt(self.GM / self.r_CR**3) * \
                self.Mstar**6/self.Mfin**5/self.dMdt/self.r_CR**2
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

    def Save_pickle(self):
        savefile = dn_radmc + '/' + self.fn_model_pkl
        pd.to_pickle(self.__dict__, savefile, protocol=2)
        msg('Saved : %s\n' % savefile)

    def set_params(self, T=None, CR=None, M=None, t=None, Omega=None, j0=None):  # choose 3
        #       eq1(cs,T),  eq2(dMdt,cs) ==> eq3(CR,M,j0), eq4(M,t), eq5(j0,Omega) : 5 vars 3 eqs
        if sum(a is None for a in locals().values()) != 3:
            raise Exception("Too many given parameters.")
        m0 = 0.975
        cs = np.sqrt(cst.kB * T / cst.mn)
        dMdt = cs**3 * m0 / cst.G
        try:
            if M:
                t = M / dMdt
            elif t:
                M = dMdt * t

            if CR:
                j0 = np.sqrt(CR * cst.G * M)
            elif j0:
                CR = j0**2 / (cst.G * M)

            if j0:
                Omega = j0 / (0.5*cs*m0*t)**2
            elif Omega:
                j0 = (0.5*cs*m0*t)**2 * Omega
        except:
            raise Exception("Something wrong in parameter equations.")

        return T, cs, dMdt, CR, M, t, Omega, j0

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
    R_mg, z_mg = r_mg * [np.sin(th_mg),  np.cos(th_mg)]
    x_mg, y_mg = R_mg * [np.cos(ph_mg), np.sin(ph_mg)]

    plmap = mp.Plotter(dn_fig, x=R_mg.take(0, 2)/cst.au, y=z_mg.take(0, 2)/cst.au, 
                       logx=False, logy=False, logcb=True, leg=False, 
                       xl='Radius [au]', yl='Height [au]', xlim=[0, r_lim], ylim=[0, r_lim], 
                       fn_wrapper=lambda s:'map_%s_%s'%(s, stamp),
                       decorator=lambda x: x.take(0,2))
    Vec = np.array([D.uR.take(0, 2), D.uz.take(0, 2)])
    # Density and velocity map
    plmap.map(D.rho, 'rho', cblim=[1e-21, 1e-16], cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec, n_sl=40)
    # Ratio between mu0 and mu : where these gas come from
    plmap.map(np.arccos(D.mu0)*180/np.pi, 'theta0', cblim=[0, 90], cbl=r'$\theta_0$ [degree]', div=10, Vector=Vec, n_sl=40, logcb=False)

    def zdeco_plane(z):
        if z.shape[2] == 1:
            return np.concatenate([z]*31, axis=2).take(-1,1)
        else:
            return z.take(-1,1)

    ux = D.ur * np.cos(ph_mg) - D.uph*np.sin(ph_mg)
    uy = D.ur * np.sin(ph_mg) + D.uph*np.cos(ph_mg)
    Vec = np.array([ux.take(-1, 1), uy.take(-1, 1)])

    plplane = mp.Plotter(dn_fig, x=x_mg.take(-1, 1)/cst.au, y=y_mg.take(-1, 1)/cst.au,
                       logx=False, logy=False, leg=False,
                       xl='x [au]', yl='y [au] (-:our direction)', xlim=[-1000, 1000], ylim=[-1000, 1000],
                       fn_wrapper=lambda s:'plmap_%s_%s'%(s, stamp),
                       decorator=zdeco_plane)

    # Analyze radial profiles at the midplane
    # Slicing
    V_LS = x_mg/r_mg * D.uph - y_mg/r_mg*D.ur

    plplane.map(V_LS/1e5, 'Vls', cblim=[-2.0, 2.0], cbl=r'$V_{\rm LS}$ [km s$^{-1}$]',
                   div=20, n_sl=40, logcb=False, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    plplane.map(D.rho, 'rho', cblim=[1e-18, 1e-16], cbl=r'log Density [g/cm$^{3}$]',
                   div=6, n_sl=40, logcb=True, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    rho0, uR0, uph0, rho_tot0 = slice_at_midplane(
        th_mg, D.rho, D.uR, D.uph, D.rho_tot)

    pl = mp.Plotter(dn_fig, x=D.r_ax/cst.au, leg=True, xlim=[0, 500], xl="Radius [au]")

    # Density as a function of distance from the center
    pl.plot([['nH2_env', rho0/cst.mn], ['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'rhos_%s' % stamp, ylim=[1e4, 1e15],  xlim=[1, 1000],
            lw=[3, 3, 6], c=[None, None, 'k'], ls=['--', ':', '-'],
            logxy=True, vl=[2*D.r_CB/cst.au])

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


###########
if __name__ == '__main__':
    main()
###########

