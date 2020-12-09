#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from scipy import optimize, interpolate, integrate
import itertools
#
from pmodes.header import inp, dpath_home, dpath_radmc, dpath_fig
import pmodes.myplot as mp
from pmodes import cst, tools, cubicsolver, tsc

#msg = tools.Message(__file__, debug=False)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#######################

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
        fn_model_pkl=None, submodel=False, calc_TSC=True,
        **args):

        tools.set_arguments(self, locals() )

        # Non Variable Paramters Through Calculation
        self.model = ire_model

        if 1:
            self.ri = np.logspace(np.log10(r_in), np.log10(r_out), nr+1)
            self.thetai = np.linspace(theta_in, theta_out, ntheta+1)
            self.phii = np.linspace(phi_in, phi_out, nphi+1)
            self.r_ax = 0.5 * (self.ri[0:nr] + self.ri[1:nr+1])
            self.th_ax = 0.5 * (self.thetai[0:ntheta] + self.thetai[1:ntheta+1])
            self.ph_ax = 0.5 * (self.phii[0:nphi] + self.phii[1:nphi+1])
        else:
            self.r_ax = np.logspace(np.log10(r_in), np.log10(r_out), nr)
            self.th_ax = np.linspace(theta_in, theta_out, ntheta)
            self.ph_ax = np.linspace(phi_in, phi_out, nphi) if nphi > 1 else np.array([0])

        self.rr, self.tt, self.pp = np.meshgrid(self.r_ax, self.th_ax, self.ph_ax, indexing='ij')
        self.RR = self.rr * np.sin(self.tt)
        self.zz = self.rr * np.cos(self.tt)

        self.mm = np.round(np.cos(self.tt), 15)
        self.ss = np.where(self.mm == 1, 1e-100, np.sqrt(1 - self.mm**2))

        self.Tenv, self.cs, self.Mdot, self.r_CR, self.Mstar, self.t, self.Omega, self.j0 = self.set_params(
            T=Tenv_K, CR=rCR_au*cst.au, M=Mstar_Msun*cst.Msun, t=t, Omega=Omega, j0=j0, Mdot=Mdot)

        self.r_CB = self.r_CR * 0.5  # 50 * cst.au
        self.r_in_lim = self.cs * self.Omega**2 * self.t**3
        self.Req = self.r_CB * 2 * np.sin(cavity_angle / 180 * np.pi)**2
        self.mu_cav = np.cos(self.cavity_angle/180*np.pi)

        if self.disk:
            self.Mfin = 0.7 * cst.Msun
            self.Mdisk = disk_star_fraction * self.Mstar
            self.cs_disk = np.sqrt(cst.kB * self.Tdisk / cst.mn)

        if args != {}:
            logger.debug("There is unused args : {args}")

        self.print_params()
        logger.info(f"tau is {self.Omega*self.t}")
        logger.info(f"xin_TSC is {self.r_in_lim/self.cs/self.t}")

        if self.calc_TSC and self.r_in_lim < self.r_out:
            self.tsc = tsc.solve_TSC(self.t, self.cs, self.Omega, run=True)
        self.rho, self.ur, self.uth, self.uph, self.zeta, self.mu0, self.uR, self.uz = self.calc_physical_structures()
        self.save_pickle()

    def print_params(self):
        logger.info('Model Parameters:')
        logger.info(f'Model {self.model}')
        value_list=[['T', self.Tenv, 'K'],
                    ['cs', self.cs/cst.kms, 'km/s'],
                    ['t', self.t/cst.yr, 'yr'],
                    ['M', self.Mstar/cst.Msun, 'Msun'],
                    ['Omega', self.Omega, 's^-1'],
                    ['dM/dt', self.Mdot/(cst.Msun/cst.yr), 'M/yr'],
                    ['r_lim', self.r_in_lim/cst.au, 'au'],
                    ['j0', self.j0/(cst.kms*cst.au), 'au*km/s'],
                    ['j0', self.j0/(cst.kms*cst.pc), 'pc*km/s'],
                    ['r_CB', self.r_CB/cst.au, 'au'],
                    ['r_CR', self.r_CR/cst.au, 'au'],
                    ['Req', self.Req/cst.au, 'au'],
                    ['c.a.', self.cavity_angle, 'degree']]
        for var, val, unit in value_list:
            logger.info(var.ljust(10) + f'is {val:10.2g} ' + unit.ljust(10))

    def calc_physical_structures(self):
        rho, ur, uth, uph, zeta, mu0, uR, uz = self.calc_Kinematics(model=self.model)
        RR, zz = self.rr * np.sin(self.t), self.rr * np.cos(self.tt)

        if self.calc_TSC:
            r_crit = self.r_in_lim*0.3
            b_tsc = self.rr > r_crit
            rho[b_tsc] = self.tsc.calc_rho(self.rr[b_tsc], self.tt[b_tsc])
            ur[b_tsc], uth[b_tsc], uph[b_tsc] = self.tsc.calc_velocity(self.rr[b_tsc], self.tt[b_tsc])

        if self.disk:
            rho_disk = self.put_Disk_sph(mode='CM_visc')
            b_disk = rho_disk > np.nan_to_num(rho)
            v_Kep = np.sqrt(cst.G * self.Mstar / self.RR)

            rho += rho_disk
            ur = np.where(b_disk, 0, ur)
            uth = np.where(b_disk, 0, uth)
            uphi = np.where(b_disk, v_Kep, uph)

        return rho, ur, uth, uph, zeta, mu0, uR, uz

    def calc_Kinematics(self, model='CM'):
        if model=='CM':
            rho, ur, uth, uph, zeta, mu0 = self.get_Kinematics_CM()
        elif model=='Simple':
            rho, ur, uth, uph, zeta, mu0 = self.get_Kinematics_SimpleBalistic()
        else:
            Exception

        if tools.isnan_values([rho, ur, uth, uph, zeta, mu0]):
            raise Exception("Bad values.", rho, ur, uth, uph, zeta, mu0)

        if self.simple_density:
            rho = self.get_Kinematics_SimpleBalistic()[0]

        if self.counterclockwise_rotation:
            uph *= -1

        uR = ur * self.ss + uth * self.mm
        uz = ur * self.mm - uth * self.ss
        return rho, ur, uth, uph, zeta, mu0, uR, uz

    def get_Kinematics_CM(self):
        zeta = self.j0**2 / (cst.G*self.Mstar * self.rr)
        mu0 = self.get_mu0(zeta, method='cubic')
        sin0 = np.sqrt(1 - mu0**2)
        v0 = np.sqrt(cst.G*self.Mstar / self.rr)
        mu_over_mu0 = 1 - zeta*(1 - mu0**2)
        ur = - v0 * np.sqrt(1 + mu_over_mu0)
        uth = v0 * zeta*sin0**2*mu0/self.ss * np.sqrt(1 + mu_over_mu0)
        uph = v0 * sin0**2/self.ss * np.sqrt(zeta)
        rho = - self.Mdot / (4 * np.pi * self.rr**2 * ur * (1 + zeta*(3*mu0**2-1) ) )
        mask = 1.0
        if self.cavity_angle is not None:
            mask = np.where(mu0 < self.mu_cav, 1, 0)
        return rho*mask, ur, uth, uph, zeta, mu0

    def get_mu0(self, zeta, method='roots'):
        def sol_with_roots(m,zeta):
            sols = [ round(sol, 10) for sol in np.roots([zeta, 0, 1-zeta, -m]).real if 0 <= round(sol, 10) <= 1 ]
            return sols[0]

        def sol_with_cubic(m,zeta):
            sols = [ round(sol, 8) for sol in cubicsolver.solve(zeta, 0, 1-zeta, -m).real if 0 <= round(sol, 8) <= 1 ]
            try:
                return sols[0]
            except:
                sols_exc = cubicsolver.solve(zeta, 0, 1-zeta, -m).real
                print("val is {:.20f}".format(sols_exc[0]))
                raise Exception("No solution.", sols_exc, 0 <= round(sols_exc[0], 10) <= 1 )

        csol = np.frompyfunc(sol_with_cubic, 2, 1)
        sol = csol(self.mm, zeta)
        return sol.astype(np.float64)


    def get_Kinematics_SimpleBalistic(self, fillv=0):
        vff = np.sqrt(2 * cst.G*self.Mstar / self.rr)
        xx = self.rr/self.r_CB
        b_env = np.logical_and(xx >= 1, self.mu <= self.mu_cav)
        def value_env(val):
            return np.where(b_env, val, fillv)
        rho_prof = self.Mdot/(4*np.pi*r**2 * vff) \
            if (self.rho0 is None) or (self.rho_index is None) else self.rho0*(self.rr/cst.au)**self.rho_index
        rho = np.where(b_env, rho_prof, rho_prof*0 )
        ur = - vff * np.sqrt(value_env(1-1/xx))
        uth = value_env(0)
        uph = value_env(vff/np.sqrt(xx))
        zeta = 0
        mu0 = self.mm
        return rho, ur, uth, uph, zeta, mu0

    def put_Disk_sph(self, mode='exponential_cutoff', ind=-1):
    ## Still constructing
        if not self.disk:
            return np.zeros_like(self.th_ax)
        #R = r*self.sin
        #z = r*self.mu
        if mode == 'CM_visc':
            # Viscous case
            u = self.RR/self.r_CR
#           P = self.Mfin / self.Mdot / ( self.r_CR**2 / (3*0.01*self.csd**2) * np.sqrt(cst.G*self.M /self.r_CR_fin**3) )
            P = (3*0.01*self.cs_disk**2) / np.sqrt(cst.G*self.Mstar / self.r_CR**3) * \
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
            Sigma = Sigma_0 * (self.RR/cst.au)**ind * np.exp(-(self.RR/self.r_CR)**(2+ind))
        elif mode == 'tapered_cutoff':
            Sigma_0 = self.Mdisk/(2*np.pi*self.r_CR**2) * (ind+3)
            Sigma = self.Mdisk/(2*np.pi) * (ind+3) * \
                self.r_CR**(-ind-3) * (self.RR/cst.au)**ind
        H = np.sqrt(cst.kB * self.Tdisk / cst.mn) / np.sqrt(cst.G*self.Mstar/self.RR**3)
        rho = Sigma/np.sqrt(2*np.pi)/H * np.exp(-0.5*self.zz**2/H**2)
        return rho

    def save_pickle(self):
        savefile = dpath_radmc + '/' + self.fn_model_pkl
        if self.calc_TSC:
            del self.tsc
        pd.to_pickle(self.__dict__, savefile, protocol=2)
        logger.info('Saved : %s\n' % savefile)

    def save_hdf5(self):
        from evtk.hl import gridToVTK, pointsToVTK
        L = self.r_ax[-1]
        xi = np.linspace(-L/10,L/10,200)
        yi = np.linspace(-L/10,L/10,200)
        zi = np.linspace(0,L/10,100)
        xc = tools.make_array_center(xi)
        yc = tools.make_array_center(yi)
        zc = tools.make_array_center(zi)
        xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
        xxc, yyc, zzc = np.meshgrid(xc, yc, zc, indexing='ij')
        rr_cert = np.sqrt(xxc**2 + yyc**2 + zzc**2)
        tt_cert = np.arccos(zzc/rr_cert) #np.arctan(np.sqrt(xxc**2 + yyc**2)/zzc)#(zzc/rr_cert)
        pp_cert =  np.arctan2(yyc, xxc) #np.where(xxc>0, np.arcsin(yyc/np.sqrt(xxc**2 + yyc**2)) , -np.arcsin(yyc/np.sqrt(xxc**2 + yyc**2))) #cos(xxc/np.sqrt(xxc**2 + yyc**2))
        #print(self.rho.transpose(0,1,2).shape, self.rho.shape, self.r_ax.shape, self.th_ax.shape, self.ph_ax.shape,  rr_cert.shape, tt_cert.shape, pp_cert.shape)
        def interper(val):
            return tools.interpolator3d(val, self.r_ax, self.th_ax, self.ph_ax, rr_cert, tt_cert, pp_cert, logx=False, logy=False, logz=False, logv=False)
        den_cert = interper(self.rho)
        ur_cert = interper(self.ur)
        uth_cert = interper(self.uth)
        uph_cert = interper(self.uph)
        uux = ur_cert * np.sin(tt_cert) * np.cos(pp_cert) + uth_cert * np.cos(tt_cert) * np.cos(pp_cert) - uph_cert  * np.sin(pp_cert)
        uuy = ur_cert * np.sin(tt_cert) * np.sin(pp_cert) + uth_cert * np.cos(tt_cert) * np.sin(pp_cert) + uph_cert * np.cos(pp_cert)
        uuz = ur_cert * np.cos(tt_cert) - uth_cert * np.sin(tt_cert)
        ri = tools.make_array_interface(self.r_ax)
        ti = tools.make_array_interface(self.th_ax)
        pi = tools.make_array_interface(self.ph_ax)
        rr, tt, pp = np.meshgrid(self.r_ax,self.th_ax, self.ph_ax, indexing='ij')
        rri, tti, ppi = np.meshgrid(ri,ti,pi, indexing='ij')
        gridToVTK(dpath_radmc + '/' +"model.vtk", xi/cst.au, yi/cst.au, zi/cst.au, cellData = {"den" :den_cert, "ux" :uux, "uy" :uuy, "uz" :uuz})


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

###########################################
###########################################

###########
if __name__ == '__main__':
    data = EnvelopeDiskModel(**vars(inp.model))
    #plot_physical_model(data, dpath_fig=dpath_fig)
###########
