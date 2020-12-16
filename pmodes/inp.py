#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Here we can controll the full paramters used in pmodes.
#

import sys
import numpy as np
from pmodes import cst
import os

def gen_input():
    object_name   = "L1527"
    r_lim_au      = [1, 10000]
    theta_lim_rad = [0, np.pi/2]
    dr_au_100     = 5
    CR_au         = 130
    M_Msun        = 0.2
    Mdot_Msyr     = 4.5e-6
    cavangle_deg  = 85
    Lstar_Lsun    = 2.75
    f_dg          = 0.01
    opac_filename = "MRN20"
    mol_abun      = 1e-17
    mol_name      = "c18o"
    mol_iline     = 3
    dpc           = 140
    incl_deg      = 85     # angle between direction to us , 0: face-on
    sizex_au      = 100
    sizey_au      = 100
    pixsize_au    = 10    # def: 0.08pc = 11au
    vwidth_kms    = 6     # def : 3
    dv_kms        = 0.04  # def : 0.08 km/s
    beam_maj_asec = 0.7    # FWHM of beam, beam size
    beam_min_asec = 0.7
    beam_pa_deg   = 0.0    # -41
    v_reso_kms    = 0.1

    inp = InputSet(object_name, debug=True, omp=True, n_thread=12) # oreyou
    inp.set_log_square_grid_2d(r_lim_au[0], r_lim_au[1], theta_lim_rad[0], theta_lim_rad[1], dr_au_100=dr_au_100)
    inp.set_model_params(cavangle_deg=cavangle_deg, CR_au=CR_au, M_Msun=M_Msun, Mdot_Msyr=Mdot_Msyr,
                         ccw_rotation=True, add_outenv=False, add_disk=False)
    inp.set_radmc(Lstar_Lsun=Lstar_Lsun, f_dg=f_dg, opac_filename=opac_filename)
    inp.set_molecule(mol_abun, mol_name, mol_iline, mol_rlim_au=1000)
    inp.set_view(dpc=dpc, incl_deg=incl_deg)
    inp.set_observation_grid(sizex_au, sizey_au, pixsize_au, vwidth_kms, dv_kms)
    inp.set_convolution_info(beam_maj_asec, beam_min_asec, beam_pa_deg, v_reso_kms)
    inp.show_all_input_parameters()

    return inp

class InputParams:
    home_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    def __init__(self, object_name, debug, omp, n_thread, dpath_run="./run", dname_radmc="radmc", dpath_radmc=None, kinemo_pkl="kinemo.pkl", dpath_radmc_storage=f"{home_dir}/radmc_storage"):
        self.object_name = object_name
        self.debug = debug
        self.omp = omp
        self.n_thread = n_thread
        self.kinemo_pkl = kinemo_pkl
        self.dpath_run = os.path.abspath(dpath_run)
        self.dpath_radmc = dpath_radmc if dpath_radmc is not None else os.path.abspath(os.path.join(dpath_run, dname_radmc))
        self.dpath_radmc_storage = os.path.abspath(dpath_radmc_storage)

class InputSet:
    def __init__(self, object_name, debug=False, omp=False, n_thread=1):
        self.mod = InputParams(object_name, debug, omp, n_thread)
        self.rmc = InputParams(object_name, debug, omp, n_thread)
        self.obs = InputParams(object_name, debug, omp, n_thread)

    def set_log_square_grid_2d(self, r_in_au, r_out_au, theta_in_rad, theta_out_rad,
                              phi_in_rad=0, phi_out_rad=np.pi*2, dr_au_100=3, assratio=1/4, nphi=1):
        nr = int(100/dr_au_100 * np.log10(r_out_au/r_in_au) * 2.3)
        nth = self._get_square_ntheta([r_in_au, r_out_au], nr, assratio=assratio)
        nph = nphi
        self.mod.ri_ax = np.logspace(np.log10(r_in_au*cst.au), np.log10(r_out_au*cst.au), nr+1)
        self.mod.ti_ax = np.linspace(theta_in_rad, theta_out_rad, nth+1)
        self.mod.pi_ax = np.linspace(phi_in_rad, phi_out_rad, nph+1)

    @staticmethod
    def _get_square_ntheta(r_range, nr, assratio=1):
        dlogr = np.log10(r_range[1]/r_range[0])/nr
        dr_over_r = 10**dlogr -1
        return int(round(0.5*np.pi/dr_over_r /assratio ))

    def set_model_params(self, envelope_model="CM", cavangle_deg=0, mean_mol_weight=2.3,
                         T=None, CR_au=None, M_Msun=None, t_yr=None, Omega=None, j0=None, Mdot_Msyr=None,
                         ccw_rotation=False, inenv_density_mode="default", rho0_1au=1e-10, rho_index=-1.5,
                         add_outenv=True, add_disk=False, Tdisk=30, frac_Md=0.1, kinemo_pkl="kinemo.pkl"):
        self.mod.kinemo_pkl = kinemo_pkl
        self.mod.Tenv = T
        self.mod.Mdot_Msyr = Mdot_Msyr
        self.mod.CR_au = CR_au
        self.mod.M_Msun = M_Msun
        self.mod.t_yr = t_yr
        self.mod.Omega = Omega
        self.mod.j0 = j0
        self.mod.mean_mol_weight = mean_mol_weight
        self.mod.envelope_model = envelope_model
        self.mod.cavangle_deg = cavangle_deg
        self.mod.mean_mol_weight = mean_mol_weight
        self.mod.rot_ccw = ccw_rotation
        self.mod.inenv_density_mode = inenv_density_mode # "Shu" "powerlaw"
        self.mod.rho0_1au = rho0_1au
        self.mod.rho_index = rho_index
        self.mod.add_outenv = add_outenv
        self.mod.add_disk = add_disk
        self.mod.Tdisk = Tdisk
        self.mod.frac_Md = frac_Md

    def set_radmc(self, Lstar_Lsun=1, f_dg=0.01, temp_mode="mctherm", nphot=1e5, opac_filename="silicate", scattering=False):
        self.rmc.Lstar = Lstar_Lsun * cst.Lsun
        self.rmc.Rstar = 1 * cst.Rsun
        self.rmc.f_dg = f_dg
        self.rmc.temp_mode = temp_mode
        self.rmc.nphot = nphot
        self.rmc.mean_mol_weight = self.mod.mean_mol_weight
        self.rmc.opac = opac_filename
        self.rmc.scattering_mode_max = 1 if scattering else 0
        self.rmc.T_const = None

    def set_userdefined_temperature(self, mode="const", const_T=None, const_vfwhm=None, temp_func=None):
        if mode=="const":
            if const_vfwhm is not None:
                self.rmc.const_T = self.mol_mass * const_vfwhm**2 /(16 * np.log(2) * cst.kB )
            elif const_T is not None:
                self.rmc.const_T = const_T
            elif const_T is None :
                self.rmc.const_T = self.Tenv

        elif mode == "user":
            # arg: GridModel class
            # T0_lam = 10
            # qT_lam = -1.5
            # temp_func= lambda x: T0_lam * (x.rr/1000)**qT_lam
            self.rmc.use_user_temp = True
            self.rmc.usr_temp_func = temp_func

    def set_molecule(self, mol_abun, mol_name, mol_iline, mol_rlim_au=1000):
        self.rmc.calc_line = True
        self.rmc.mol_abun = mol_abun
        self.rmc.mol_name = mol_name
        self.rmc.mol_mass = {"c18o":30, "cch":25}[self.rmc.mol_name] * cst.amu
        self.rmc.iline = mol_iline
        self.rmc.mol_rlim  = mol_rlim_au * cst.au
        self.rmc.mfrac_H2 = 0.74

    def set_view(self, dpc=100, incl_deg=0, phi_deg=0, posang_deg=0):
        self.obs.dpc = dpc
        self.obs.incl_deg = incl_deg
        self.obs.phi_deg = phi_deg
        self.obs.posang_deg = posang_deg

    def set_observation_grid(self, sizex_au=2000, sizey_au=2000, pixsize_au=10, vwidth_kms=None, dv_kms=None, sizex_asec=None, sizey_asec=None):
        self.obs.sizex_au = sizex_au if sizex_asec is None else sizex_asec*self.dpc
        self.obs.sizey_au = sizey_au if sizey_asec is None else sizey_asec*self.dpc
        self.obs.pixsize_au = pixsize_au
        self.obs.vwidth_kms = vwidth_kms
        self.obs.dv_kms = dv_kms
        self.obs.iline = self.rmc.iline
        self.obs.mol_name = self.rmc.mol_name

    def set_convolution_info(self, beam_maj_asec, beam_min_asec, beam_pa_deg, v_reso_kms, convolver="fft"):
        self.obs.beam_maj_au = beam_maj_asec * self.obs.dpc
        self.obs.beam_min_au = beam_min_asec * self.obs.dpc
        self.obs.beam_pa_deg = beam_pa_deg
        self.obs.vreso_kms = v_reso_kms
        self.obs.convmode = convolver

    def show_all_input_parameters(self):
        for  k, v in self.__dict__.items():
            print(f"{k} input parameters:")
            for kk, vv in v.__dict__.items():
                if np.isscalar(vv):
                    print(f"{kk: <20} = {vv: <10}")
                elif vv is None:
                    print(f"{kk: <20} = None ")
                else:
                    print(f"{kk: <20} = [{vv[0]:.3g}:{vv[-1]:.3g}]")
            else:
                print("")


