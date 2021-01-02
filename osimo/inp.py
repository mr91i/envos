#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Here we can controll the full paramters used in osimo.
#

import sys
import numpy as np
from osimo import nconst as nc
import os
from osimo import log
logger = log.set_logger(__name__, ini=True)

def gen_input(log=False, **kwargs):
    ginp = GeneralInput()
    minp = ModelInput()
    rinp = RadmcInput()
    oinp = ObsInput()

    ginp.set_params(**kwargs, log=log)
    minp.set_params(**kwargs, log=log)
    rinp.set_params(**kwargs, log=log)
    oinp.set_params(**kwargs, log=log)

    builder = InputBuilder(ginp, minp, rinp, oinp)
    inp = builder.build()
    return inp


class InputBuilder:
    def __init__(self, generalinp=None, modelinp=None, radmcinp=None, obsinp=None):
        self.generalinp = generalinp
        self.modelinp = modelinp
        self.radmcinp = radmcinp
        self.obsinp = obsinp

    def set_general_input(self, generalinp):
        self.generalinp = generalinp

    def set_model_input(self, modelinp):
        self.modelinp = modelinp

    def set_radmc_input(self, radmcinp):
        self.modelinp = radmcinp

    def set_obs_input(self, obsinp):
        self.obsinp = obsinp

    def build(self):
        self.modelinp.combine_input(self.generalinp)
        self.radmcinp.combine_input(self.generalinp)
        self.obsinp.combine_input(self.generalinp)

        self.radmcinp.update_common_variable(self.modelinp)
        self.obsinp.update_common_variable(self.modelinp)
        self.obsinp.update_common_variable(self.radmcinp)

        self.modelinp.show_all_input_parameters()
        self.radmcinp.show_all_input_parameters()
        self.obsinp.show_all_input_parameters()

        combinput = Input()
        combinput.add_params( modelinp=self.modelinp, radmcinp=self.radmcinp, obsinp=self.obsinp)
        return combinput


class Input:
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #else:
            #     logger.debug(f"Setting: {self.__class__.__name__} does not have {k} and so skipped to set it.")

    def add_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                logger.warning(f"Adding: {self.__class__.__name__} have already had {k} and so skipped to set it.")
            else:
                setattr(self, k, v)

    def combine_input(self, inputclass):
        for k, v in inputclass.__dict__.items():
            if hasattr(self, k):
                logger.warning(f"Combining: {self.__class__.__name__} have already had {k}. Please note that.")
            setattr(self, k, v)

    def update_common_variable(self, inputclass):
        for k, v in inputclass.__dict__.items():
            if hasattr(self, k):
                setattr(self, k, v)
                logger.info(f"Common-update: {k} in {self.__class__.__name__} is updated to {k} in {inputclass.__class__.__name__}.")

    def show_all_input_parameters(self):
        for  k, v in self.__dict__.items():
            #logger.info(f"{k} input parameters:")
            #for kk, vv in v.items():
           if np.isscalar(v):
               logger.info(f"{k: <20} = {v: <10}")
           elif v is None:
               logger.info(f"{k: <20} = None ")
           elif hasattr(v, "__len__"):
               logger.info(f"{k: <20} = [{v[0]:.3g}:{v[-1]:.3g}]")
           else:
               pass
        else:
           logger.info("")

home_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

class GeneralInput(Input):
    def __init__(self,
                 object_name="model",
                 debug=False,
                 omp=True,
                 n_thread=1,
                 dpath_run="./run",
                 dname_radmc="radmc",
                 dpath_radmc=None,
                 kmodel_pkl="kinemo.pkl",
                 dname_radmc_storage="radmc_storage",
                 dpath_radmc_storage=None,
                 logname="log.dat"):
        global logger

        self.object_name = object_name
        self.debug = debug
        self.omp = omp
        self.n_thread = n_thread
        self.kmodel_pkl =  kmodel_pkl
        self.dpath_run = os.path.abspath(dpath_run)
        self.dpath_radmc = dpath_radmc if dpath_radmc is not None\
                           else os.path.abspath(os.path.join(dpath_run, dname_radmc))
        self.dpath_radmc_storage = dpath_radmc_storage if dpath_radmc_storage is not None\
                                  else os.path.abspath(os.path.join(home_dir, dname_radmc_storage))
        self.fpath_log = os.path.join(self.dpath_run, logname)
        logger = log.set_logger(__name__, self.fpath_log)


class ModelInput(Input):
    def __init__(self, T=None, CR_au=None, Ms_Msun=None, t_yr=None, Omega=None, j0=None, Mdot_Msyr=None):
        self.kmodel_pkl = "kmodel.pkl"
        self.ri_ax = None
        self.ti_ax = None
        self.pi_ax = None
        self.Tenv = T
        self.Mdot_Msyr = Mdot_Msyr
        self.CR_au = CR_au
        self.Ms_Msun = Ms_Msun
        self.t_yr = t_yr
        self.Omega = Omega
        self.maxangmom = j0
        self.mean_mol_weight = 2.3
        self.cavangle_deg = 0
        self.inenv = "CM"
        self.rot_ccw = False
        self.usr_density_func = None
        self.outenv = ""
        self.disk = ""
        self.Tdisk = 30
        self.frac_Md = 0.1

    def set_log_square_grid_2d(self, rlim_au, thetalim_rad, plim_rad=(0, 2*np.pi), dr_au_100=3, assratio=1/4, nphi=1):
        nr = int(100/dr_au_100 * np.log10(rlim_au[1]/rlim_au[0]) * 2.3)
        nth = self._get_square_ntheta(rlim_au, nr, assratio=assratio)
        nph = nphi
        self.mod.ri_ax = np.logspace(np.log10(rlim_au[0]*nc.au), np.log10(rlim_au[1]*nc.au), nr+1)
        self.mod.ti_ax = np.linspace(*thetalim_rad, nth+1)
        self.mod.pi_ax = np.linspace(*philim_rad, nph+1)

    @staticmethod
    def _get_square_ntheta(r_range, nr, assratio=1):
        dlogr = np.log10(r_range[1]/r_range[0])/nr
        dr_over_r = 10**dlogr -1
        return int(round(0.5*np.pi/dr_over_r /assratio ))

class RadmcInput(Input):
    def __init__(self):
        self.Lstar = 1 * nc.Lsun
        self.Rstar = 1 * nc.Rsun
        self.f_dg = 0.01
        self.temp_mode = "mctherm"
        self.nphot = 1000000
        self.mean_mol_weight = 2.3
        self.opac = "silicate"
        self.scattering_mode_max = 0
        self.T_const = None

    def set_molecule(self, mol_abun, mol_name, mol_iline, mol_rlim_au=1000):
        self.rmc.calc_line = True
        self.rmc.mol_abun = mol_abun
        self.rmc.mol_name = mol_name
        self.rmc.mol_mass = {"c18o":30, "cch":25}[self.rmc.mol_name] * nc.amu
        self.rmc.iline = mol_iline
        self.rmc.mol_rlim  = mol_rlim_au * nc.au
        self.rmc.mfrac_H2 = 0.74

    def set_userdefined_temperature(self, mode="const", const_T=None, const_vfwhm=None, temp_func=None):
        if mode=="const":
            if const_vfwhm is not None:
                self.rmc.const_T = self.mol_mass * const_vfwhm**2 /(16 * np.log(2) * nc.kB )
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

class ObsInput(Input):
    def __init__(self):
        self.dpc = 100
        self.incl_deg = 0
        self.phi_deg = 0
        self.posang_deg = 0

        self.sizex_au = None
        self.sizey_au = None
        self.pixsize_au = None
        self.vwidth_kms = None
        self.dv_kms = None
        self.iline = None
        self.mol_name = None

        self.beam_maj_au = None
        self.beam_min_au = None
        self.beam_pa_deg = None
        self.vreso_kms = None
        self.convmode = "fft"

    def set_view(self, dpc=100, incl_deg=0, phi_deg=0, posang_deg=0):
        self.dpc = dpc
        self.incl_deg = incl_deg
        self.phi_deg = phi_deg
        self.posang_deg = posang_deg

    def set_observation_grid(self, sizex_au=2000, sizey_au=2000, pixsize_au=10, vwidth_kms=None, dv_kms=None, sizex_asec=None, sizey_asec=None):
        self.sizex_au = sizex_au if sizex_asec is None else sizex_asec*self.dpc
        self.sizey_au = sizey_au if sizey_asec is None else sizey_asec*self.dpc
        self.pixsize_au = pixsize_au
        self.vwidth_kms = vwidth_kms
        self.dv_kms = dv_kms
        self.iline = self.rmc.iline
        self.mol_name = self.rmc.mol_name

    def set_convolution_info(self, beam_maj_asec, beam_min_asec, beam_pa_deg, v_reso_kms, convolver="fft"):
        self.beam_maj_au = beam_maj_asec * self.obs.dpc
        self.beam_min_au = beam_min_asec * self.obs.dpc
        self.beam_pa_deg = beam_pa_deg
        self.vreso_kms = v_reso_kms
        self.convmode = convolver

