#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import importlib
#from osimo import mkmodel, setradmc, sobs

#from plot_example import plot_physical_model, plot_radmc_data, plot_pvdiagram, plot_mom0_map
#from header import dpath_fig
#import pvcor


#p = importlib.import_module('plot_example')

import numpy as np
from osimo.inp import gen_input
# from osimo.mkmodel import KinematicModel
from osimo.mkmodel import Grid, KinematicModel
from osimo.setradmc import RadmcController
from osimo.sobs import ObsSimulator, ObsData
import osimo.config as oc




#inp = gen_input(object_name   = "L1527",
#                r_lim_au      = [1, 10000],
#                theta_lim_rad = [0, np.pi/2],
#                dr_au_100     = 20,
#                CR_au         = 130,
#                M_Msun        = 0.2,
#                Mdot_Msyr     = 4.5e-6,
#                cavangle_deg  = 85,
#                Lstar_Lsun    = 2.75,
#                f_dg          = 0.01,
#                opac_filename = "MRN20",
#                mol_abun      = 1e-17,
#                mol_name      = "c18o",
#                mol_iline     = 3,
#                dpc           = 140,
#                incl_deg      = 85,   # angle between direction to us , 0: face-on
#                sizex_au      = 100,
#                sizey_au      = 100,
#                pixsize_au    = 10,    # def: 0.08pc = 11au
#                vwidth_kms    = 6,     # def : 3
#                dv_kms        = 0.04,  # def : 0.08 km/s
#                beam_maj_asec = 0.7,    # FWHM of beam, beam size
#                beam_min_asec = 0.7,
#                beam_pa_deg   = 0.0,    # -41
#                v_reso_kms    = 0.1,
#                log=True
#               )
###############################################

#inp = InputSet(object_name, debug=True, omp=True, n_thread=12) # oreyou
#inp.set_log_square_grid_2d(r_lim_au[0], r_lim_au[1], theta_lim_rad[0], theta_lim_rad[1], dr_au_100=dr_au_100)
#inp.set_model_params(cavangle_deg=cavangle_deg, CR_au=CR_au, M_Msun=M_Msun, Mdot_Msyr=Mdot_Msyr,
#                     ccw_rotation=True, add_outenv=False, add_disk=False)
#
#inp.set_radmc(Lstar_Lsun=Lstar_Lsun, f_dg=f_dg, opac_filename=opac_filename)
#inp.set_molecule(mol_abun, mol_name, mol_iline, mol_rlim_au=1000)
#
#inp.set_view(dpc=dpc, incl_deg=incl_deg)
#inp.set_observation_grid(sizex_au, sizey_au, pixsize_au, vwidth_kms, dv_kms)
#inp.set_convolution_info(beam_maj_asec, beam_min_asec, beam_pa_deg, v_reso_kms)
#
#inp.show_all_input_parameters()

#  km =  mkmodel.build_with_inp(inp)

#  logger = log.set_logger(__name__)
#grid = Grid(rau_lim=[1, 10000], theta_lim=[0, np.pi/2], dr_to_r0=0.03, aspect=1/4)
# ppar = PhysicalParameters(CR_au=130, M_Msun=0.2, Mdot_Msyr=4.5e-6, meanmolw=2.3)
#builder = KinematicModelBuilder()
switch = [0, 0, 0, 0]
switch = [1]*4

from plot_example import *
if switch[0]:
    model = KinematicModel()
    model.set_grid(rau_lim=[1, 10000], theta_lim=[0, np.pi/2], dr_to_r0=0.03, aspect=1/4)
    model.set_physical_params(CR_au=130, M_Msun=0.2, Mdot_Msyr=4.5e-6, meanmolw=2.3)
    model.build(inenv="CM", disk=None, outenv="TSC")
    plot_physical_model(model)

if switch[1]:
    rc = RadmcController(model=model)
    rc.set_parameters(n_thread=12, opac="MRN20", Lstar_Lsun=1, molname="c18o", molabun=1e-17, iline=3)
    rc.exe_mctherm()
    pmodel = rc.get_model()
    plot_radmc_data(pmodel)

if switch[2]:
    dpc = 140
    osim = ObsSimulator(dpc=dpc, n_thread=12)
    osim.set_resolution(1000, 1000, pixsize_au=10, vwidth_kms=6, dv_kms=0.04)
    osim.set_convolver(0.7*dpc, 0.7*dpc, vreso_kms=0.1)

if switch[3]:
    odat_cont = osim.observe_cont(1249, incl=85)
    odat_line = osim.observe_line(3, "c18o", incl=85)
    odat_line.save_instance("obs.pkl")
    odat_line.make_mom0_map()
else:
    odat_line = ObsData()
    odat_line.read_instance("run/radmc/obs.pkl")
    odat_line.set_dpc(140)
PV = odat_line.make_PV_map(pangle_deg=0)

plot_lineprofile( odat_line )
plot_pvdiagram(PV, dpath_fig=dpath_fig, n_lv=5, Mstar_Msun=0.2, rCR_au=150, f_crit=0.1, mapmode="grid")
exit()

#exit()
#rc = RadmcController(inp=inp.rmc)
#rc.initialize(model)
#rc.exe_mctherm()

#model = rc.get_model()
#osim = ObsSimulator(inp=inp.obs)
#odat_cont = osim.observe_cont(1249)
#odat_line = osim.observe_line(iline=inp.obs.iline, vwidth_kms=inp.obs.vwidth_kms, dv_kms=inp.obs.dv_kms, ispec=inp.obs.mol_name)
#odat_line.save_instance("obs.pkl")
#odat_line.make_mom0_map()

###############################################
from plot_example import plot_physical_model, plot_radmc_data, plot_pvdiagram, plot_mom0_map

plot_mom0_map(odat, inp.obs.posang)
plot_pvdiagram(PV, dpath_fig=dpath_fig, n_lv=5, Mstar_Msun=0.2, rCR_au=150, f_crit=0.1, mapmode="contour")

exit()

PV = sobs.PVmap(fitsfile="PVmodel.fits", dpc=inp.obs.dpc)
PV_ref = sobs.PVmap(fitsfile="2mm_spw1_C3H2_pv.fits",  dpc=inp.obs.dpc)
plot_pvdiagram(PV_ref, out='pvd_obs', dpath_fig=dpath_fig, n_lv=5, Mstar_Msun=0.2, rCR_au=150, f_crit=0.1, mapmode="grid")
# print(PV.xau , PV.vkms, PV_ref.xau, PV_ref.vkms)
rangx = [-700, 700]
rangv = [-2, 2]
ind_x_1 = [np.abs(PV.xau - xlim).argmin() for xlim in rangx]
ind_v_1 = [np.abs(PV.vkms - vlim).argmin() for vlim in rangv]
ind_x_2 = [np.abs(PV_ref.xau - xlim).argmin() for xlim in rangx]
ind_v_2 = [np.abs(PV_ref.vkms - vlim).argmin() for vlim in rangv]
# print(ind_x_1, ind_v_1, ind_x_2, ind_v_2)

cor = pvcor.calc_correlation(PV_ref.Ipv[ind_v_2[0]:ind_v_2[1],ind_x_2[0]:ind_x_2[1]], PV.Ipv[ind_v_1[0]:ind_v_1[1],ind_x_1[0]:ind_x_1[1]], method="ZNCC", threshold=0, with_noise=False)
print("Correlateness:", cor)
