#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from envos.config import Config
from envos.model_generator import ModelGenerator  # Grid, KinematicModel
from envos.obs import ObsSimulator
import envos.log as log
# Above will go to: ?
# import envos
# envos.Config
# envos.ModelGenerator
# ...

from plot_example import *

#switch = [1, 0, 0, 0]
#calc = [1, 1, 1, 1]
#switch = [1] * 4

log.enable_saving_output()

config = Config()

config.set_grid(
    rau_lim=[1, 1000],
    dr_to_r=0.04,
    aspect_ratio=1 / 4
)

config.set_physical_parameters(
    CR_au=130,
    Ms_Msun=0.2,
    Mdot_smpy=4.5e-6,
    meanmolw=2.3,
    cavangle_deg=40
)


config.set_model_input(
    inenv="CM",
    outenv="TSC",
    #disk="exptail"
)

config.set_radmc_input(
    nphot=1e6,
    n_thread=12,
    f_dg=0.01,
    opac="MRN20",
    Lstar_Lsun=1.0,
    molname="c18o",
    molabun=1e-17,
    iline=3,
    mol_rlim=1000.0,
)

config.set_observation_input(
        omp=True,
        nthread=10,
        sizex_au=1000,
        pixsize_au=50,
        vwidth_kms=3,
        dv_kms=0.2,
        beam_maj_au=100,
        beam_min_au=100,
        vreso_kms=0.1,
        beam_pa_deg=0,
        convmode="null",
        incl=85,
        phi=0,
        posang=90,
        dpc=140,
        iline=3, # already set radmc input
        molname="c18o" # already set radmc input
)

mgen = ModelGenerator(config)
mgen.calc_kinematic_structure()
kstr = mgen.get_kinematic_structure()

mgen.calc_thermal_structure()
model = mgen.get_model()
#plot_density_map(model)
#plot_temperature_map(model)

#r_crit = model.ppar.cs * model.ppar.t
#start_points = [(r_crit, np.radians(deg)) for deg in range(0, 90, 10)]
#streamline.calc_streamlines_from_model(
#    model, ["rhogas", "Tgas"], ["g/cm3", "K"], start_points
#)

#osim = ObsSimulator(dpc=dpc, n_thread=12)
osim = ObsSimulator(config)

#cont = osim.observe_cont(1249)
#plot_mom0_map(cont)
#exit()


odat_line = osim.observe_line()
PV = odat_line.make_PV_map(pangle_deg=0)
plot_pvdiagram(
    PV,
    dpath_fig=dpath_fig,
    n_lv=5,
    Ms_Msun=0.2,
    rCR_au=150,
    f_crit=0.1,
    mapmode="grid",
)
#print(odat_line.Ippv.shape, odat_line.xau.shape)
#print(vars(odat_line))
#plot_lineprofile(odat_line)


import psutil
mem = psutil.virtual_memory()
print("memory results:")
print(mem.percent)
print(mem.total)
print(mem.used)
print(mem.available)

exit()
if switch[2]:
    dpc = 140
    osim = ObsSimulator(dpc=dpc, n_thread=12)
    osim.set_resolution(1000, 1000, pixsize_au=10, vwidth_kms=6, dv_kms=0.04)
    osim.set_convolver(0.7 * dpc, 0.7 * dpc, vreso_kms=0.1)

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

plot_lineprofile(odat_line)
plot_pvdiagram(
    PV,
    dpath_fig=dpath_fig,
    n_lv=5,
    Ms_Msun=0.2,
    rCR_au=150,
    f_crit=0.1,
    mapmode="grid",
)

PV = sobs.PVmap(fitsfile="PVmodel.fits", dpc=inp.obs.dpc)
PV_ref = sobs.PVmap(fitsfile="2mm_spw1_C3H2_pv.fits", dpc=inp.obs.dpc)
plot_pvdiagram(
    PV_ref,
    out="pvd_obs",
    dpath_fig=dpath_fig,
    n_lv=5,
    Ms_Msun=0.2,
    rCR_au=150,
    f_crit=0.1,
    mapmode="grid",
)
rangx = [-700, 700]
rangv = [-2, 2]
ind_x_1 = [np.abs(PV.xau - xlim).argmin() for xlim in rangx]
ind_v_1 = [np.abs(PV.vkms - vlim).argmin() for vlim in rangv]
ind_x_2 = [np.abs(PV_ref.xau - xlim).argmin() for xlim in rangx]
ind_v_2 = [np.abs(PV_ref.vkms - vlim).argmin() for vlim in rangv]

cor = pvcor.calc_correlation(
    PV_ref.Ipv[ind_v_2[0] : ind_v_2[1], ind_x_2[0] : ind_x_2[1]],
    PV.Ipv[ind_v_1[0] : ind_v_1[1], ind_x_1[0] : ind_x_1[1]],
    method="ZNCC",
    threshold=0,
    with_noise=False,
)
print("Correlateness:", cor)
