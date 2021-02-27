#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tracemalloc
import envos
import plot_example as pe

tracemalloc.start()

def synth_obs(config, read_model=0, read_odat=0):
    if read_odat:
        odat = envos.read_obsdata("run/lineobs.pkl")

    else:
        if read_model:
            model = envos.read_model("run/model.pkl")
        else:
            mg = envos.ModelGenerator(config)
            mg.calc_kinematic_structure()
            mg.calc_thermal_structure()
            model = mg.get_model()
            model.save_pickle("model.pkl")

        pe.plot_density_map(model, trajectries=False)
        pe.plot_temperature_map(model, trajectries=False)

        osim = envos.ObsSimulator(config)
        osim.set_model(model)
        odat = osim.observe_line()
        odat.save_instance(filename="lineobs.pkl")

    PV = odat.get_PV_map(pangle_deg=0)
    pe.plot_mom0_map(odat, pangle_deg=[0] )
    pe.plot_pvdiagram(PV, f_crit=0.1)

#--------------------------------------------------------#

conf = envos.Config(
    n_thread=12,
    rau_in=10,
    rau_out=1000,
    dr_to_r=0.02,
    aspect_ratio=1,
    CR_au=200,
    Ms_Msun=0.2,
    T=10,
    cavangle_deg=45,
    f_dg=0.01,
    opac="MRN20",
    Lstar_Lsun=1.0,
    molname="c18o",
    molabun=1e-17,
    iline=3,
    size_au=2000,
    pixsize_au=4,
    vfw_kms=6,
    dv_kms=0.05,
    beam_maj_au=50,
    beam_min_au=50,
    vreso_kms=0.1,
    beam_pa_deg=0,
    convmode="normal",
    incl=90,
    posang=0,
    dpc=100,
)

conf = conf.replaced(fig_dir = "./run/fig_fid")
import envos.gpath as gp
print(conf)
print(gp.fig_dir, gp.run_dir)
synth_obs(conf)
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
print('[ Top 3 ]')
for stat in top_stats[:3]:
    print(stat)
exit()

for M in [0.05, 0.1, 0.4, 0.8]:
    conf = conf.replaced(Ms_Msun=M, fig_dir = f"./run/fig_M{M}")
    synth_obs(conf)

for cr in [50, 100, 200, 400, 800]:
    conf = conf.replaced(CR_au=cr, fig_dir = f"./run/fig_cr{cr}")
    synth_obs(conf)






#import copy
#conf = copy.copy(config)
#for M in [0.05, 0.1, 0.4, 0.8]:
#    conf.set_run_config(
#        rundir = f"run_{M}"
#    )
#    conf.ppar.Ms_Msun = M
#    do_all(conf)

#tools.show_used_memory()

print("\a\a\a\a \a\a\a\a")

exit()

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
