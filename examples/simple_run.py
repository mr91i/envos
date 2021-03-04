#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import envos

config = envos.Config(
    run_dir="./run",
    n_thread=1,
    rau_in=10,
    rau_out=1000,
    dr_to_r=0.1,
    aspect_ratio=1,
    CR_au=100,
    Ms_Msun=0.1,
    T=10,
    cavangle_deg=45,
    f_dg=0.01,
    opac="MRN20",
    Lstar_Lsun=1.0,
    molname="c18o",
    molabun=1e-20,
    iline=3,
    size_au=2000,
    pixsize_au=25,
    vfw_kms=6,
    dv_kms=0.1,
    beam_maj_au=50,
    beam_min_au=50,
    vreso_kms=0.2,
    beam_pa_deg=0,
    convmode="normal",
    incl=90,
    posang=0,
    dpc=100,
)

print(config)

mg = envos.ModelGenerator(config)
mg.calc_kinematic_structure()
mg.calc_thermal_structure()
model = mg.get_model()

# Once you save a data file as:
#   model.save_pickle("model.pkl")
# you can read the model as
#   model = envos.read_model("run/model.pkl")

print(model)

osim = envos.ObsSimulator(config)
osim.set_model(model)
odat = osim.observe_line()

# Once you save a data file as:
#   odat.save_instance(filename="lineobs.pkl")
# you can read the data as:
#   odat = envos.read_obsdata("run/lineobs.pkl")

PV = odat.get_PV_map(pangle_deg=0)

print(PV)

# To visualize the PV diagram, one can use "envos.plot_tools"
# which provide functions to plot results.
envos.plot_tools.plot_pvdiagram(PV)

