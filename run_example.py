#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
from pmodes import mkmodel, setradmc, sobs
from pmodes.header import inp, dpath_home, dpath_radmc, dpath_fig

from plot_example import plot_pvdiagram, plot_mom0_map
from header import dpath_fig
import pvcor

p = importlib.import_module('plot_example')

switch = [0, 0, 0, 0, 0, 1]
# switch = [1]*5

if switch[0]:
    data = mkmodel.EnvelopeDiskModel(**vars(inp.model))
    p.plot_physical_model(data, dpath_fig=dpath_fig)

if switch[1]:
    setradmc.set_radmc_parameters(dpath_radmc, **vars(inp.radmc))
    # or you can input parmeters directly.
    if inp.radmc.temp_mode=="mctherm":
        setradmc.exe_radmc_therm()
    else:
        setradmc.del_mctherm_files()

if switch[2]:
    rmc_data = setradmc.PhysicalModel(dpath_radmc=dpath_radmc, dpath_fig=dpath_fig, ispec=inp.radmc.mol_name, mol_abun=inp.radmc.mol_abun, opac=inp.radmc.opac, fn_model=inp.radmc.fn_model_pkl)
    p.plot_radmc_data(rmc_data)

#if switch[3]:
#    osim = sobs.ObsSimulator(dpath_radmc, dpath_fits=dpath_radmc, **vars(inp.obs))
#    # osim.cont_observe(1249)
#    osim.observe()
#    osim.save_instance()
#
#if switch[4]:
#    obs = visualize.ObsData(fitsfile=dpath_radmc + '/' + inp.vis.filename + '.fits', **vars(inp.vis))
#    conv_info = visualize.ConvolutionInfo(inp.vis.beama_au, inp.vis.beamb_au, inp.vis.vwidth_kms, inp.vis.beam_posang)
#    obs.convolve(conv_info)
#    obs.make_mom0_map()
#    p.plot_mom0_map(obs)
#    obs.make_PV_map()
#    p.plot_pvdiagram(obs.PV_list[0])
#

if switch[3]:
    osim = sobs.ObsSimulator(dpath_radmc, dpc=inp.obs.dpc, sizex_au=inp.obs.sizex_au, sizey_au=inp.obs.sizey_au, pixsize_au=inp.obs.pixsize_au, incl=inp.obs.incl, phi=inp.obs.phi, posang=inp.obs.posang, omp=inp.obs.omp, n_thread=inp.obs.n_thread)
    # obsdata = osim.observe_cont(1249)
    iout = 1
    calc = 1
    if iout:
        if calc:
            obsdata2 = osim.observe_line(iline=inp.obs.iline, vwidth_kms=inp.obs.vwidth_kms, dv_kms=inp.obs.dv_kms, ispec=inp.radmc.mol_name)
            obsdata2.save_instance("obs.pkl")
        else:
            obsdata2 = sobs.ObsData(pklfile="obs.pkl") # when read
    else: # or
        if calc:
            osim.observe_line(iline=inp.obs.iline, vwidth_kms=inp.obs.vwidth_kms, dv_kms=inp.obs.dv_kms, ispec=inp.radmc.mol_name)
            osim.output_fits("obs.fits")
        # osim.output_instance("obs.pkl")
        else:
            obsdata2 = ObsData(fitsfile="obs.fits")

if switch[4]:
    obsdata2 = sobs.ObsData(pklfile="obs.pkl")
    obsdata2.convolve(inp.vis.beama_au, inp.vis.beamb_au, inp.vis.vreso_kms, inp.vis.beam_posang)
    obsdata2.make_mom0_map()

    PV = obsdata2.make_PV_map(inp.obs.posang)
    PV.save_fitsfile("PVmodel.fits")
    plot_mom0_map(obsdata2, inp.obs.posang)
    plot_pvdiagram(PV, dpath_fig=dpath_fig, n_lv=5, Mstar_Msun=0.2, rCR_au=150, f_crit=0.1, mapmode="contour")


if switch[5]:
    PV = sobs.PVmap(fitsfile="PVmodel.fits", dpc=inp.obs.dpc)
    PV_ref = sobs.PVmap(fitsfile="2mm_spw1_C3H2_pv.fits",  dpc=inp.obs.dpc)
    plot_pvdiagram(PV_ref, dpath_fig=dpath_fig, n_lv=5, Mstar_Msun=0.2, rCR_au=150, f_crit=0.1, mapmode="contour")
    print(vars(PV_ref))
    cor = pvcor.calc_correlation(PV_ref.Ipv, PV.Ipv, method="ZNCC", threshold=0, with_noise=False)
    print(cor)

