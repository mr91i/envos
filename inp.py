#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Here we can controll the full paramters used in pmodes.
#

import sys
import numpy as np
sys.path.append('./pmodes')
import cst

class InputParams:
    def __init__(self):
        pass

def get_square_ntheta(r_range, nr):
    dlogr = np.log10(r_range[1]/r_range[0])/nr
    dr_over_r = 10**dlogr -1
    return int(round(0.5*np.pi/dr_over_r))

# Common params
object_name = "L1527"
dpc = 140
Tenv = 10
beam_scale = 1.0 #
dr_100 = 1
debug = 1

##########################################################################################
# Kinematic structure
model = InputParams()
model.ire_model = "CM" # "CM" or "Simple"
model.fn_model_pkl = f"model_{object_name}.pkl"

# set grid of model
model.r_in = 1*cst.au
model.r_out = 1e4*cst.au
model.nr = int(100/dr_100 * np.log10(model.r_out/model.r_in) * 2.3)
model.theta_in = 0
model.theta_out = np.pi/2
model.ntheta = get_square_ntheta([model.r_in, model.r_out], model.nr)
model.phi_in = 0
model.phi_out = 2*np.pi
model.nphi = 1

# set
model.Tenv_K = Tenv
model.Mdot = None # 1e-5*cst.Mpyr
model.rCR_au = 150 # free parameter
model.Mstar_Msun = 0.2 # free parameter
model.t = None
model.Omega = None
model.j0 = None
model.cavity_angle = 45 # degree from z-axis, disk_an

# set disk params
model.disk = False # True
model.Tdisk = 0
model.disk_star_fraction = 0.1 # not used now

# additional options
model.calc_TSC = False
model.simple_density = False
model.counterclockwise_rotation = False
##########################################################################################

##########################################################################################
# Thermal Structure
radmc = InputParams()
radmc.fn_model_pkl = model.fn_model_pkl
radmc.fn_radmcset_pkl = 'set_radmc.pkl'

# set model parameter
radmc.Lstar = 2.75 * cst.Lsun
radmc.Rstar = 1 * cst.Rsun
radmc.fdg = 0.01  # 0.01

radmc.line = True
radmc.mol_abun = 1e-17
radmc.mol_name = {1:"c18o", 2:"cch", 3:"c18omod"}[1]  # c18o or cch
radmc.m_mol = {"c18o":28, "cch":25}[radmc.mol_name] * cst.amu
iline = {"c18o":3, "cch":28}[radmc.mol_name]
radmc.mol_radius = 1e3*cst.au

radmc.v_fwhm = None # 0.04 * cst.kms  # or give Tenv
radmc.Tenv = Tenv

radmc.temp_mode = {1:"const", 2:"mctherm", 3:"lambda"}[2]
T0_lam = 10
qT_lam = -1.5
radmc.temp_lambda = lambda x: T0_lam * (x/1000)**qT_lam
radmc.opac = {1:"h2oice", 2:"MRN20", 3:"B18MRN"}[2]

# radmc params
radmc.nphot = 1000000
radmc.scattering_mode_max = 0
##########################################################################################

##########################################################################################
obs = InputParams()
obs.filename = "sobs"
obs.dpc = dpc
obs.sizex_au = 16 * obs.dpc # 16
obs.sizey_au = 16 * obs.dpc # 16
obs.pixsize_au = 0.08 * obs.dpc # def: 0.08pc = 11au
obs.incl = 85  # angle between direction to us , 0: face-on
obs.phi = 0
obs.posang = 95  # angle between y-axis, ccw is positive
obs.omp = True
obs.n_thread = 12
obs.iline = iline  # 1 # see table
obs.vwidth_kms = 6 # def : 3
obs.dv_kms = 0.04  # def : 0.08 km/s
##########################################################################################

##########################################################################################
vis = InputParams()
vis.posang_PV = obs.posang
# beam
vis.convolver = "fft"
vis.beama_au = 0.4 * dpc * beam_scale
vis.beamb_au = 0.4 * dpc * beam_scale
vis.beam_posang = 0.0 # -41
vis.vreso_kms = 0.4 * beam_scale

### Plot
vis.plotmode_PV = {1:"grid", 2:"contour", 3:"contourf", 4:"contourfill"}[2]
vis.mass_estimation = 1
vis.poffset_yau = 0.0
vis.oplot_KeplerRotation = 0
vis.oplot_LocalPeak_Vax = 0
vis.oplot_LocalPeak_Pax = 0
vis.oplot_LocalPeak_2D = 0
vis.oplot_Maximums_2D = 1
##########################################################################################
