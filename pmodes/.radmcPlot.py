#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,	absolute_import, division
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import natconst as cst
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from scipy import interpolate, integrate


from matplotlib import pyplot as plt
plt.switch_backend('agg')
from matplotlib import cm
import matplotlib.pylab as plb
from radmc3dPy.image import *	# Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
from radmc3dPy.analyze import *  # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
#from radmc3dPy.natconst import * # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
#from radmc3dPy.analyze import 
# If error raise bacause of "No module named _libfunc",
# please add the path "~/bin/python/radmc3dPy/models" to $PYTHONPATH
# 

#dn_home = os.path.dirname( os.path.abspath(__file__) )
#dn_here = os.path.dirname(os.path.abspath(__file__)) 
dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + "/../../")
sys.path.append(dn_home)
dn_fig = dn_home + '/fig/'
print("Execute %s:\n"%__file__,dn_home)
os.chdir(dn_here)



#import myplot as mp
import plotter as mp

DensityMap	= 0
DensProf	= 1
TempProf	= 1
VelProf		= 1
ChemTimeProf= 1
Emissivity_prof =1
Line		= 0
ChannelMap	= 0
PVd			= 0
tau_surf	= 0
Fits		= 1

## Global parameters
iline = 2
incl = 85 #90 
phi  = 0
posang = 0
n_thread = 1

class RadmcData:
	def __init__(self, ddens=True, gdens=True, gtemp=True, dtemp=True, gvel=True):
		self.xauc = None
		self.xaui = None
		self.rrc = None
		self.ttc = None
		self.rri = None
		self.tti = None

		self.dn_fig = None

	def chemical_dest_time(rho, T, spc="CCH"):
		if spc=="CCH":
			k = 1.2e-11*np.exp(-998/T)
			return 1/(rho/cst.mp/2* k)

	def total_intensity_model():
		f_rho = interpolate.interp1d( self.xauc , self.data.ndens_mol[:,-1,0,0] , fill_value = (0,) , bounds_error=False ,	kind='cubic')
		def f_totemsv(x, y):
			return f_rho( (x**2 + y**2)**0.5 ) 
		#x_im = data_im.x/cst.au
		yax = np.linspace(self.x_im[0], self.x_im[-1], 10000)
		return np.array([ integrate.simps(f_totemsv( x, yax ), yax ) for x in self.x_im ])	
	
	def total_intenisty_image():
		image = data_im.image.sum(axis=2) if 1 else -integrate.simps(data_im.image, data_im.freq, axis=2)
		return np.average(image, axis=1)

	def read(self, ):
		self.data = readData(ddens=True, gdens=True, gtemp=True, dtemp=True, gvel=True, ispec='c18o', binary=False)
		if isinstance(data.gastemp,float) and  data.gastemp == -1:
			data.gastemp = data.dusttemp
		self.data_im = pd.read_pickle(dn_here+'/obs.pkl')
		self.x_im = self.data_im.x/cst.au
		self.xauc = data.grid.x/cst.au
		self.xaui = data.grid.xi/cst.au
		self.rrc, self.ttc = np.meshgrid(self.xauc , data.grid.y, indexing='xy')
		self.rri, self.tti = np.meshgrid(self.xaui , data.grid.yi, indexing='xy')
		self.RR = rrc*np.sin(ttc)
		self.zz = rrc*np.cos(ttc)

		# Chemical lifetime
		self.t_dest = chemical_dest_time(self.data.ndens_mol[:,:,0,0].T/1e-7, self.data.gastemp[:,:,0,0].T)
		self.t_dyn = 5.023e6 * np.sqrt( self.rrc**3 /0.18 ) ## sqrt(au^3/GMsun) = 5.023e6

	def plotall(self, dn_fig, dens=True, temp=True, vel=True)
		if dens:
			plot_plofs(data.ndens_mol[:,:,0,0].T, "nden", yl=[1e-3,1e3], logz=True)

		if temp:
			plot_plofs(data.gastemp[:,:,0,0].T, "temp", yl=[1,1000], ylb='Temperature [K]')

		if vel:
			plot_plofs(data.gasvel[:,:,0,0].T/1e5, "gvelr", yl=[1e-1,1e1], logy=False)
			plot_plofs(data.gasvel[:,:,0,2].T/1e5, "gvelp", yl=[1e-1,1e1], logy=False)

		if totint:
			plb.plot(data_im.x/cst.au , image, label="Total Intensity: Synthetic Obs",lw=2.5)
			plb.plot(x_im,	emsv  * ( image[ len(image)//2 ] + image[ len(image)//2-1 ])/(emsv[ len(emsv)//2 ] + emsv[ len(emsv)//2 - 1]), label="Total Intensity: Expectation", ls='--', lw=1.5)
			plt.legend()
			plt.ylim([0,None])
			fig.savefig(dn_fig+"emsv_img_plf.pdf")


	def plot_plofs(self, d, fname, ylim=None, logy=True, yl=None):
		plot_2d(d, fname, ylim=None, logy=True, yl=None)
		plot_1d(d, fname, ylim=None, logy=True, yl=None)

	def plot_2d(self, d, fname, ylim=None, logy=True, yl=None):
		mp.plot(self.RR, self.zz, d, 
				self.dn_fig+fname+".pdf",
				xl='x [au]', yl='y [au]', cbl=yl, 
				xlim=[0,500], ylim=[0,500], cblim=ylim, 
				logcb=logy)

	def plot_1d(self, d, fname, ylim=None, logy=True, yl=None):
		mp.plot(d[-1,:], self.dn_fig+fname+"_plf.pdf", x=self.xauc, 
				xl='x [au]', yl=yl, 
				xlim=[10,10000], ylim=ylim, 
				logx=True, logy=logy)	 

	def plot_total_intensity(self):



	def plot_chemical_time(self):
		plot_plof( t_dest/t_dyn, "tche", yl=[1e-10,1e10])
	
	
def find_tau_surface():
	common = "incl %d phi %d posang %d setthreads %d "%(incl,phi,posang,n_thread)
	wl = "iline %d "%iline	#	"lambda %f "%wl
	cmd = "radmc3d tausurf 1 npix 100 sizeau 500 " + common + wl
#	subprocess.call(cmd,shell=True)
	a=readImage()
	fig = plt.figure()
	c	= plb.contourf( a.x/cst.au , a.y/cst.au , a.image[:,:,0].T.clip(0)/cst.au, levels=np.linspace(0.0, 30, 20+1) )
	cb = plb.colorbar(c)
	plt.savefig(dn_fig+"tausurf.pdf")

if __name__=='__main__':

#	if DensProf or TempProf or ChemTimeProf:
#		plot_Physical_Profile()
	RadmcData.read()
	RadmcData.plot()


#	if tau_surf:
#		find_tau_surface()


