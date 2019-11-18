#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd
import natconst as cst
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d

#mpl.use('Agg')

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from matplotlib import cm
#import matplotlib.pylab as plb
#from radmc3dPy.image import *	 # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
#from radmc3dPy.analyze import *  # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
#from radmc3dPy.natconst import * # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
import subprocess

#dn_home = os.path.dirname( os.path.abspath(__file__) )
dn_here = os.path.dirname(os.path.abspath(__file__)) 
dn_home = os.path.abspath(dn_here + "/../../")
sys.path.append(dn_home)
dn_fig = dn_home + '/fig/'
print("Execute %s:\n"%__file__,dn_home)

exit()

#import myplot as mp
#import plotter as pl

DensityMap	= 0
DensProf	= 1
TempProf	= 1
ChemTimeProf= 1
Obs			= 0
SED			= 0	
Line		= 0
ChannelMap	= 0
PVd			= 0
tau_surf	= 0
Fits		= 1

## Global parameters
iline = 2
incl = 90 #90 
phi  = 0
posang = 0
n_thread = 1

def plot_Physical_Profile():

	data = readData(ddens=True,dtemp=True,binary=False)
	rr, tt = np.meshgrid(data.grid.x/cst.au, data.grid.y, indexing='xy')
	plt.xlim([0,500])
	plt.ylim([0,500])

	if DensProf:
		fig  = plt.figure()
		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt) , np.log10(data.rhodust[:,:,0,0].T), 30)
		cb = plb.colorbar(c)
		fig.savefig(dn_fig+"dens.pdf")

		fig  = plt.figure()
		print( data.rhodust.shape)
		plb.plot( data.grid.x/cst.au , data.rhodust[:,-1,0,0].T)
		plt.xlim([10,10000])
		#plt.ylim([1,1000])	
		plt.xscale('log')
		plt.yscale('log')
		fig.savefig(dn_fig+"dens_plf.pdf")
		plb.show()

	if TempProf:
		fig  = plt.figure()
		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt), data.dusttemp[:,:,0,0].T, 30)
		cb = plb.colorbar(c)
		fig.savefig(dn_fig+"temp.pdf")

		fig  = plt.figure()
		plb.plot( data.grid.x/cst.au , data.dusttemp[:,-1,0,0].T)
		plt.xlim([10,10000])
		plt.ylim([1,1000])	
		plt.xscale('log')
		plt.yscale('log')
		fig.savefig(dn_fig+"temp_plf.pdf")
		plb.show()

	if ChemTimeProf:
		def chemical_dest_time(rho, T, spc="CCH"):
			if spc=="CCH":
				k = 1.2e-11*np.exp(-998/T)
				return 1/(rho/cst.mp/2* k)
		t_dest = chemical_dest_time(data.rhodust[:,:,0,0].T*100, data.dusttemp[:,:,0,0].T)
		t_dyn = 5.023e6 * np.sqrt( rr **3 /0.18 ) ## sqrt(au^3/GMsun) = 5.023e6
		fig  = plt.figure()
#		import pdb; pdb.set_trace()
		plb.plot( data.grid.x/cst.au , (t_dest/t_dyn)[-1] )
		plt.xlim([10,10000])
		plt.ylim([1e-6,1e6])
		plt.xscale('log')
		plt.yscale('log')
		fig.savefig(dn_fig+"tchem_plf.pdf")
		plb.show()		
	
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
	
	if DensProf or TempProf or ChemTimeProf:
		plot_Physical_Profile()
	
	if tau_surf:
		find_tau_surface()

