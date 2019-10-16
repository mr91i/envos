#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import natconst as cst
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
#mpl.use('Agg')

from matplotlib import pyplot as plt
plt.switch_backend('agg')
from matplotlib import cm
import matplotlib.pylab as plb
from radmc3dPy.image import *	 # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
from radmc3dPy.analyze import *  # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
from radmc3dPy.natconst import * # Make sure that the shell variable PYTHONPATH points to the RADMC-3D python directory
import subprocess
#import myplot as mp
#import plotter as pl

DensityMap = 0
DensProf   = 1
TempProf   = 1
Obs		   = 0
SED		   = 1	
Line	   = 0
ChannelMap = 0
PVd		   = 0

Fits  = 0

## Global parameters
iline = 2
incl = 90 
phi  = 0
posang = 0
n_thread = 16

def plot_Physical_Profile():

	data = readData(ddens=True,dtemp=True,binary=False)
	rr, tt = np.meshgrid(data.grid.x/cst.au, data.grid.y, indexing='xy')
	plt.xlim([0,500])
	plt.ylim([0,500])

	if DensProf:
		fig  = plt.figure()
		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt) , np.log10(data.rhodust[:,:,0,0].T), 30)
		cb = plb.colorbar(c)
		fig.savefig("dens.pdf")

		fig  = plt.figure()
		plb.plot( data.grid.x/cst.au , data.rhodust[:,-1,0,0].T)
		plt.xlim([10,10000])
		#plt.ylim([1,1000])	
		plt.xscale('log')
		plt.yscale('log')
		fig.savefig("dens_plf.pdf")
		plb.show()

	if TempProf:
		fig  = plt.figure()
		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt), data.dusttemp[:,:,0,0].T, 30)
		cb = plb.colorbar(c)
		fig.savefig("temp.pdf")

		fig  = plt.figure()
		plb.plot( data.grid.x/cst.au , data.dusttemp[:,-1,0,0].T)
		plt.xlim([10,10000])
		plt.ylim([1,1000])	
		plt.xscale('log')
		plt.yscale('log')
		fig.savefig("temp_plf.pdf")
		plb.show()

		

def Synthetic_Observation(wl=5):
	makeImage(npix=200, incl=incl, phi=phi, posang=posang, wav=wl , sizeau=500 )	 # This calls radmc3d 
	fig		  = plt.figure()
	obs_data  = readImage()
	plotImage(obs_data, log=True, cmap=cm.hot, maxlog=6, bunit='snu', dpc=140, arcsec=True)
	fig.savefig("obs.pdf")


def make_fits_data():
	common = "incl %d phi %d posang %d setthreads %d "%(incl,phi,posang,n_thread)
	option = "fluxcons "
	cmd = "radmc3d image iline %d widthkms 8 linenlam 100 npix 200 sizeau 500 "%(iline) + common + option
#	cmd = "radmc3d image  iline %d widthkms 10 linenlam 200 npix 200 sizeau 400 "%(iline) + common + option
#	cmd = "radmc3d image  iline %d widthkms 5 linenlam 2 zoomau -200 200 -20 20 npixx 160 npixy 16 truepix "%(iline) + common + option
	subprocess.call(cmd,shell=True)
#	if "ERROR" in ret :
#		exit(1)
	fig		  = plt.figure()
	obs_data  = readImage()
	print(vars(obs_data))
	print(obs_data.image.shape)
	obs_data.writeFits(fname='obs.fits', dpc=140 )
#	fitsheadkeys = {'MEMO': 'This is just test'  }
#	obs_data.writeFits(fname='obs.fits', dpc=140 , fitsheadkeys=fitsheadkeys )


def calc_Line_profile():
	os.system("radmc3d spectrum iline %d incl %d phi %d widthkms 2 linenlam 4 setthreads 4 zoomau -1 1 -1 1 truepix"%(iline,incl,phi))
	fig = plt.figure()
	s	= readSpectrum()
	mol = readMol('co')
	plotSpectrum(s,mol=mol,ilin=iline,dpc=140.)
	plt.show()
	fig.savefig("line.pdf")


def calc_PVdiagram():
	p_wl_int = []
	intens = [] 
	mol   = readMol('co')
	freq0  = mol.freq[iline-1]
	dx = 2
	angle = "incl %d phi %d"%(incl,phi)	
	vreso = "widthkms 6 linenlam 72"
	x_ax = np.arange(0,300,dx)
	for x in x_ax:
		pos = ( x-0.5*dx, x+0.5*dx , -dx , dx)
		zoomau = "zoomau %f %f %f %f"%(pos)
		os.system("radmc3d spectrum iline %d %s %s %s setthreads 16 nofluxcons"%(iline,vreso,angle,zoomau))
		s=readSpectrum()
		freq = 1e4*cc/s[:,0] 
		vkms = 2.9979245800000e+10*(freq0-freq)/freq0/1.e5 
		s[:,0] = vkms
		p_wl_int.append(s)
		intens.append( s[:,1] )
	fig= plt.figure()
	c  = plb.contourf( x_ax ,vkms , np.array(intens).T , 30)
	cb = plb.colorbar(c)
	fig.savefig("PVd.pdf")


#
# Make and plot the SED as seen at 1 pc distance
#
def calc_SED():
	os.system("radmc3d sed incl %d phi %d setthreads %d noscat "%(incl,phi,n_thread))
	fig3  = plt.figure()
	s	  = readSpectrum()
	lam   = s[:,0]
	nu	  = 1e4*cc/lam
	fnu   = s[:,1]
	nufnu = nu*fnu
	plt.plot(lam,nufnu)
	plt.xscale('log')
	plt.yscale('log')
	plt.axis([1e-1, 1e4, 1e-10, 1e-4])
	plt.xlabel('$\lambda\; [\mu \mathrm{m}$]')
	plt.ylabel('$\\nu F_\\nu \; [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$')
	fig3.savefig( "sed.pdf")


if __name__=='__main__':
	
	if DensProf or TempProf:
		plot_Physical_Profile()

	if Obs:
		Synthetic_Observation()

	if Fits:
		make_fits_data()

	if Line:
		calc_Line_profile()

	if PVd:
		calc_PVdiagram()

	if SED:
		calc_SED()	
