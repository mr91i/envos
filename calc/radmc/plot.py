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
		fig.savefig("tchem_plf.pdf")
		plb.show()		
	

def Synthetic_Observation(wl=5):
	makeImage(npix=200, incl=incl, phi=phi, posang=posang, wav=wl , sizeau=500 )	 # This calls radmc3d 
	fig		  = plt.figure()
	obs_data  = readImage()
	plotImage(obs_data, log=True, cmap=cm.hot, maxlog=6, bunit='snu', dpc=140, arcsec=True)
	fig.savefig("obs.pdf")


def find_tau_surface():

	common = "incl %d phi %d posang %d setthreads %d "%(incl,phi,posang,n_thread)
	wl = "iline %d "%iline	#	"lambda %f "%wl
	cmd = "radmc3d tausurf 1 npix 100 sizeau 500 " + common + wl
#	subprocess.call(cmd,shell=True)
	a=readImage()
	fig = plt.figure()
	c	= plb.contourf( a.x/cst.au , a.y/cst.au , a.image[:,:,0].T.clip(0)/cst.au, levels=np.linspace(0.0, 30, 20+1) )
	cb = plb.colorbar(c)
	plt.savefig("tausurf.pdf")


def subcalc(args):
	dn , cmd = args
	print(cmd)
	if not os.path.exists(dn):
		os.makedirs(dn)
#	os.system("rm %s/*"%dn)
	os.system("cp *.inp *.dat %s/"%dn)
	os.chdir(dn)
	subprocess.call(cmd,shell=True)
	return readImage()

def make_fits_data():
	rectangle_camera = 1
	multi_threads = 1

	widthkms = 3 # 
	linenlam = 60 # ~ 0.1km/s
	sizeau	 = 1960 #1400 #	
	npix	 = 100 # ~ 0.1''
	common	 = "incl %d phi %d posang %d"%(incl,phi,posang)
	option	 = " "
#	line	 = "iline %d widthkms %f linenlam %d "%(iline,widthkms,linenlam)

	if rectangle_camera:	
		Lh	   = sizeau/2
		npix   = [ npix ,  10  ]
		zoomau = [ -Lh, Lh, -Lh*npix[1]/float(npix[0]) , Lh*npix[1]/float(npix[0]) ]
 #		npix   = [ 100 ,  1	]
#		zoomau = [ -700, 700, -7 , 7 ]

		camera	 = "npixx {:d} npixy {:d} ".format(*npix) + "zoomau {:f} {:f} {:f} {:f} truepix ".format(*zoomau)

	if multi_threads:
		from multiprocessing import Pool
		for i in range(16,0,-1):
			print(i)
			if linenlam % i == 0:
				n_thread = i 
				break
		v_ranges = np.linspace( -widthkms ,  widthkms , n_thread + 1 )
		dv = 2*widthkms/linenlam
		cmd = lambda p :  " radmc3d image iline {:d} vkms {} widthkms {} linenlam {:d} ".format( \
						iline , 0.5*(v_ranges[p+1] + v_ranges[p]), 0.5*(v_ranges[p+1]  - v_ranges[p]) , int(linenlam/float(n_thread)) ) \
						+ camera + common + option
		rets =	Pool(n_thread).map( subcalc , [ ( 'proc' + str(p) , cmd(p) ) for p in range(n_thread)	] )
		obs_data = rets[0]
#		print(vars(rets[0]))
		for ret in rets[1:]:
#			print(vars(ret))
			print(ret.image.shape)
			obs_data.image = np.append( obs_data.image, ret.image[:,:,1:], axis=2 )
			obs_data.imageJyppix = np.append( obs_data.imageJyppix, ret.imageJyppix[:,:,1:], axis=2 )
			obs_data.freq = np.append( obs_data.freq, ret.freq[1:], axis=-1 )
			obs_data.nfreq += ret.nfreq -1
			obs_data.wav = np.append( obs_data.wav, ret.wav[1:], axis=-1 )
			obs_data.nwav += ret.nwav -1 
		print(vars(obs_data))
		print(obs_data.image.shape)
	else:
		cmd		 = "radmc3d image " + line + camera + common + option
#		 cmd = "radmc3d image iline 2 widthkms 12 linenlam 100 sizeau 500 npix 200 truepix " + common + option
		subprocess.call(cmd,shell=True)
		obs_data = readImage()

	obs_data.writeFits(fname='obs.fits', dpc=140 )



#def make_fits_data():
#	camera	 = "npix %d sizeau %f truepix "%(npix,sizeau)
#	subprocess.call(cmd,shell=True)
#	fig		  = plt.figure()
#			
#	obs_data  = readImage()
#	print(vars(obs_data))
#	print(obs_data.image.shape)
#	obs_data.writeFits(fname='obs.fits', dpc=140 )
#	fitsheadkeys = {'Ms':  }
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
	
	if DensProf or TempProf or ChemTimeProf:
		plot_Physical_Profile()

	if Obs:
		Synthetic_Observation()
	
	if tau_surf:
		find_tau_surface()

	if Fits:
		make_fits_data()

	if Line:
		calc_Line_profile()

	if PVd:
		calc_PVdiagram()

	if SED:
		calc_SED()	
