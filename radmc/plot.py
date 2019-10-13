#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#import set as p
import numpy as np
import pandas as pd
#import cst
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


DensityMap = 0
DensProf   = 0
TempProf   = 0
Obs		   = 1
SED		   = 0	
Line	   = 0
ChannelMap = 0
PVd		   = 1

iline = 2
incl = 90 
phi  = 0
posang = 0
#
# Make sure to have done the following beforhand:
#
#  First compile RADMC-3D
#  Then run:
#	python problem_setup.py
#	radmc3d mctherm
#

#
# View a 2-D slice of the 3-D array of the setup
#

def convert_Sph_to_Cyl( r , th , v ):
	R = r * np.sin(th)
	z = r * np.cos(th)
	return R


if DensProf or TempProf:
	data = readData(ddens=True,dtemp=True,binary=False)
	rr, tt = np.meshgrid(data.grid.x/cst.au, data.grid.y, indexing='xy')
	plt.xlim([0,500])
	plt.ylim([0,500])
	if DensProf:
		fig  = plt.figure()
		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt) , np.log10(data.rhodust[:,:,0,0].T), 30)
		cb = plb.colorbar(c)
		fig.savefig("dens.pdf")
	if TempProf:
		fig  = plt.figure()
#		c  = plb.contourf( rr*np.sin(tt) , rr*np.cos(tt), data.dusttemp[:,:,0,0].T, 30)
#		cb = plb.colorbar(c)
#		fig.savefig("temp.pdf")

		print( data.dusttemp[:,-1,0,0].T )
		plb.plot( data.grid.x/cst.au , data.dusttemp[:,-1,0,0].T)
		plt.xlim([1,1000])
		plt.ylim([1,1000])	
		plt.xscale('log')
		plt.yscale('log')
		plb.show()
		
	exit()


if Obs:

#	for wl in [3, 30 , 1400 ]:
#	wl = 5
#	makeImage(npix=200, incl=incl, phi=phi, posang=posang, wav=wl , sizeau=500)	 # This calls radmc3d 
#        makeImage(npix=200, incl=incl, phi=phi, posang=posang, sizeau=500 , iline=iline, widthkms=10, linenlam=100, fluxcons=False)      # This calls radmc3d 

	fig2  = plt.figure()
	a=readImage()
	fitsheadkeys = {'MEMO': 'This is just test'  }
	a.writeFits(fname='test3.fits', dpc=140 , fitsheadkeys=fitsheadkeys )
	plotImage(a, log=True, cmap=cm.hot, maxlog=6, bunit='snu', dpc=140, arcsec=True)
		#plotImage(a,log=True,au=True,maxlog=6,cmap='hot')
	fig2.savefig( "obs.pdf")

	if ChannelMap:
		for vkms in [0.5,1.0]:
			fig5  = plt.figure()
	#		makeImage( npix=300., incl=incl, phi=phi, sizeau=100., vkms=vkms, iline=2)
			a = readImage()
			#print(a)
			plotImage(a, arcsec=True, dpc=140., cmap=plb.cm.gist_heat)
			fig5.savefig("chmap_%f.pdf"%(vkms))

if Line:
#	os.system("radmc3d calcpop")
#	os.system('radmc3d spectrum iline 2 widthkms 1 lambda 5')
#	lam = np.linspace(	)
	os.system("radmc3d spectrum iline %d incl %d phi %d widthkms 2 linenlam 4 setthreads 4 circ zoomau -1 1 -1 1 truepix"%(iline,incl,phi))
#	makeImage( npix=100., incl=incl, phi=phi,  wav=wl, sizeau=100., vkms=2, iline=2)
	fig4  = plt.figure()
	s=readSpectrum()
	print(s)
	mol=readMol('co')
	print(mol.__str__())
	plotSpectrum(s,mol=mol,ilin=iline,dpc=140.)
	#plotSpectrum(s,ilin=iline,dpc=140.)
#	plotImage(s,log=True,maxlog=4,mol=mol,ilin=iline,cmap=cm.hot,bunit='snu',dpc=140,arcsec=True)
	plt.show()
	fig4.savefig("line.pdf")


if PVd:
#	fig4  = plt.figure()

	p_wl_int = []

	intens = [] 
	mol   = readMol('co')
	freq0  = mol.freq[iline-1]

	dx = 2
	x_ax = np.arange(0,300,dx)
#	dx = x_ax[1] - x_ax[0]

	for x in x_ax:
		pos = ( x-0.5*dx, x+0.5*dx , -dx , dx)
		zoomau = "zoomau %f %f %f %f"%(pos)
#		print(zoomau)
		angle = "incl %d phi %d"%(incl,phi)	
#		vreso = "widthkms 6 linenlam 72"
		vreso = "widthkms 6 linenlam 72"
		os.system("radmc3d spectrum iline %d %s %s %s setthreads 16 nofluxcons"%(iline,vreso,angle,zoomau))
		s=readSpectrum()
#		print(s)
		freq = 1e4*cc/s[:,0] 
		vkms = 2.9979245800000e+10*(freq0-freq)/freq0/1.e5 
		s[:,0] = vkms
		
		p_wl_int.append(s)

		intens.append( s[:,1] )

	print(x_ax,vkms,intens)
#	plotSpectrum(s,mol=mol,ilin=iline,dpc=140.)
#	plt.show()
#	fig4.savefig("line.pdf")
        fig6  = plt.figure()
	c  = plb.contourf( x_ax ,vkms , np.array(intens).T , 30)
	cb = plb.colorbar(c)
	#plt.show()	
        fig6.savefig("PVd.pdf")




#
# Make and plot an example image
#
#
# Make and plot the SED as seen at 1 pc distance

if SED:
	os.system("radmc3d sed incl %d phi %d setthreads 4"%(incl,phi))
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

