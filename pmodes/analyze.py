#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function,	absolute_import, division
import numpy as np
import astropy.io.fits as iofits
import matplotlib.pyplot as plt
from scipy import integrate
import os, sys
from scipy import interpolate
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from scipy import signal
plt.switch_backend('agg')
plt.rc('savefig',dpi=200,facecolor="None",edgecolor='none',transparent=True)
from skimage.feature import peak_local_max

import matplotlib		 as mpl
#mpl.rc('xtick.minor',size=3,width=10,)
#mpl.rc('ytick.minor',size=3,width=2)
mpl.rc('xtick',direction="in", bottom=True, top=True  )
mpl.rc('ytick',direction="in", left=True  , right=True)

dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + "/../../")
sys.path.append(dn_home)
print("Execute %s:\n"%__file__)
from calc import cst

import scipy.ndimage
def Contour(xx, yy ,  z , n_lv=20 , cbmin=None, cbmax=None,mode="default",cmap = plt.get_cmap('viridis'),smooth=False):

	if smooth:
		z = scipy.ndimage.zoom(z, 4).clip(0)
		xx = scipy.ndimage.zoom(xx, 4)
		yy = scipy.ndimage.zoom(yy, 4)

	dx = xx[0,1] - xx[0,0]
	dy = yy[1,0] - yy[0,0]

	if cbmin == None:
		cbmin = z.min()
	if cbmax == None:
		cbmax = z.max()
#	levels = MaxNLocator(nbins=n_lv).tick_values(cbmin, cbmax)
	levels =np.linspace(cbmin, cbmax,n_lv+1)
#	levels = MaxNLocator(nbins=n_lv).tick_values(tick_min, tick_max)
#	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
	norm = BoundaryNorm(levels, ncolors=cmap.N)
#	norm = Normalize(vmin=0.0,vmax=1)
	print("levels is ", levels, ' norm is', np.array(norm) )
	if mode=="grid":
#		n_lv = 6	
#		std = np.std(z)
#		tick_min = 3*std
#		tick_max = 15*std		
		## Make interface coordinate
		xxi = np.hstack(( xx-dx/2. , (xx[:,-1]+dx/2.).reshape(-1,1) ))
		xxi = np.vstack(( xxi , xxi[-1] ))
		yyi = np.vstack(( yy-dy/2. , (yy[-1,:]+dy/2.).reshape(1,-1) ))
		yyi = np.hstack(( yyi , yyi[:,-1].reshape(-1,1)  ))
		return plt.pcolormesh(xxi, yyi, z , cmap=cmap, norm=norm, rasterized=True)
	if mode=="contourf":
		return plt.contourf( xx , yy , z , n_lv , cmap=cmap)
	if mode=="contour":
		jM, iM =	np.unravel_index(np.argmax(z), z.shape)
		plt.scatter( xx[jM,iM] , yy[jM,iM] , c='y', s=6, zorder=12)
		return plt.contour( xx , yy , z , cmap=cmap, levels=levels )

#		return plt.contour( xx , yy , z , n_lv , cmap=cmap, vmin=cbmin, vmax=cbmax ,levels=levels)
	if mode=="scatter":
		return plt.scatter(xx, yy, vmin=cbmin, vmax=cbmax, c=z ,s=1 , cmap=cmap)
	


chmap = 1
pvd = 1
conv = 1
# x[au] = d[pc] * theta[marc]
distance = 137 * cst.pc
beamsx = 0.4 * distance/cst.pc
beamsy = 0.9 * distance/cst.pc
beam_size = 0.516 * distance/cst.pc## np.sqrt( beamsx * beamsy ) # au
beam_area = np.pi * beamsx * beamsy / 4.0

v_width = 0.5 # kms
figdir = dn_home+"/fig/"
op_LocalPeak_P = 0
op_LocalPeak_V = 0
op_LocalPeak_2D = 1
op_Kepler = 0
op_Maximums_2D = 1
Ms = 0.18

fitsfilename = "obs_L1527_cr200"
fitsfilename ="obs"

pic = iofits.open(dn_here + "/" + fitsfilename + ".fits")[0]
header = pic.header
data = pic.data

Nx, Ny, Nz = header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]  
dx, dy = np.array([ header["CDELT1"] , header["CDELT2"] ]) *np.pi/180.0 *distance/cst.au
Lx, Ly = Nx*dx, Ny*dy 
## center of pixel
x = Lx/2. - (np.arange(Nx)+0.5)*dx
y = Ly/2. - (np.arange(Ny)+0.5)*dy
freq_max , dfreq = header["CRVAL3"] , header["CDELT3"]
freq0 = freq_max + dfreq*(Nz-1)/2.
vkms = cst.c/1e5* dfreq*(0.5*(Nz-1) - np.arange(len(data)) )/freq0
dv = vkms[1] - vkms[0]





print("fitsfilenam : %s"%fitsfilename )
print("beam_size[au]: %s"%beam_size)
print("pixel size[au]: %s"%dx,dy)
print("L[au]: {} {}".format(Lx,Ly))

if conv:
	from scipy.ndimage import gaussian_filter , gaussian_filter1d
	I_max = data.max()
#	print(data.shape, v_width/dv , np.abs( beam_size/dy ) ,   np.abs( beam_size/dx ) )
#	std = np.array([ v_width/dv , np.abs( beam_size/dy ) , np.abs( beam_size/dx )] )/2.354820 ## same FWHM
#	data = gaussian_filter(data, sigma=std )
#	data *= np.abs(dx*dy*dv) /( (np.pi/4*0.9*0.4) * v_width )

	from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel, Box1DKernel
#	try:
	Convolver_space = Gaussian2DKernel(abs(beamsx/dx)/2.35482,	y_stddev=abs(beamsy/dy)/2.35482 , theta = -41/180*np.pi)
#	except:
#		Convolver_space = Gaussian2DKernel(abs(beamsx/dx)/2.35482 )
	Convolver_veloc = Gaussian1DKernel(v_width/dv/2.35482) # Box1DKernel 
	for i in range(data.shape[0]):
		data[i] = convolve(data[i], Convolver_space )
	for j in range(data.shape[1]):
		for k in range(data.shape[2]): 
			data[:,j,k] = convolve(data[:,j,k], Convolver_veloc )
	data *= ( beam_area * v_width )/ np.abs(dx*dy*dv) 

	
def make_chmap():
	if len(y) == 1:
		print("Too small y pix!")
		return
	xx, yy = np.meshgrid(x,y)
	cbmax = data.max()*1e4
	for i in range(len(data)):
		n_lv = 60
		vkms = cst.c/1e5* dfreq*(0.5*(Nz-1)-i)/freq0
#		im = plt.contourf(xx, yy, data[i]*1e4, cmap='viridis',levels=20)
		im = Contour(xx, yy, data[i]*1e4 , n_lv=n_lv ,cbmin=0, cbmax=cbmax, mode='grid')
		plt.colorbar(label = r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
		plt.xlabel("Position [au]")
		plt.ylabel("Position [au]")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.title(figdir+"v = {:.3f} km/s".format(vkms))
		plt.savefig(figdir+"chmap_{:0=4d}.pdf".format(i), bbox_inches="tight" , dpi = 300)
		plt.clf()

def make_PVdiagram():
	n_lv = 5
	xx, vv = np.meshgrid(x,vkms)
#	xx,vv = np.log10((xx,vv))


	if len(y) > 1:
		print(y,data)
		interp_func = interpolate.interp1d( y ,  data , axis=1 , kind='cubic')
		int_ax = np.linspace( -beam_size*0.5 , beam_size*0.5 , 11	)
		I_pv = integrate.simps( interp_func(int_ax) , x=int_ax , axis=1 )
	else:	
		I_pv = data[ : , 0 , : ] 



#	maxId = np.array( [ signal.argrelmax( I_pv[ i , : ] ) for i in range(len( I_pv[ : , 0 ])) ] )
#	from matplotlib.ticker import (MultipleLocator, FormatStrFormatt
	from matplotlib.ticker import AutoMinorLocator
	fig, ax = plt.subplots(figsize=(9,3))	
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
#	ax.tick_params(which='minor',length=1,width=3,color='r')

#	im = Contour(xx, vv, I_pv , n_lv=n_lv , mode="grid")

	
	im = Contour(xx, vv, I_pv , n_lv=n_lv , mode="contour",cbmin=0.0)
#	im = Contour(xx, vv, I_pv , n_lv=n_lv , mode="contourf",cbmin=0)
#	im = Contour(xx, vv, I_pv , n_lv=n_lv , mode="contour", cmap=plt.get_cmap('Greys'),cbmin=0)


	if op_Kepler:
		plt.plot( -x , np.sqrt(cst.G*Ms*cst.Msun/x/cst.au)/cst.kms	, c="cyan", ls=":")

	from scipy.interpolate import interp1d
	from scipy import optimize
	from scipy.interpolate import InterpolatedUnivariateSpline as IUS
	def get_peaks( x , y ):
		maxis = []
		for mi in peak_local_max(y, min_distance=3)[:,0] :
			df = IUS(x[mi-2:mi+3] , y[mi-2:mi+3] ).derivative(1)
			maxis.append( optimize.root(df,x[mi]).x[0] )
		return np.array( maxis )

	if op_LocalPeak_V or op_LocalPeak_P:
		if op_LocalPeak_V:
			for I_v, a_v in zip( I_pv.transpose(0,1) , vkms ):
				for xM in get_peaks(x,I_v):
					plt.plot(  xM , a_v , c="red", markersize=1 , marker='o')

		if op_LocalPeak_P:
			for I_p, a_x in zip( I_pv.transpose(1,0) , x ):
				for vM in get_peaks(vkms,I_p):
					plt.plot(  a_x , vM , c="blue", markersize=1 , marker='o')

	if op_LocalPeak_2D:
		for jM, iM in peak_local_max(I_pv, min_distance=5):
			plt.scatter( x[iM] , vkms[jM] , c="k", s=20, zorder=10)
			print("Local Max:	{:.1f}	au	, {:.1f}	km/s  ({}, {})".format( x[iM] , vkms[jM] , jM,iM ) )
#	if op_Maximums_2D:
#		print( np.where(I_pv == np.max(I_pv))  

#	plt.xscale('log')
#	plt.yscale('log')
#	plt.grid(True, which='minor', axis='both')

#	ax.xaxis.set_minor_locator(MultipleLocator(5))
	plt.xlabel("Position [au]")
	plt.ylabel(r"Velocity [km s$^{-1}$]")
#	plt.xlim( x[0],x[-1] )
	plt.xlim( -500, 500 )
#	plt.ylim( vkms[0],vkms[-1] )
	plt.ylim( -2.5, 2.5 )
	#im.set_xaxis(True, which='minor')
#	plt.tick_params(which ="minor",direction = 'in' ,axis="both"
#				, length = 10 ,width = 1)
#	plt.xlim( -250,250 )
#	plt.ylim( -3,3 )
	print(I_pv.max())
	im.set_clim(0, I_pv.max())
	cbar=fig.colorbar(im)
	cbar.set_label(r'Intensity [ Jy beam$^{-1}$ (km/s)$^{-1}$ ]')
	plt.savefig(figdir+"pvd.pdf" , bbox_inches="tight", dpi = 300)
	print("Saved : "+figdir+"pvd.pdf")




#if chmap:
#	make_chmap()

if pvd:
	make_PVdiagram()
