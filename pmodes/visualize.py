#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function,	absolute_import, division
import os
import sys
import numpy as np
import astropy.io.fits as iofits
import matplotlib.pyplot as plt
from scipy import integrate, optimize, interpolate
from skimage.feature import peak_local_max
import matplotlib		 as mpl
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
plt.switch_backend('agg')
plt.rc('savefig', dpi=200, facecolor="None", edgecolor='none', transparent=True)
mpl.rc('xtick', direction="in", bottom=True, top=True)
mpl.rc('ytick', direction="in", left=True, right=True)

dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + "/../")
dn_radmc = dn_home + "/radmc"
figdir = dn_home+"/fig/"
sys.path.append(dn_home)
print("Execute %s:\n"%__file__)

import cst
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel, Box1DKernel

class Fits:
	def __init__(self):
		self.xau = None
		self.yau = None
		self.vkms = None
		self.Ipv = None
		self.dx = None
		self.dy = None
		self.dnu = None
		self.dv = None
		self.dpc = 137
		self.nu_max = None
#		self.beam_size_average = 


def read_fits(fitsfilename="obs"):
	fd = Fits()
	pic = iofits.open("{}/{}.fits".format(dn_radmc, fitsfilename))[0]
	fd.Ipv = pic.data
	header = pic.header
	Nx = header["NAXIS1"]
	Ny = header["NAXIS2"]
	Nz = header["NAXIS3"]
	fd.dx = header["CDELT1"]*np.pi/180.0*fd.dpc*cst.pc/cst.au
	fd.dy = header["CDELT2"]*np.pi/180.0*fd.dpc*cst.pc/cst.au
	Lx = Nx*fd.dx
	Ly = Ny*fd.dy
	fd.xau = 0.5*Lx - (np.arange(Nx)+0.5)*fd.dx
	fd.yau = 0.5*Ly - (np.arange(Ny)+0.5)*fd.dy
	fd.nu_max = header["CRVAL3"]
	fd.dnu = header["CDELT3"]
	nu0 = fd.nu_max + 0.5*fd.dnu*(Nz-1)
	fd.vkms = cst.c / 1e5 * fd.dnu * (0.5*(Nz-1)-np.arange(len(fd.Ipv))) / nu0
	fd.dv = fd.vkms[1] - fd.vkms[0]
	print("fitsfilenam: {}".format(fitsfilename))
	print("pixel size[au]: {} {}".format(fd.dx, fd.dy))
	print("L[au]: {} {}".format(Lx, Ly))
	return fd

def _convolution(fd, v_width_kms=0.5, beam_x_au=50, beam_y_au=50):
	Ipv_max = fd.Ipv.max()
	Kernel_xy = Gaussian2DKernel(abs(beam_x_au/fd.dx)/2.35482, y_stddev=abs(beam_y_au/fd.dy)/2.35482, theta=-41/180*np.pi)
	Kernel_v = Gaussian1DKernel(v_width_kms/fd.dv/2.35482) # Box1DKernel 

	for i in range(fd.Ipv.shape[0]):
		fd.Ipv[i] = convolve(fd.Ipv[i], Kernel_xy )

	for j in range(fd.Ipv.shape[1]):
		for k in range(fd.Ipv.shape[2]): 
			fd.Ipv[:,j,k] = convolve(fd.Ipv[:,j,k], Kernel_v )

	beam_area = np.pi * beam_x_au * beam_y_au / 4.0
	fd.Ipv *= ( beam_area * v_width_kms ) / np.abs(fd.dx*fd.dy*fd.dv) 
	return fd

def _get_peaks(x, y):
	maxis = []
	for mi in peak_local_max(y, min_distance=3)[:,0] :
		df = interpolate.InterpolatedUnivariateSpline(x[mi-2:mi+3], y[mi-2:mi+3]).derivative(1)
		maxis.append(optimize.root(df, x[mi]).x[0])
	return np.array(maxis)
	
def make_pvdiagram(fd, n_lv=5, op_LocalPeak_V=0, op_LocalPeak_P=0, 
				   op_LocalPeak_2D=1, op_Kepler=0, op_Maximums_2D=1, M=0.18):

	fd = _convolution(fd, beam_x_au=0.4*fd.dpc, beam_y_au=0.9*fd.dpc)
	xx, vv = np.meshgrid(fd.xau, fd.vkms)

	if len(fd.yau) > 1:
		interp_func = interpolate.interp1d(fd.yau, fd.Ipv, axis=1, kind='cubic')
		int_ax = np.linspace(-fd.beam_size*0.5, fd.beam_size*0.5, 11)
		Ixv = integrate.simps(interp_func(int_ax), x=int_ax, axis=1)
#		I_pv = integrate.simps( data[:,-1:,:], x=y[  ] , axis=1 )
	else:	
		Ixv = fd.Ipv[:,0,:] 

	fig, ax = plt.subplots(figsize=(9,3))	
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	im = _contour_plot(xx, vv, Ixv , n_lv=n_lv, mode="contour", cbmin=0.0)

	if op_Kepler:
		plt.plot(-x, np.sqrt(cst.G*M*cst.Msun/x/cst.au)/cst.kms, c="cyan", ls=":")

	if op_LocalPeak_V:
		for Iv, v_ in zip(Ixv.transpose(0,1), vkms):
			for xM in fd.get_peaks(x, Iv):
				plt.plot(xM, v_, c="red", markersize=1, marker='o')

	if op_LocalPeak_P:
		for Ix, x_ in zip(Ixv.transpose(1,0), x):
			for vM in fd.get_peaks(vkms, Ix):
				plt.plot(x_, vM, c="blue", markersize=1, marker='o')

	if op_LocalPeak_2D:
		for jM, iM in peak_local_max(Ixv, min_distance=5):
			plt.scatter( fd.xau[iM] , fd.vkms[jM] , c="k", s=20, zorder=10)
			print("Local Max:	{:.1f}	au	, {:.1f}	km/s  ({}, {})".format(fd.xau[iM], fd.vkms[jM], jM, iM))

	plt.xlabel("Position [au]")
	plt.ylabel(r"Velocity [km s$^{-1}$]")
	plt.xlim( -500, 500 )
	plt.ylim( -2.5, 2.5 )
	im.set_clim(0, Ixv.max())
	cbar=fig.colorbar(im)
	cbar.set_label(r'Intensity [Jy beam$^{-1}$ (km/s)$^{-1}$]')
	plt.savefig(figdir+"pvd.pdf", bbox_inches="tight", dpi=300)
	print("Saved : "+figdir+"pvd.pdf")


def _contour_plot(xx, yy, z, n_lv=20, cbmin=None, cbmax=None, 
				 mode="default", cmap=plt.get_cmap('viridis'), 
				 smooth=False):

	dx = xx[0,1] - xx[0,0]
	if len(yy) > 1:
		dy = yy[1,0] - yy[0,0]
	else:
		dy = dx*10

	if cbmin == None:
		cbmin = z.min()

	if cbmax == None:
		cbmax = z.max()

	levels = np.linspace(cbmin, cbmax, n_lv+1)
	norm = BoundaryNorm(levels, ncolors=cmap.N)
#	print("levels is ", levels, ' norm is', vars(norm) )

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
		jM, iM = np.unravel_index(np.argmax(z), z.shape)
		plt.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
		return plt.contour(xx, yy, z, cmap=cmap, levels=levels)

#		return plt.contour( xx , yy , z , n_lv , cmap=cmap, vmin=cbmin, vmax=cbmax ,levels=levels)
	if mode=="scatter":
		return plt.scatter(xx, yy, vmin=cbmin, vmax=cbmax, c=z, s=1, cmap=cmap)

def make_chmap(fd):
#	if len(fd.yau) == 1:
#		print("Too small y pix!")
#		return

	xx, yy = np.meshgrid(fd.xau, fd.yau)
	print(yy)
	cbmax = fd.Ipv.max()*1e4
	for i in range(len(fd.Ipv)):
		n_lv = 60
#		vkms_ = fd.vkms[i]
#		vkms = cst.c / 1e5 * fd.dnu * (0.5*(fd.Nz-1)-i) / fd.nu0
#		im = plt.contourf(xx, yy, data[i]*1e4, cmap='viridis',levels=20)
		im = _contour_plot(xx, yy, fd.Ipv[i]*1e4, n_lv=n_lv, cbmin=0, cbmax=cbmax, mode='grid')
		plt.colorbar(label=r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
		plt.xlabel("Position [au]")
		plt.ylabel("Position [au]")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.title("v = {:.3f} km/s".format(fd.vkms[i]))
		plt.savefig(figdir+"chmap_{:0=4d}.pdf".format(i), bbox_inches="tight", dpi = 300)
		plt.clf()

def make_mom0map(fd):
	xx, yy = np.meshgrid(fd.xau, fd.yau)
	cbmax = fd.Ipv.max()*1e4
	n_lv = 60
	Ip = integrate.simps(fd.Ipv,x=fd.vkms,axis=0)
	im = _contour_plot(xx, yy, Ip*1e4, n_lv=n_lv, cbmin=0, cbmax=cbmax, mode='grid')
	plt.colorbar(label=r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
	plt.xlabel("Position [au]")
	plt.ylabel("Position [au]")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.savefig(figdir+"mom0map.pdf", bbox_inches="tight", dpi = 300)
	plt.clf()

if __name__=="__main__":
	fd = read_fits()	
	make_pvdiagram(fd)
	make_mom0map(fd)
	make_chmap(fd)
