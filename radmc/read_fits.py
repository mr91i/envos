import numpy as np
import astropy.io.fits as iofits
import matplotlib.pyplot as plt
from scipy import integrate
import sys
import cst
from scipy import interpolate
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
plt.switch_backend('agg')

def GridContour(xx, yy ,  z , n_lv=20 , cbmin=None, cbmax=None):
	cmap = plt.get_cmap('viridis')
	if cbmin == None:
		cbmin = z.min()
	if cbmax == None:
		cbmax = z.max() 
	levels = MaxNLocator(nbins=n_lv).tick_values(cbmin, cbmax)
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
	return plt.pcolormesh(xx, yy, z, cmap=cmap, norm=norm, rasterized=True)

chmap = 0
pvd = 1
conv = 0
beam_size = 10
d = 140 * cst.pc
figdir = "F/"


pic = iofits.open("obs.fits")[0]
header = pic.header
data = pic.data
print(vars(header), data )

Nx, Ny, Nz = header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]
dx, dy = np.array([ header["CDELT1"] , header["CDELT2"] ]) *np.pi/180.0 *d/cst.au
Lx, Ly = Nx*dx, Ny*dy 
x = Lx/2 - (np.arange(Nx)+0.5)*dx
y = Ly/2 - (np.arange(Ny)+0.5)*dy
freq_max , dfreq = header["CRVAL3"] , header["CDELT3"]
freq0 = freq_max + dfreq*(Nz-1)/2

if conv:
	from scipy.ndimage import gaussian_filter 
	data = np.array([ gaussian_filter(d, sigma=np.abs( beam_size/dx ) ) for d in data ])

if chmap:
	xx, yy = np.meshgrid(x,y)
	cbmax = 0.1  # data.max()*1e4
	for i in range(len(data)):
		n_lv = 60
		vkms = cst.c/1e5* dfreq*(0.5*(Nz-1)-i)/freq0
#		im = plt.contourf(xx, yy, data[i]*1e4, cmap='viridis',levels=20)
		im = GridContour(xx, yy, data[i]*1e4 , n_lv=n_lv ,cbmin=0, cbmax=cbmax)
		plt.colorbar(label = r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
		plt.xlabel("Position [au]")
		plt.ylabel("Position [au]")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.title(f"v = {vkms:.3f} km/s")
		plt.savefig(figdir+"chmap_{0=4d}.pdf"%i, bbox_inches="tight" , dpi = 300)
		plt.clf()

if pvd:
	mark_peak = 0	
	n_lv = 20
	vkms = cst.c/1e5* dfreq*(0.5*(Nz-1) - np.arange(len(data)) )/freq0
	print( vkms[1] - vkms[0] )
	exit()

	xx, vv = np.meshgrid(x,vkms)

	from scipy import signal
	I_pv = integrate.simps(  data[ : , int(Ny/2-0.5*beam_size/dy) : int(Ny/2+0.5*beam_size/dy) , : ] , axis=1) 
	maxId = np.array( [ signal.argrelmax( I_pv[ i , : ] ) for i in range(len( I_pv[ : , 0 ])) ] )

	im = GridContour(xx, vv, I_pv*1e4 , n_lv=n_lv )
	plt.plot( x , np.sqrt(cst.G*0.77*cst.Msun/x/cst.au)/cst.kms  )

	if mark_peak:
		maxxv = []
		for i in range(len( I_pv[ : , 0 ] )):
			maxi = signal.argrelmax( I_pv[ i , : ] ) 
			for ii in maxi[0]:
				maxxv.append( [ x[ ii ] , vkms[i] ] )
		maxxv = np.array(maxxv)
		plt.scatter( maxxv[:,0] , maxxv[:,1] , c="red", s=1)

	plt.xlabel("Position [au]")
	plt.ylabel(r"Velocity [km s$^{-1}$]")
	plt.ylim( vkms[0],vkms[-1] )
	cbar=plt.colorbar(im)
	cbar.set_label(r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
	plt.savefig(figdir+"pvd.pdf" , bbox_inches="tight", dpi = 300)
