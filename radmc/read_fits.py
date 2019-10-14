import numpy as np
import astropy.io.fits as iofits
import matplotlib.pyplot as plt
from scipy import integrate
import cst
from scipy import interpolate
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable

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

list = iofits.open("obs.fits")
pic = list[0]
header = pic.header
data = pic.data
d = 140 * cst.pc

Nx, Ny, Nz = header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]

dx, dy = np.array( [ header["CDELT1"], header["CDELT2"] ]  ) *np.pi/180.0 *d/cst.au
Lx, Ly = Nx*dx, Ny*dy 
x = Lx/2 - (np.arange(Nx)+0.5)*dx
y = Ly/2 - (np.arange(Ny)+0.5)*dy

freq_max = header["CRVAL3"]
dfreq = header["CDELT3"]
freq0 = freq_max + dfreq*(Nz-1)/2

if chmap:
	xx, yy = np.meshgrid(x,y)
	cbmax = 0.7  # data.max()*1e4
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
		plt.savefig(f"chmap_{i:0=4d}.pdf", bbox_inches="tight" , dpi = 300)
		plt.clf()
if pvd:	
	n_lv = 20
	vkms = cst.c/1e5* dfreq*(0.5*(Nz-1) - np.arange(len(data)) )/freq0
	xx, vv = np.meshgrid(x,vkms)
	I_pv = integrate.simps(  data[ : , int(Nx/2)-4:int(Nx/2)+4 , : ] , axis=1) 
	print(I_pv,x, vkms)
#	im = plt.plot(xx, vv, I_pv*1e4, cmap='viridis',levels=n_lv)

	print(xx.shape, vv.shape, I_pv.shape )
	
	im = GridContour(xx, vv, I_pv*1e4 , n_lv=n_lv )

	plt.xlabel("Position [au]")
	plt.ylabel(r"Velocity [km s$^{-1}$]")
	cbar=plt.colorbar(im)
	cbar.set_label(r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
#	plt.show()
#	plt.savefig(f"pvd.pdf" , bbox_inches="tight")
	plt.savefig(f"pvd.pdf" , bbox_inches="tight", dpi = 300)


