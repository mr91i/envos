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
plt.rc('savefig',dpi=200,facecolor='none',edgecolor='none')


def Contour(xx, yy ,  z , n_lv=20 , cbmin=None, cbmax=None,mode="default"):
	cmap = plt.get_cmap('viridis')
	dx = xx[0,1] - xx[0,0]
	dy = yy[1,0] - yy[0,0]
	if cbmin == None:
		cbmin = z.min()
	if cbmax == None:
		cbmax = z.max() 
	if mode=="grid":	
		levels = MaxNLocator(nbins=n_lv).tick_values(cbmin, cbmax)
		norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
		z_avg = np.zeros_like(xx)
		print(z_avg.shape, len(x), len(y) )
		i_max, j_max = z_avg.shape
		print(i_max, j_max	)
		for i in range(i_max):
			for j in range(j_max):
				if (i+1< i_max) and  (j+1 < j_max ):
					print(i,j)
					z_avg[i,j] = 0.25 * ( z[i,j] + z[i+1,j] + z[i,j+1] + z[i+1,j+1] )
		return plt.pcolormesh(xx, yy, z_avg, cmap=cmap, norm=norm, rasterized=True)
#		return plt.pcolormesh(xx-dx/2, yy-dy/2, z, cmap=cmap, norm=norm, rasterized=True)
	if mode=="contourf":
		return plt.contourf( xx , yy , z , n_lv )
	if mode=="contour":
		return plt.contour( xx , yy , z , n_lv)
	if mode=="scatter":
		return plt.scatter(xx, yy, vmin=cbmin, vmax=cbmax, c=z ,s=1 )
	


chmap = 0
pvd = 1
conv = 1
beam_size = 14
d = 140 * cst.pc
figdir = "F/"
mark_peak = 0
Ms = 0.18


fitsfilename = "obs_L1527_cr200"
fitsfilename ="obs"

pic = iofits.open(fitsfilename + ".fits")[0]
header = pic.header
data = pic.data
print(vars(header), data )

Nx, Ny, Nz = header["NAXIS1"], header["NAXIS2"], header["NAXIS3"]
dx, dy = np.array([ header["CDELT1"] , header["CDELT2"] ]) *np.pi/180.0 *d/cst.au
Lx, Ly = Nx*dx, Ny*dy 
x = Lx/2 - (np.arange(Nx)+0.5)*dx
y = Ly/2 - (np.arange(Ny)+0.5)*dy
#x = Lx/2 - (np.arange(Nx))*dx
#y = Ly/2 - (np.arange(Ny))*dy
#print(dx,dy)
freq_max , dfreq = header["CRVAL3"] , header["CDELT3"]
freq0 = freq_max + dfreq*(Nz-1)/2

#print(	x , y ,  freq0	 )
#exit()

if conv:
	from scipy.ndimage import gaussian_filter 
	I_max = data.max()
	data = np.array([ gaussian_filter(d, sigma=np.abs( beam_size/dx ) ) for d in data ])


if chmap:
	xx, yy = np.meshgrid(x,y)
	cbmax = data.max()*1e4
	for i in range(len(data)):
		n_lv = 60
		vkms = cst.c/1e5* dfreq*(0.5*(Nz-1)-i)/freq0
#		im = plt.contourf(xx, yy, data[i]*1e4, cmap='viridis',levels=20)
		im = GridContour(xx, yy, data[i]*1e4 , n_lv=n_lv ,cbmin=0, cbmax=cbmax)
		plt.colorbar(label = r'Intensity [10$^{-4}$ Jy pixel$^{-1}$ ]')
		plt.xlabel("Position [au]")
		plt.ylabel("Position [au]")
		plt.gca().set_aspect('equal', adjustable='box')
		plt.title(figdir+"v = {:.3f} km/s".format(vkms))
		plt.savefig(figdir+"chmap_{:0=4d}.pdf".format(i), bbox_inches="tight" , dpi = 300)
		plt.clf()

if pvd:
	n_lv = 6
	vkms = cst.c/1e5* dfreq*(0.5*(Nz-1) - np.arange(len(data)) )/freq0
	xx, vv = np.meshgrid(x,vkms)
#	xx,vv = np.log10((xx,vv))
	from scipy import signal

	print(y,data,data.shape)
	interp_func = interpolate.interp1d( y ,  data , axis=1 , kind='cubic')
	int_ax = np.linspace( -beam_size*0.5 , beam_size*0.5 , 11	)
	I_pv = integrate.simps( interp_func(int_ax) , x=int_ax , axis=1 )
#	I_pv = integrate.simps(  data[ : , int(Nx/2)-4:int(Nx/2)+4 , : ] , axis=1)
	

	maxId = np.array( [ signal.argrelmax( I_pv[ i , : ] ) for i in range(len( I_pv[ : , 0 ])) ] )
	
	im =Contour(xx, vv, I_pv , n_lv=n_lv , mode="contour")

	plt.plot( x , np.sqrt(cst.G*Ms*cst.Msun/x/cst.au)/cst.kms  , c="cyan", ls=":")

	from scipy.interpolate import interp1d
	from scipy import optimize
	from scipy.interpolate import InterpolatedUnivariateSpline
	def get_peaks( x , y ):
		maxi = signal.argrelmax( y )
		maxis = []
		for amaxi in maxi[0]:
			print(x[amaxi] )
			if (amaxi <= 1) or (amaxi >= 1+len(x)) :
				continue
			f_interp = InterpolatedUnivariateSpline(x[amaxi-2:amaxi+3] , y[amaxi-2:amaxi+3] )
			df = f_interp.derivative(1)
			maxis.append( optimize.root(df,x[amaxi]).x[0] )
			solx = optimize.root(df,x[amaxi]).x
			print(len(solx),solx )
		return	np.array( maxis	)

		try:
			f = interp1d( x , np.gradient(y,x[1]-x[0]) )
#			print( signal.argrelmax( y ) , x[signal.argrelmax( y )] )
			return	optimize.fsolve( f , x[signal.argrelmax( y )] )
		except:
			return np.array([])

	if mark_peak:
		maxxv = []
		for i in range(len( I_pv[ : , 0 ] )):
			maxx = get_peaks( x	, I_pv[ i , : ] )
			for amaxx in maxx:
				maxxv.append( [  amaxx , vkms[i]] )
		maxxv = np.array(maxxv)

		print(maxxv)
		plt.scatter( maxxv[:,0] , maxxv[:,1] , c="red", s=1)

		maxxv = []
		for j in range(len( I_pv[ 0 , : ] )):
			maxv = get_peaks( vkms , I_pv[ : , j ] )
			for amaxv in maxv:
				maxxv.append( [  x[j] , amaxv ] )
		maxxv = np.array(maxxv)
		plt.scatter( maxxv[:,0] , maxxv[:,1] , c="blue", s=1)

#	plt.xscale('log')
#	plt.yscale('log')
	plt.xlabel("Position [au]")
	plt.ylabel(r"Velocity [km s$^{-1}$]")
	plt.xlim( x[0],x[-1] )
	plt.ylim( vkms[0],vkms[-1] )
#	plt.xlim( -250,250 )
#	plt.ylim( -3,3 )
	cbar=plt.colorbar(im)
	cbar.set_label(r'Intensity [ Jy pixel$^{-1}$ ]')
	plt.savefig(figdir+"pvd.pdf" , bbox_inches="tight", dpi = 300)
