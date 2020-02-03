#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function,  absolute_import, division
import os
import sys
import numpy as np
import astropy.io.fits as iofits
import matplotlib.pyplot as plt
from scipy import integrate, optimize, interpolate
from skimage.feature import peak_local_max
import matplotlib        as mpl
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
plt.switch_backend('agg')
plt.rc('savefig', dpi=200, facecolor="None", edgecolor='none', transparent=True)
mpl.rc('xtick', direction="in", bottom=True, top=True)
mpl.rc('ytick', direction="in", left=True, right=True)
#sys.path.append(dn_home)
#print("Execute %s:\n"%__file__)
import cst
from myplot import Plotter
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
#       self.beam_size_average = 


class FitsAnalyzer:
    def __init__(self, fits_file_path, fig_dir_path):
#       self.fd = fits_data
        self.figd = fig_dir_path
        self.dpc = 137
    
        pic = iofits.open(fits_file_path)[0]
        self.Ippv = pic.data
        header = pic.header
        print(vars(header))
        self.Nx = header["NAXIS1"]
        self.Ny = header["NAXIS2"]
        self.Nz = header["NAXIS3"]
        self.dx = header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
        self.dy = header["CDELT2"]*np.pi/180.0*self.dpc*cst.pc/cst.au
        Lx = self.Nx*self.dx
        Ly = self.Ny*self.dy
        self.xau = 0.5*Lx - (np.arange(self.Nx)+0.5)*self.dx
        self.yau = 0.5*Ly - (np.arange(self.Ny)+0.5)*self.dy

        if header["CRVAL3"] > 1e8:
            nu_max = header["CRVAL3"]
            dnu = header["CDELT3"]
            nu0 = nu_max + 0.5*dnu*(self.Nz-1)
            self.vkms = cst.c / 1e5 * dnu/nu0 * (0.5*(self.Nz-1)-np.arange(self.Nz))
            self.dv = self.vkms[1] - self.vkms[0]
        else:
            self.dv = header["CDELT3"]/1e3
            self.vkms = self.dv * ( 0.5 * (self.Nz-1) - np.arange(self.Nz) )
        print(self.dy, self.yau, Lx ,Ly, self.dx)
        #exit()

    
        if self.dx < 0:
            self.dx *= -1
            self.Ippv = np.flip(self.Ippv, axis=2)
        if self.dy < 0:
            self.dy *= -1
            self.Ippv = np.flip(self.Ippv, axis=1)
        if self.dv < 0:
            self.dv *= -1
            self.Ippv = np.flip(self.Ippv, axis=0)  

        print("fits file path: {}".format(fits_file_path))
        print("pixel size[au]: {} {}".format(self.dx, self.dy))
        print("L[au]: {} {}".format(Lx, Ly))

    
    def chmap(self, n_lv=20):
        xx, yy = np.meshgrid(self.xau, self.yau)
        cbmax = self.Ippv.max()
        for i in range(self.Nz):
            im = _contour_plot(xx, yy, self.Ippv[i], n_lv=n_lv, cbmin=0, cbmax=cbmax, mode='grid')
            plt.colorbar(label=r'Intensity [Jy pixel$^{-1}$ ]')
            plt.xlabel("Position [au]")
            plt.ylabel("Position [au]")
            plt.title("v = {:.3f} km/s".format(self.vkms[i]))
            plt.gca().set_aspect('equal', adjustable='box')
            self._save_fig("chmap_{:0=4d}.pdf".format(i))
            plt.clf()

#   @profile    
    def mom0map(self, n_lv=40):
        self._convolution(beam_a_au=0.4*self.dpc, beam_b_au=0.9*self.dpc, v_width_kms=0.5, theta_deg=-41)
        xx, yy = np.meshgrid(self.xau, self.yau)
#       print(self.Ippv.shape)
        Ipp = integrate.simps(self.Ippv, axis=0)
#       print(Ipp.shape)
        Plt = Plotter(self.figd)
        Plt.map(self.xau, self.yau, Ipp, "mom0map",
              xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]', 
              div=n_lv,
              square=True)  

        Plt.save("mom0map")

#       im = _contour_plot(xx, yy, Ipp, n_lv=n_lv, cbmin=0, 
#                          cbmax=Ipp.max(), mode='grid')
#       self._save_fig("mom0map")
#       plt.clf()

    def pvdiagram(self, n_lv=5, op_LocalPeak_V=0, op_LocalPeak_P=0, 
                  op_LocalPeak_2D=1, op_Kepler=0, op_Maximums_2D=1, M=0.18, posang=5):

        #self._convolution(beam_a_au=0.4*self.dpc, beam_b_au=0.9*self.dpc, v_width_kms=0.5, theta_deg=-41)  
        self._perpix_to_perbeamkms(beam_a_au=0.4*self.dpc, beam_b_au=0.9*self.dpc, v_width_kms=0.5)




        xx, vv = np.meshgrid(self.xau, self.vkms)
    
        if len(self.yau) > 1:
            points = np.stack( np.meshgrid(self.xau, self.yau, self.vkms), axis=3).reshape(-1,3)
            p_new = np.array([ self.vkms, self.xau * np.cos(posang/180.*np.pi) , self.xau * np.sin(posang/180.*np.pi) ])

            
#           print(p_new, p_new.reshape(-1,2)  )
#           interp_func = np.vectorize( interpolate.interp2d(self.xau, self.yau, self.Ippv[0] )    )
#           print(interp_func(p_new[0], p_new[1]).shape )
#           exit()
            #print(self.yau)
#           f = interpolate.RectBivariateSpline(self.xau, self.yau, self.Ippv[0])
            print( self.vkms,  self.xau, self.yau )
            print(points.shape, self.Ippv.shape, p_new.reshape(-1,3).shape ) 
#           print( interpolate.griddata( points, self.Ippv.flatten() , p_new.reshape(-1,3) ) )
            f = interpolate.RegularGridInterpolator(( self.vkms,  self.xau, self.yau ), self.Ippv )
            
            print(  f( p_new.reshape(-1,3)) )

            exit()
#           print(interpolate.interp2d(self.xau, self.yau, self.Ippv[0] )( p_new.reshape(-1,2)[0]  ) )
#           Ipv = np.array([ interp_func(p_new[0], p_new[1]) for Ipp in self.Ippv])
            #interp = np.vectorize( interpolate.interp2d  )
            #Ipv = [ interp(p_new[0], p_new[1])(    ) for x_new , y_new in p_new ]          
    
        #   Ipv = interpolate.griddata( points, self.Ippv.flatten() , )


            #interp_func = interpolate.interp1d(self.yau, self.Ippv, axis=1, kind='cubic')
            #int_ax = np.linspace(-self.beam_size*0.5, self.beam_size*0.5, 11)
            #Ixv = integrate.simps(interp_func(int_ax), x=int_ax, axisi=1)
    #       I_pv = integrate.simps( data[:,-1:,:], x=y[  ] , axis=1 )
        else:   
            Ipv = self.Ippv[:,0,:] 
    
#       fig, ax = plt.subplots(figsize=(9,3))   
#       ax.xaxis.set_minor_locator(AutoMinorLocator())
#       ax.yaxis.set_minor_locator(AutoMinorLocator())
#       im = _contour_plot(xx, vv, Ixv , n_lv=n_lv, mode="contour", cbmin=0.0)


        pltr = Plotter(self.figd)

        print(Ipv, Ipv.shape)
        exit()
        pltr.map(self.xau, self.vkms, Ipv, "pvd",
              xl="Position [au]", yl=r"Velocity [km s$^{-1}$]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
              div=n_lv,
              square=True)
    
        if op_Kepler:
            plt.plot(-self.x, np.sqrt(cst.G*M*cst.Msun/self.xau/cst.au)/cst.kms, c="cyan", ls=":")
    
        if op_LocalPeak_V:
            for Iv, v_ in zip(Ixv.transpose(0,1), self.vkms):
                for xM in _get_peaks(self.xau, Iv):
                    plt.plot(xM, v_, c="red", markersize=1, marker='o')
    
        if op_LocalPeak_P:
            for Ix, x_ in zip(Ixv.transpose(1,0), self.x):
                for vM in _get_peaks(self.vkms, Ix):
                    plt.plot(x_, vM, c="blue", markersize=1, marker='o')
    
        if op_LocalPeak_2D:
            for jM, iM in peak_local_max(Ixv, min_distance=5):
                plt.scatter(self.xau[iM], self.vkms[jM], c="k", s=20, zorder=10)
                print("Local Max:   {:.1f}  au  , {:.1f}    km/s  ({}, {})".format(self.xau[iM], self.vkms[jM], jM, iM))
    
        pltr.save("pvd")

#       plt.xlabel("Position [au]")
#       plt.ylabel(r"Velocity [km s$^{-1}$]")
#       plt.xlim( -500, 500 )
#       plt.ylim( -2.5, 2.5 )
#       im.set_clim(0, Ixv.max())
#       cbar=fig.colorbar(im)
#       cbar.set_label(r'Intensity [Jy beam$^{-1}$ (km/s)$^{-1}$]')
#       self._save_fig("pvd")
#       plt.clf()   

    def _save_fig(self, name):
        plt.savefig(self.figd+"/"+name+".pdf", bbox_inches="tight", dpi=300)
        print("Saved : "+self.figd+"/"+name+".pdf")

    def _convolution(self, beam_a_au, beam_b_au, v_width_kms, theta_deg=0):
        sigma_over_FWHM = 2 * np.sqrt( 2 * np.log(2) )
        Kernel_xy = Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                     y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                     theta=theta_deg/180*np.pi)
        Kernel_v = Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM)  

            
        for i in range(self.Nz):
            self.Ippv[i] = convolve(self.Ippv[i], Kernel_xy)    

        for j in range(self.Ny):
            for k in range(self.Nx): 
                self.Ippv[:,j,k] = convolve(self.Ippv[:,j,k], Kernel_v)


    def _perpix_to_perbeamkms(self, beam_a_au, beam_b_au, v_width_kms): 
        beam_area = np.pi * beam_a_au / 2.0 * beam_b_au / 2.0
        self.Ippv *= ( beam_area * v_width_kms ) / np.abs(self.dx*self.dy*self.dv) 




def _get_peaks(x, y):
    maxis = []
    for mi in peak_local_max(y, min_distance=3)[:,0] :
        dydx = interpolate.InterpolatedUnivariateSpline(x[mi-2:mi+3], y[mi-2:mi+3]).derivative(1)
        maxis.append(optimize.root(dydx, x[mi]).x[0])
    return np.array(maxis)


        ## self.data_im = pd.read_pickle(dn_home+'/obs.pkl')

##    def total_intensity_model():
##        f_rho = interpolate.interp1d( self.xauc , self.data.ndens_mol[:,-1,0,0] , fill_value = (0,) , bounds_error=False ,  kind='cubic')
##        def f_totemsv(x, y):
##            return f_rho( (x**2 + y**2)**0.5 ) 
##        #x_im = data_im.x/cst.au
##        yax = np.linspace(self.x_im[0], self.x_im[-1], 10000)
##        return np.array([ integrate.simps(f_totemsv( x, yax ), yax ) for x in self.x_im ])  
##    
##    def total_intenisty_image():
##        image = data_im.image.sum(axis=2) if 1 else -integrate.simps(data_im.image, data_im.freq, axis=2)
##        return np.average(image, axis=1)

##        if totint:
##            plb.plot(data_im.x/cst.au , image, label="Total Intensity: Synthetic Obs",lw=2.5)
##            plb.plot(x_im,  emsv  * ( image[ len(image)//2 ] + image[ len(image)//2-1 ])/(emsv[ len(emsv)//2 ] + emsv[ len(emsv)//2 - 1]), label="Total Intensity: Expectation", ls='--', lw=1.5)
##            plt.legend()
##            plt.ylim([0,None])
##            fig.savefig(dn_fig+"emsv_img_plf.pdf")


if __name__=="__main__":
    dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
    dn_home = os.path.abspath(dn_here + "/../")
#   dn_radmc = dn_home + "/radmc"
#   figdir = dn_home+"/fig/"
#   fd = read_fits()    
    fa = FitsAnalyzer(fits_file_path=dn_home+"/radmc/obs3.fits", fig_dir_path=dn_home+"/fig")
#   fa = FitsAnalyzer(fits_file_path=dn_home+"/radmc/L1527.fits", fig_dir_path=dn_home+"/fig")
    fa.pvdiagram()
    fa.mom0map()
#   fa.chmap()
