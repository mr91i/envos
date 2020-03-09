#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import radmc3dPy.analyze as rmca
import radmc3dPy.image as rmci
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool

import copy
import astropy.io.fits as iofits
from scipy import integrate, optimize, interpolate
from skimage.feature import peak_local_max
import astropy.convolution as aconv
#import convolve, convolve_fft, Gaussian1DKernel, Gaussian2DKernel
from header import inp, dn_home, dn_radmc, dn_fig
import myplot as mp
import cst
import mytools

msg = mytools.Message(__file__)
#####################################

def main():
    fa = FitsAnalyzer(dn_radmc=dn_radmc, dn_fig=dn_fig, **vars(inp.fitsa))
    fa.pvdiagram()
    fa.mom0map()
#    fa.chmap()

class FitsAnalyzer:
    def __init__(self, dn_radmc=None, dn_fig=None, filename=None, dpc=0, convolution_pvdiagram=True,
                 convolution_mom0map=True, posang_PV=0,
                 beam_posang=-41, oplot_KeplerRotation=0,
                 oplot_LocalPeak_Vax=0, oplot_LocalPeak_Pax=0,
                 oplot_LocalPeak_2D=1, oplot_Maximums_2D=1,
                 Mstar=0.18, beama_au=10, beamb_au=10,
                 vwidth_kms=0.5, plotmode_PV='grid',
                 convolve_PV_p=True, convolve_PV_v=True,
                 convolver='normal', pointsource_test=False,
                 logcolor_PV=False, normalize=None, oplot_ire_fit=False, Imax=None,
                 ):

        mytools.set_arguments(self, locals(), printer=msg)

        self.fits_file_path = dn_radmc + '/' + filename + '.fits'
        pic = iofits.open(self.fits_file_path)[0]
        self.Ippv_raw = pic.data
        header = pic.header

        self.datatype="pv"

        if self.datatype=="ppv":
            self.Nx = header["NAXIS1"]
            self.Ny = header["NAXIS2"]
            self.Nz = header["NAXIS3"]
            self.dx = - header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
            self.dy = + header["CDELT2"]*np.pi/180.0*self.dpc*cst.pc/cst.au
            Lx = self.Nx*self.dx
            Ly = self.Ny*self.dy
            self.xau = -0.5*Lx + (np.arange(self.Nx)+0.5)*self.dx
            self.yau = - 0.5*Ly + (np.arange(self.Ny)+0.5)*self.dy

            if header["CRVAL3"] > 1e8:
                nu_max = header["CRVAL3"] # freq: max --> min
                dnu = header["CDELT3"]
                nu0 = nu_max + 0.5*dnu*(self.Nz-1)
                self.dv = - cst.c / 1e5 * dnu / nu0
                self.vkms = (-0.5*(self.Nz-1)+np.arange(self.Nz)) * self.dv
            else:
                self.dv = header["CDELT3"]/1e3
                self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))

            if ((self.dx < 0) or (self.xau[1] < self.xau[0])) \
                or ((self.dy < 0) or (self.yau[1] < self.yau[0])) \
                or ((self.dv < 0) or (self.vkms[1] < self.vkms[0])):
                raise Exception("reading axis is wrong.")
            print("fits file path: {}".format(self.fits_file_path))
            print("pixel size[au]: {} {}".format(self.dx, self.dy))
            print("L[au]: {} {}".format(Lx, Ly))

        elif self.datatype=="pv":

            self.Nx = header["NAXIS1"]
            self.Nz = header["NAXIS2"]
            self.dx = header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
            Lx = self.Nx*self.dx
            self.xau = -0.5*Lx + (np.arange(self.Nx)+0.5)*self.dx

            if header["CRVAL2"] > 1e8:
                nu_max = header["CRVAL2"] # freq: max --> min
                dnu = header["CDELT2"]
                nu0 = nu_max + 0.5*dnu*(self.Nz-1)
                self.dv = - cst.c / 1e5 * dnu / nu0
                self.vkms = (-0.5*(self.Nz-1)+np.arange(self.Nz)) * self.dv
            else:
                self.dv = header["CDELT2"]/1e3 # in m/s to in km/s
                self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))
            self.Ippv_raw = self.Ippv_raw[::-1,:]
            print(self.xau, self.dx, self.vkms, self.dv)

            if ((self.dx < 0) or (self.xau[1] < self.xau[0])) \
                or ((self.dv < 0) or (self.vkms[1] < self.vkms[0])):
                raise Exception("reading axis is wrong.")


    def chmap(self, n_lv=20):
#        xx, yy = np.meshgrid(self.xau, self.yau)
        cbmax = self.Ippv.max()
        pltr = mp.Plotter(self.dn_fig, x=self.xau, y=self.yau,
                          xl="Position [au]", yl="Position [au]",
                          cbl=r'Intensity [Jy pixel$^{-1}$ ]')
        for i in range(self.Nz):
            pltr.map(self.Ippv[i], out="chmap_{:0=4d}".format(i),
                     n_lv=n_lv, cbmin=0, cbmax=cbmax, mode='grid',
                     title="v = {:.3f} km/s".format(self.vkms[i]) )

    def mom0map(self, n_lv=10):
        Ippv = copy.copy(self.Ippv_raw)
        if self.convolution_mom0map:
            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
        Ipp = integrate.simps(Ippv, axis=0)
        Plt = mp.Plotter(self.dn_fig, x=self.xau, y=self.yau)
        Plt.map(Ipp, out="mom0map",
                xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
                div=n_lv, mode='grid', cbmin=0, cbmax=Ipp.max(), square=True)

    def pvdiagram(self, n_lv=10):
        Ippv = copy.copy(self.Ippv_raw)
        unit = r'[Jy pixel$^{-1}$]'
        if self.pointsource_test:
            Ippv = self.use_pointsource(Ippv)

        if self.datatype=="ppv":

            if self.convolution_pvdiagram:
                Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)

                #Ippv = self._perpix_to_perbeam(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au, v_width_kms=self.vwidth_kms)
                #unit = r'[Jy beam$^{-1}$]'

                #Ippv = self._perpix_to_pergunit(Ippv)
                #unit = r'[Jy cm$^{-2}$ (km/s)$^{-1}$ ]'

            if len(self.yau) > 1:
                posang_PV_rad = self.posang_PV/180.*np.pi
                points = [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                           for r in self.xau ] for v in self.vkms]
                Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), Ippv, points)
            else:
                Ipv = Ippv[:, 0, :]
                    Ippv = self._perpix_to_perbeamkms(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au, v_width_kms=self.vwidth_kms)
                    unit = r'[Jy beam$^{-1}$ (km/s)$^{-1}$]'

            if len(self.yau) > 1:
                posang_PV_rad = self.posang_PV/180.*np.pi
                points = [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                           for r in self.xau ] for v in self.vkms]
                Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), Ippv, points)
            else:
                Ipv = Ippv[:, 0, :]

        elif self.datatype=="pv":
            Ipv = self.Ippv_raw


        if self.normalize == "peak":
            Ipv /= np.max(Ipv)
            unit = r'[$I_{\rm max}$]'
            cblim = [0, np.max(Ipv)]
        else:
            cblim = [0, np.max(Ipv)]  if not self.Imax else [0, self.Imax]


        # Set cblim
        if self.logcolor_PV:
            cblim[1] = cblim[0]*0.001

        xas = self.xau/self.dpc
        pltr = mp.Plotter(self.dn_fig, x=xas, y=self.vkms, xlim=[-5, 5], ylim=[-3, 3])

        print(cblim)
#        logoption = {"logcb":True, "cblim":[np.max(Ipv)*1e-2,np.max(Ipv)]} if self.logcolor_PV else {"cblim":[0, np.max(Ipv) for not self.Imax in self.Imax ]}
        pltr.map(z=Ipv, mode=self.plotmode_PV, logcb=self.logcolor_PV, cblim=cblim, #cblim=[np.max(Ipv)*1e-2,np.max(Ipv)],
                 xl="Angular Offset [arcsec]", yl=r"Velocity [km s$^{-1}$]",
                 cbl=r'Intensity '+unit,
                 lw=.7, #logx=True, logy=True, xlim=[0.1, 100], ylim=[0.1, 10],
                 div=n_lv, save=False)

        pltr.ax.minorticks_on()
        pltr.ax.tick_params("both",direction="inout" )
        pltr.ax2 = pltr.ax.twiny()
        pltr.ax2.minorticks_on()
        pltr.ax2.tick_params("both",direction="inout" )
        pltr.ax2.set_xlabel("Angular Offset [au]")
        pltr.ax2.set_xlim(np.array(pltr.ax.get_xlim())*self.dpc)

        rCR = 200*cst.au
        l = np.sqrt(cst.G*self.Mstar*cst.Msun*rCR)
        a = (2*self.xau*cst.au/rCR)

        if self.oplot_ire_fit:
            pltr.ax.plot(xas, 2*l/rCR*(2*self.xau.clip(0)*cst.au/rCR)**(-2/3)/cst.kms, c="hotpink", ls=":", lw=1)

        if self.oplot_KeplerRotation:
            pltr.ax.plot(xas, np.sqrt(cst.G*self.Mstar*cst.Msun/ \
                     (self.xau*cst.au))/cst.kms, c="cyan", ls=":", lw=1)

        if self.oplot_LocalPeak_Pax:
            for Iv, v_ in zip(Ipv.transpose(0, 1), self.vkms):
                for xM in get_localpeak_positions(xas, Iv, threshold_abs=np.max(Ipv)*1e-10):
                    pltr.ax.plot(xM, v_, c="red", markersize=1, marker='o')

        if self.oplot_LocalPeak_Vax:
            for Ip, x_ in zip(Ipv.transpose(1, 0), xas):
                for vM in get_localpeak_positions(self.vkms, Ip, threshold_abs=np.max(Ipv)*1e-10):
                    pltr.ax.plot(x_, vM, c="blue", markersize=1, marker='o')

        if self.oplot_LocalPeak_2D:
            for jM, iM in peak_local_max(Ipv, num_peaks=4, min_distance=10):#  min_distance=None):
                pltr.ax.scatter(xas[iM], self.vkms[jM], c="k", s=20, zorder=10)
                print("Local Max:   {:.1f}  au  , {:.1f}    km/s  ({}, {})".format(
                    self.xau[iM], self.vkms[jM], jM, iM))

            del jM, iM

        def draw_cross_pointer(x, y, c):
            pltr.ax.axhline(y=y, lw=2, ls=":", c=c, alpha=0.6)
            pltr.ax.axvline(x=x, lw=2, ls=":", c=c, alpha=0.6)
            pltr.ax.scatter(x, y, c=c, s=50 , alpha=1, linewidth=0, zorder=10)

        #vkms_peak, xau_peak
        vp_peaks = np.array([(self.vkms[jM], self.xau[iM]) for jM, iM
                             in peak_local_max(Ipv,  num_peaks=4, min_distance=10)])

        i = np.argmax( np.sign(vp_peaks[:,0])*vp_peaks[:,0] * vp_peaks[:,1]  )
        print(vp_peaks, i)
        vkms_peak, xau_peak = vp_peaks[i]
#                              if ((self.xau[iM]*cst.au > 0) and (self.vkms[jM] > 0))][0]

        #xau_peak = self.xau[iM]
        #vkms_peak = self.vkms[jM]
        M_CR = (abs(xau_peak) * cst.au * (vkms_peak*cst.kms)**2 )/(cst.G*cst.Msun)
        draw_cross_pointer(xau_peak/self.dpc, vkms_peak, mp.c_def[1])

        x_vmax, I_vmax = np.array([[ get_maximum_position(self.xau, Iv), np.max(Iv) ] for Iv in Ipv.transpose(0, 1)]).T
        if (np.min(I_vmax) < 0.1*cblim[1]) and (0.1*cblim[1] < np.max(I_vmax)):
            print("Use 0.1*cblim")
            v_10 = mytools.find_roots(self.vkms, I_vmax, 0.1*cblim[1])
        else:
            print("Use 0.1*max Ippv")
            v_10 = mytools.find_roots(self.vkms, I_vmax, 0.1*np.max(Ippv) )

        print( I_vmax, )
        x_10 = mytools.find_roots(x_vmax, self.vkms, v_10[0])
        M_CB = (abs(x_10[0]) * cst.au * (v_10[0]*cst.kms)**2 )/(2*cst.G*cst.Msun)
        draw_cross_pointer(x_10[0]/self.dpc, v_10[0], mp.c_def[0])

        plt.text(0.95, 0.05,r"$M_{\rm CR}$=%.3f"%M_CR+"\n"+r"$M_{\rm CB,10\%%}$=%.3f"%M_CB,
                 transform=pltr.ax.transAxes, ha="right", va="bottom", bbox=dict(fc="white", ec="black", pad=5))

        import matplotlib.patches as pat
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox

        beam_crosslength_asec = ( np.cos(self.beam_posang/180*np.pi)/self.beama_au**2 + np.sin(self.beam_posang/180*np.pi)/self.beamb_au**2 )**(-0.5)/self.dpc

        frameon = False
        box = AnchoredAuxTransformBox(pltr.ax.transData, loc='lower left', frameon=frameon, pad=0., borderpad=0.4)
        if frameon:
            e1 = pat.Ellipse(xy=(0,0), width=beam_crosslength_asec, height=self.vwidth_kms,
                         lw=1, fill=True, ec="k", fc="0.6")
        else:
            e1 = pat.Ellipse(xy=(0,0), width=beam_crosslength_asec, height=self.vwidth_kms,
                         lw=0, fill=True, ec="k", fc="0.7", alpha=0.6)
        box.drawing_area.add_artist(e1)
        box.patch.set_linewidth(1)
        pltr.ax.add_artist(box)



        pltr.save("pvd")

    # theta : cclw is positive
    def _convolution(self, Ippv, beam_a_au, beam_b_au, v_width_kms, theta_deg=0, ver="new"):
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        Ippv_conv = copy.copy(Ippv)
        convolver = {"normal": aconv.convolve, "fft":aconv.convolve_fft}[self.convolver]
        option = {"allow_huge":True}

        if ver=="new":# super fast
            Kernel_2d = aconv.Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                               y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                               theta=theta_deg/180*np.pi)._array
            Kernel_1d = aconv.Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM)._array
            Kernel_3d = np.multiply(Kernel_2d[np.newaxis,:,:], Kernel_1d[:,np.newaxis, np.newaxis])
            #Kernel_3d /= np.sum(Kernel_3d)
            Ippv_conv = convolver(Ippv_conv, Kernel_3d, **option)
            #return  Ippv_conv.clip(np.max(Ippv_conv)*1e-3)
            return np.where( Ippv_conv > np.max(Ippv_conv)*1e-6, Ippv_conv, -1)

        if ver=="old":
            if self.convolve_PV_p:
                Kernel_xy = aconv.Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                          y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                         #x_size= len(self.xau) + 1,#int(abs(beam_a_au/self.dx)/sigma_over_FWHM)*12+1,
            #                             y_size=4*len(self.yau) + 1,#int(abs(beam_b_au/self.dy)/sigma_over_FWHM)*12+1,
                                         theta=theta_deg/180*np.pi)
                for i in range(self.Nz):
                    Ippv_conv[i] = convolver(Ippv_conv[i], Kernel_xy, **option)

            if self.convolve_PV_v:
                Kernel_v = aconv.Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM,)
                             #           x_size=4*len(self.vkms) + 1) #int(v_width_kms/self.dv/sigma_over_FWHM)*12+1)
                for j in range(self.Ny):
                    for k in range(self.Nx):
                        Ippv_conv[:, j, k] = convolver(Ippv_conv[:, j, k], Kernel_v, **option)

            return np.where( Ippv_conv > np.max(Ippv_conv)*1e-8, Ippv_conv, 0)

    def _perpix_to_perbeam(self, intensity_ppix, beam_a_au, beam_b_au, v_width_kms):
        beam_area = np.pi * beam_a_au / 2.0 * beam_b_au / 2.0
        pixel_in_beam = beam_area/(self.dx*self.dy)
        return intensity_ppix * pixel_in_beam

    def _perpix_to_pergunit(self, intensity):
        return intensity*1./(self.dx*self.dy*self.dv)
#(self.dx*self.dy*self.dv)*cst.au**2*cst.kms
#(cst.au**2 * cst.km/s)

#        return intensity*(self.dx*self.dy*self.dv)

    def use_pointsource(self, Ippv):
        Ippv = np.zeros_like(Ippv)
        Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1
        return Ippv

def find_local_peak_position(x, y, i):
    if (2 <= i <= len(x)-3):
        grad_y = interpolate.InterpolatedUnivariateSpline(
                x[i-2:i+3], y[i-2:i+3]).derivative(1)
        return optimize.root(grad_y, x[i]).x[0]
    else:
        return np.nan

def get_localpeak_positions(x, y, min_distance=3, threshold_abs=None):
    maxis = [find_local_peak_position(x, y, mi)
             for mi
             in peak_local_max(y, min_distance=min_distance, threshold_abs=threshold_abs)[:, 0]]
    return np.array(maxis)

def get_maximum_position(x, y):
    return find_local_peak_position(x, y, np.argmax(y))


        # self.data_im = pd.read_pickle(dn_home+'/obs.pkl')

# def total_intensity_model():
# f_rho = interpolate.interp1d( self.xauc , self.data.ndens_mol[:,-1,0,0] , fill_value = (0,) , bounds_error=False ,  kind='cubic')
# def f_totemsv(x, y):
# return f_rho( (x**2 + y**2)**0.5 )
# x_im = data_im.x/cst.au
# yax = np.linspace(self.x_im[0], self.x_im[-1], 10000)
# return np.array([ integrate.simps(f_totemsv( x, yax ), yax ) for x in self.x_im ])
##
# def total_intenisty_image():
# image = data_im.image.sum(axis=2) if 1 else -integrate.simps(data_im.image, data_im.freq, axis=2)
# return np.average(image, axis=1)

# if totint:
# plb.plot(data_im.x/cst.au , image, label="Total Intensity: Synthetic Obs",lw=2.5)
# plb.plot(x_im,  emsv  * ( image[ len(image)//2 ] + image[ len(image)//2-1 ])/(emsv[ len(emsv)//2 ] + emsv[ len(emsv)//2 - 1]), label="Total Intensity: Expectation", ls='--', lw=1.5)
# plt.legend()
# plt.ylim([0,None])
# fig.savefig(dn_fig+"emsv_img_plf.pdf")


# def calc_tau_surface():
# common = "incl %d phi %d posang %d setthreads %d "%(incl,phi,posang,n_thread)
# wl = "iline %d "%iline  #   "lambda %f "%wl
# cmd = "radmc3d tausurf 1 npix 100 sizeau 500 " + common + wl
# subprocess.call(cmd,shell=True)
# a=readImage()
# fig = plt.figure()
# c   = plb.contourf( a.x/cst.au , a.y/cst.au , a.image[:,:,0].T.clip(0)/cst.au, levels=np.linspace(0.0, 30, 20+1) )
# cb = plb.colorbar(c)
# plt.savefig(dn_fig+"tausurf.pdf")

if __name__ == '__main__':
    main()


