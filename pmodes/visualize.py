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
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel, Gaussian2DKernel
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
                 logcolor_PV=False,
                 ):

        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
                msg(k.ljust(20)+"is {:20}".format(v))

        self.fits_file_path = dn_radmc + '/' + filename + '.fits'
        pic = iofits.open(self.fits_file_path)[0]
        self.Ippv_raw = pic.data
        header = pic.header
        for k, v in header.__dict__.items():
            print(k, " = ", v)

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

    def mom0map(self, n_lv=20):
        Ippv = copy.copy(self.Ippv_raw)
        if self.convolution_mom0map:
            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
        Ipp = integrate.simps(Ippv, axis=0)
        Plt = mp.Plotter(self.dn_fig, x=self.xau, y=self.yau)
        Plt.map(Ipp, out="mom0map",
                xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
                div=n_lv, mode='grid', cbmin=0, cbmax=Ipp.max())

    def pvdiagram(self, n_lv=10):
        Ippv = copy.copy(self.Ippv_raw)
        if self.pointsource_test:
            self.self._pointsource(Ippv)

        if self.convolution_pvdiagram:
            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
            Ippv = self._perpix_to_perbeamkms(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au, v_width_kms=self.vwidth_kms)

        if len(self.yau) > 1:
            posang_PV_rad = self.posang_PV/180.*np.pi
            points = [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                       for r in self.xau ] for v in self.vkms]
            Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), Ippv, points)
        else:
            Ipv = Ippv[:, 0, :]

        xas = self.xau/self.dpc
        pltr = mp.Plotter(self.dn_fig, x=xas, y=self.vkms, xlim=[-5, 5], ylim=[-3, 3])

        logoption = {"logcb":True, "cblim":[np.max(Ipv)*1e-2,np.max(Ipv)]} if self.logcolor_PV else {}
        pltr.map(z=Ipv, mode=self.plotmode_PV, **logoption, #logcb=True, cblim=[np.max(Ipv)*1e-2,np.max(Ipv)],
                 xl="Angular Offset [arcsec]", yl=r"Velocity [km s$^{-1}$]",
                 cbl=r'Intensity [Jy pixel$^{-1}$ ]',
                 lw=2, #logx=True, logy=True, xlim=[0.1, 100], ylim=[0.1, 10],
                 div=n_lv, save=False)

        rCR = 200*cst.au
        l = np.sqrt(cst.G*self.Mstar*cst.Msun*rCR)
        a = (2*self.xau*cst.au/rCR)

        #plt.plot(xas, l/(self.xau*cst.au)/cst.kms, c="green", ls=":", lw=1)
        #plt.plot(xas, l/(self.xau*cst.au)/cst.kms*(a)**(0.5), c="navy", ls=":", lw=1)
        #plt.plot(xas, l/(self.xau*cst.au)/cst.kms*(a), c="plum", ls=":", lw=1)
        #plt.plot(xas, -l/(self.xau*cst.au)/cst.kms, c="green", ls=":", lw=1)
        #plt.plot(xas, -l/(self.xau*cst.au)/cst.kms*(a)**(0.5), c="navy", ls=":", lw=1)
        #plt.plot(xas, -l/(self.xau*cst.au)/cst.kms*(a), c="plum", ls=":", lw=1)
        plt.plot(xas, 2*l/rCR*(2*self.xau*cst.au/rCR)**(-2/3)/cst.kms, c="hotpink", ls=":", lw=1)
        #plt.plot(xas, np.sqrt(2*cst.G*self.Mstar*cst.Msun/(self.xau*cst.au))*np.sqrt(2)/3**(3/4)/cst.kms, c="hotpink", ls=":", lw=1)
        
        if self.oplot_KeplerRotation:
            plt.plot(xas, np.sqrt(cst.G*self.Mstar*cst.Msun/ \
                     (self.xau*cst.au))/cst.kms, c="cyan", ls=":", lw=1)

        if self.oplot_LocalPeak_Vax:
            for Iv, v_ in zip(Ipv.transpose(0, 1), self.vkms):
                for xM in _get_peaks(xas, Iv):
                    plt.plot(xM, v_, c="red", markersize=1, marker='o')

        if self.oplot_LocalPeak_Pax:
            for Ip, x_ in zip(Ipv.transpose(1, 0), xas):
                for vM in _get_peaks(self.vkms, Ip):
                    plt.plot(x_, vM, c="blue", markersize=1, marker='o')

        if self.oplot_LocalPeak_2D:
            for jM, iM in peak_local_max(Ipv, min_distance=10):
                plt.scatter(xas[iM], self.vkms[jM], c="k", s=20, zorder=10)
                print("Local Max:   {:.1f}  au  , {:.1f}    km/s  ({}, {})".format(
                    self.xau[iM], self.vkms[jM], jM, iM))

        pltr.save("pvd")

    # theta : cclw is positive
    def _convolution(self, Ippv, beam_a_au, beam_b_au, v_width_kms, theta_deg=0):
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        Ippv_conv = copy.copy(Ippv)
        convolver = {"normal": convolve, "fft": convolve_fft}[self.convolver]

        if self.convolve_PV_p:
            Kernel_xy = Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                         y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                         x_size=4*len(self.xau) + 1,#int(abs(beam_a_au/self.dx)/sigma_over_FWHM)*12+1,
                                         y_size=4*len(self.yau) + 1,#int(abs(beam_b_au/self.dy)/sigma_over_FWHM)*12+1,
                                         theta=theta_deg/180*np.pi)
            for i in range(self.Nz):
                Ippv_conv[i] = convolver(Ippv_conv[i], Kernel_xy)

        if self.convolve_PV_v:
            Kernel_v = Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM,)
#                                        x_size=4*len(self.vkms) + 1)#int(v_width_kms/self.dv/sigma_over_FWHM)*12+1)
            for j in range(self.Ny):
                for k in range(self.Nx):
                    Ippv_conv[:, j, k] = convolver(Ippv_conv[:, j, k], Kernel_v)
        return Ippv_conv


    def _perpix_to_perbeamkms(self, intensity, beam_a_au, beam_b_au, v_width_kms):
        beam_area = np.pi * beam_a_au / 2.0 * beam_b_au / 2.0
        return intensity*(beam_area*v_width_kms)/(self.dx*self.dy*self.dv)

    def _pointsource(self, Ippv):
        Ippv = np.zeros_like(Ippv)
        Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1

def _get_peaks(x, y):
    maxis = []
    for mi in peak_local_max(y, min_distance=3)[:, 0]:
        dydx = interpolate.InterpolatedUnivariateSpline(
            x[mi-2:mi+3], y[mi-2:mi+3]).derivative(1)
        maxis.append(optimize.root(dydx, x[mi]).x[0])
    return np.array(maxis)


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
