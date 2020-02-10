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
    if inp.sobs.ovserve:
        osim = ObsSimulator(dn_radmc, dn_fits=dn_radmc, **vars(inp.sobs))
        osim.observe()
#   sotel.read_instance()

    fa = FitsAnalyzer(dn_radmc=dn_radmc, dn_fig=dn_fig, **vars(inp.fitsa))
    fa.pvdiagram()
    fa.mom0map()
#    fa.chmap()


class ObsSimulator():  # This class returns observation data
    def __init__(self, dn_radmc, dn_fits=None, filename="obs", dpc=None, iline=None,
                 sizex_au=None, sizey_au=None,
                 pixsize_au=None,
                 vwidth_kms=None, dv_kms=None,
                 incl=None, phi=None, posang=None,
                 rect_camera=True, omp=True, n_thread=1,**kwargs):

        del kwargs
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
                msg(k.ljust(20)+"is {:20}".format(v))

        self.linenlam = 2*int(round(self.vwidth_kms/self.dv_kms)) + 1
        self.set_camera_info()
        print("Total cell number is {} x {} x {} = {}".format(
              self.npixx, self.npixy,self.linenlam,
              self.npixx*self.npixy*self.linenlam))

        #if kwargs != {}:
        #    raise Exception("There is unused args :", kwargs)

    def set_camera_info(self):
        if self.rect_camera and (self.sizex_au != self.sizey_au):
            #########################################
            # Note: Before using rectangle imaging, #
            # you need to fix a bug in radmc3dpy.   #
            # Also, dx needs to be equal dy.        #
            #########################################
            if self.sizey_au == 0:
                self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
                self.zoomau_y = [-self.pixsize_au/2, self.pixsize_au/2]
                self.npixx = int(round(self.sizex_au/self.pixsize_au)) # // or int do not work to convert into int.
                self.npixy = 1
            else:
                self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
                self.zoomau_y = [-self.sizey_au/2, self.sizey_au/2]
                self.npixx = int(round(self.sizex_au/self.pixsize_au))
                self.npixy = int(round(self.sizey_au/self.pixsize_au))

        else:
            self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
            self.zoomau_y = [-self.sizex_au/2, self.sizex_au/2]
            self.npixx = int(round(self.sizex_au/self.pixsize_au))
            self.npixy = self.npixx

        dx = (self.zoomau_x[1] - self.zoomau_x[0])/self.npixx
        dy = (self.zoomau_y[1] - self.zoomau_y[0])/self.npixy
        if dx != dy:
            raise Exception("dx is not equal to dy")

    @staticmethod
    def find_proper_nthread( n_thread, n_divided):
        return max([i for i in range(n_thread, 0, -1)
                                if n_divided % i == 0])

    @staticmethod
    def divide_threads(n_thread, n_divided):
        ans = [n_divided//n_thread]*n_thread
        rem = n_divided % n_thread
        nlist = [ (n_thread-i//2 if i%2 else i//2) for i in range(n_thread)] #[ ( i if i%2==0 else n_thread -(i-1) ) for i in range(n_thread) ]
        for i in range(rem):
            ii = (n_thread-1 - i//2) if i%2 else i//2
            ans[ii] += 1
        return ans

    @staticmethod
    def calc_thread_seps(calc_points, thread_divs):
        ans = []
        sum_points = 0
        for npoints in thread_divs:
            ans.append( [calc_points[sum_points] , calc_points[sum_points+npoints-1]] )
            sum_points += npoints
        return np.array(ans)

    def observe(self):
        common = "incl %d phi %d posang %d" % (
            self.incl, self.phi, self.posang)
        option = "noscat nostar " # doppcatch"
        camera = "npixx {} npixy {} ".format(self.npixx, self.npixy)
        camera += "zoomau {:g} {:g} {:g} {:g} ".format(*(self.zoomau_x+self.zoomau_y))
        line = "iline {:d}".format(self.iline)
        v_calc_points = np.linspace( -self.vwidth_kms, self.vwidth_kms, self.linenlam )
        vseps = np.linspace( -self.vwidth_kms-self.dv_kms/2, self.vwidth_kms+self.dv_kms/2, self.linenlam+1   )

        if self.omp and (self.n_thread > 1):
            n_points = self.divide_threads(self.n_thread, self.linenlam )
            v_thread_seps = self.calc_thread_seps(v_calc_points, n_points)
            v_center = [ 0.5*(v_range[1] + v_range[0] ) for v_range in v_thread_seps ]
            v_width = [ 0.5*(v_range[1] - v_range[0] ) for v_range in v_thread_seps ]
            print("All calc points:\n", v_calc_points, '\n')
            print("Thread loading:\n", n_points, '\n')
            print("Calc points in each threads:")
            for i, (vc, vw, ncp) in enumerate(zip(v_center, v_width, n_points)):
                print( "%dth thread:"%i ,np.linspace(vc-vw, vc+vw, ncp)  )

            def cmd(p):
                freq = "vkms {:g} widthkms {:g} linenlam {:d} ".format(
                        v_center[p], v_width[p], n_points[p])
                return " ".join(["radmc3d image", line, freq, camera, common, option])

            args = [ (self, (p, 'proc'+str(p), cmd(p)) ) for p in range(self.n_thread)]
            rets = Pool(self.n_thread).map(call_subcalc, args)

            for i, r in enumerate(rets):
                print("\nThe",i,"th return")
                for k, v in r.__dict__.items():
                    if isinstance(v, (np.ndarray)):
                        print("{}: shape is {}, range is [{}, {}]".format(k, v.shape, np.min(v), np.max(v) ) )
                    else:
                        print("{}: {}".format(k, v ) )

            data = rets[0]
            for ret in rets[1:]:
                data.image = np.append(data.image, ret.image, axis=2)
                data.imageJyppix = np.append(data.imageJyppix, ret.imageJyppix, axis=2)
                data.freq = np.append(data.freq, ret.freq, axis=-1)
                data.wav = np.append(data.wav, ret.wav, axis=-1)
                data.nfreq += ret.nfreq
                data.nwav += ret.nwav
            self.data = data
        else:
            freq = "widthkms {} linenlam {:d} ".format(
                    self.vwidth_kms, self.linenlam)
            cmd = " ".join(["radmc3d image", line, freq, camera, common, option])
            print(cmd)
            os.chdir(self.dn_radmc)
            subprocess.call(cmd, shell=True)
            self.data = rmci.readImage()

        freq0 = (self.data.freq[0] + self.data.freq[-1])*0.5
        dfreq = self.data.freq[1] - self.data.freq[0]
        vkms = np.round(mytools.freq_to_vkms(freq0, self.data.freq-freq0), 8)
        print("x_au is:\n", np.round(self.data.x,8)/cst.au,"\n")
        print("v_kms is:\n", vkms,"\n")
        self.save_instance()
        self.save_fits()

        return self.data

    def subcalc(self, args):
        p, dn, cmd = args
        print("execute: ", cmd)
        dpath_sub = self.dn_radmc + '/' + dn
        if not os.path.exists(dpath_sub):
            os.makedirs(dpath_sub)
        os.system("cp %s/{*.inp,*.dat} %s/" % (self.dn_radmc, dpath_sub))
        os.chdir(dpath_sub)
        if p == 1:
            subprocess.call(cmd, shell=True)
        else:
            subprocess.call(cmd, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return rmci.readImage()

    def save_fits(self):
        fp_fitsdata = self.dn_fits+'/'+self.filename+'.fits'
        if os.path.exists(fp_fitsdata):
            os.remove(fp_fitsdata)
        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)

    def save_instance(self):
        pd.to_pickle(self, self.dn_fits+'/'+self.filename+'.pkl')

    def read_instance(self):
        instance = pd.read_pickle(self.dn_fits+'/'+self.filename+'.pkl')
        for k,v in instance.__dict__.items():
            setattr(self, k, v)

def call_subcalc(args_list):
    "This fucntion is implemented for fucking Python2."
    instance, args = args_list
    return getattr(instance, "subcalc")(args)


class FitsAnalyzer:
    def __init__(self, dn_radmc=None, dn_fig=None, filename=None, dpc=0, convolution_pvdiagram=True,
                 convolution_mom0map=True,
                 posang_PV = 0,
                 posang_beam = -41,
                 oplot_KeplerRotation = 0,
                 oplot_LocalPeak_Vax = 0,
                 oplot_LocalPeak_Pax = 0,
                 oplot_LocalPeak_2D = 1,
                 oplot_Maximums_2D = 1,
                 Mstar = 0.18,
                 beama_au = 10,
                 beamb_au = 10,
                 vwidth_kms = 0.5,
                 plotmode_PV='grid',
                 convolve_PV_p = True,
                 convolve_PV_v = True,
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
                                     v_width_kms=self.vwidth_kms, theta_deg=self.posang_beam)
        Ipp = integrate.simps(Ippv, axis=0)
        Plt = mp.Plotter(self.dn_fig, x=self.xau, y=self.yau)
        Plt.map(Ipp, out="mom0map",
                xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
                div=n_lv, mode='grid', cbmin=0, cbmax=Ipp.max())

    def pvdiagram(self, n_lv=5):
        Ippv = copy.copy(self.Ippv_raw)
        if self.convolution_pvdiagram:
            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
                                     v_width_kms=self.vwidth_kms, theta_deg=self.posang_beam)
            Ippv = self._perpix_to_perbeamkms(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au, v_width_kms=self.vwidth_kms)

        if len(self.yau) > 1:
            posang_PV_rad = self.posang_PV/180.*np.pi
            points = [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                       for r in self.xau ] for v in self.vkms]
            Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), Ippv, points)
        else:
            Ipv = Ippv[:, 0, :]

        pltr = mp.Plotter(self.dn_fig, x=self.xau/self.dpc, y=self.vkms, xlim=[-5, 5], ylim=[-3, 3])
        pltr.map(z=Ipv, out="pvd", mode=self.plotmode_PV,
                 xl="Angular Offset [arcsec]", yl=r"Velocity [km s$^{-1}$]",
                 cbl=r'Intensity [Jy pixel$^{-1}$ ]',
                 lw=2,
                 div=n_lv, save=False)

        if self.oplot_KeplerRotation:
            plt.plot(-self.x, np.sqrt(cst.G*self.Mstar*cst.Msun/ \
                     self.xau/cst.au)/cst.kms, c="cyan", ls=":")

        if self.oplot_LocalPeak_Vax:
            for Iv, v_ in zip(Ipv.transpose(0, 1), self.vkms):
                for xM in _get_peaks(self.xau, Iv):
                    plt.plot(xM, v_, c="red", markersize=1, marker='o')

        if self.oplot_LocalPeak_Pax:
            for Ip, x_ in zip(Ipv.transpose(1, 0), self.x):
                for vM in _get_peaks(self.vkms, Ip):
                    plt.plot(x_, vM, c="blue", markersize=1, marker='o')

        if self.oplot_LocalPeak_2D:
            for jM, iM in peak_local_max(Ipv, min_distance=5):
                plt.scatter(self.xau[iM], self.vkms[jM],
                            c="k", s=20, zorder=10)
                print("Local Max:   {:.1f}  au  , {:.1f}    km/s  ({}, {})".format(
                    self.xau[iM], self.vkms[jM], jM, iM))

        pltr.save("pvd")

    # theta : cclw is positive
    def _convolution(self, Ippv, beam_a_au, beam_b_au, v_width_kms, theta_deg=0):
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        Ippv_conv = copy.copy(Ippv)
        if self.convolve_PV_p:
            Kernel_xy = Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                         y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                         theta=theta_deg/180*np.pi)
            for i in range(self.Nz):
                Ippv_conv[i] = convolve_fft(Ippv_conv[i], Kernel_xy)

        if self.convolve_PV_v:
            Kernel_v = Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM)
            for j in range(self.Ny):
                for k in range(self.Nx):
                    Ippv_conv[:, j, k] = convolve_fft(Ippv_conv[:, j, k], Kernel_v)
        return Ippv_conv


    def _perpix_to_perbeamkms(self, intensity, beam_a_au, beam_b_au, v_width_kms):
        beam_area = np.pi * beam_a_au / 2.0 * beam_b_au / 2.0
        return intensity*(beam_area*v_width_kms)/(self.dx*self.dy*self.dv)

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
