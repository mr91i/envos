#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import integrate, optimize, interpolate
from skimage.feature import peak_local_max
import astropy.io.fits as iofits
import astropy.convolution as aconv
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import logging

from pmodes.header import inp, dpath_home, dpath_radmc, dpath_fig
import pmodes.myplot as mp
from pmodes import cst, tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#####################################

def main():
    #obs = ObsData(fitsfile=dpath_radmc + '/' + inp.vis.filename + '.fits', **vars(inp.vis))
    #obs = ObsData(fitsfile=dpath_radmc + '/' + inp.vis.filename + '.fits', use_inp=inp.vis)
    obs = ObsData(use_inp=inp.vis)
    #obs_data = ObsData(**vars(inp.fitsa))
    #obs_data.read_fits()
    conv_info = ConvolutionInfo(inp.vis.beama_au, inp.vis.beamb_au, inp.vis.vwidth_kms, inp.vis.beam_posang)
    obs.convolve(conv_info)
    obs.make_mom0_map()
    # plot_mom0_map(obs)

    obs.make_PV_map()
    # plot_pvdiagram(obs.PV_list[0])


    #fa = FitsAnalyzer(dpath_radmc=dpath_radmc, dpath_fig=dpath_fig, **vars(inp.fitsa))
    #fa.pvdiagram()
    #fa.mom0map()
#    fa.chmap()

vis = InputParams()

vis.fits_filename = obs.filename
vis.dpc = dpc
vis.posang_PV = obs.posang

vis.normalize = {1:None, 2:"peak"}[2]
vis.Imax = 10e-4 * (radmc.mol_abun/1e-7)
vis.Mstar = model.Mstar_Msun
vis.f_crit = 0.2

# beam
vis.convolver = "fft"
vis.beama_au = 0.4 * vis.dpc * beam_scale
vis.beamb_au = 0.4 * vis.dpc * beam_scale
vis.beam_posang = 0.0 # -41
vis.vwidth_kms = 0.4 * beam_scale

vis.plotmode_PV = {1:"grid", 2:"contour", 3:"contourf", 4:"contourfill"}[2]
vis.mass_estimation = 1
vis.poffset_yau = 0.0

vis.oplot_KeplerRotation = 0
vis.oplot_LocalPeak_Vax = 0
vis.oplot_LocalPeak_Pax = 0
vis.oplot_LocalPeak_2D = 0
vis.oplot_Maximums_2D = 1




class PVmap:
    def __init__(self, Ipv, pau, vkms, pangle_deg, poffset_au):
        self.Ipv = Ipv
        self.pau = pau
        self.vkms = vkms
        self.pangle_deg = pangle_deg
        self.poffset_au = poffset_au
        self.Imax = np.max(Ipv)
        self.unit_I = r'[Jy pixel$^{-1}$]'

    def normalize(Ipv_norm=None):
        if Ipv_norm is None:
            self.Ipv /= slef.Imax
            self.unit_I = r'[I_{\rm max}]'
        else:
            self.Ipv /= Ipv_norm
            self.unit_I = rf'[{Ipv_norm} Jy pixel$^{{-1}}$]]'

    def save_fitsfile(self, pau, vkms, filename="PVimage.fits", unitp='au', unitv='km/s' ):
        np = len(pau)
        nv = len(vkms)

        dp = (pau[1] - pau[0])
        dv = (vkms[1] - vkms[0])

        hdu = iofits.PrimaryHDU(Image2d)
        hdu.header['NAXIS'] = 2

        hdu.header['CTYPE1'] = 'Position'
        hdu.header['CUNIT1'] = unitp
        hdu.header['NAXIS1'] = np
        hdu.header['CRVAL1'] = 0.0
        hdu.header['CRPIX1'] = (np + 1.)/2.  # index of pixel of the referece point
        hdu.header['CDELT1'] = dp

        hdu.header['CTYPE2'] = 'Velocity'
        hdu.header['CUNIT2'] = unitv
        hdu.header['NAXIS2'] = nv
        hdu.header['CRVAL2'] = 0.0
        hdu.header['CRPIX2'] = (nv + 1.)/2.
        hdu.header['CDELT2'] = dv

        hdu.header['BTYPE'] =  'Intensity'
        hdu.header['BUNIT'] =  'Jy/pixel'

    #    hdu.header['BMAJ'] = 3.2669255431920E-05
    #    hdu.header['BMIN'] = 2.1551359086060E-05
    #    hdu.header['BPA'] =  -8.5868155934400E+01
        hdulist = iofits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)


#class ConvolutionInfo(self):
#    def __init__(self, beam_a_au, beam_b_au, v_width_kms, theta_deg, mode_convolve="fft", pointsource_test=False#):
#        self.beam_a_au = beam_a_au
#        self.beam_b_au = beam_n_au
#        self.v_width_kms = v_width_kms
#        self.theta_deg = theta_deg
#        self.mode_convolve = mode_convolve
#        self.pointsource_test = pointsource_test

class ObsData:
    def __init__(self, fitsfile=None, radmcdata=None, dpc=None):
        #tools.set_arguments(self, locals())
        #print(kwargs)
        self.Ippv = None
        self.Ippv_raw = None
        self.dpc = dpc
        self.Nx = None
        self.Ny = None
        self.Nv = None
        self.xau = None
        self.yau = None
        self.vkms = None
        self.dx = None
        self.dy = None
        self.dv = None
        self.Lx = None
        self.Ly = None
        self.Lv = None

        self.conv = False
        self.beam_a_au = None
        self.beam_b_au = None
        self.v_width_kms = None
        self.beam_theta_deg = None
        self.mode_convolve = None

#        if use_inp:
#            logger.info(vars(use_inp))
#            fitsfile = "{inp.}{inp.fits_filename}"

        # self.PA_rad = self.posang_PV/180.*np.pi

        # beam information
        #self.beama_au = None
        #self.beamb_au = None


        if fitsfile is not None:
            self.read_fits(fitsfile)
        elif radmcdata is not None:
            self.read_radmcdata(radmcdata)
        else:
            pass

    def read_fits(self, file_path, dpc):
        pic = iofits.open(file_path)[0]
        self.Ippv_raw = pic.data
        self.Ippv = copy.copy(self.Ippv_raw)
        header = pic.header

        self.Nx = header["NAXIS1"]
        self.Ny = header["NAXIS2"]
        self.Nz = header["NAXIS3"]
        self.dx = - header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
        self.dy = + header["CDELT2"]*np.pi/180.0*self.dpc*cst.pc/cst.au
        self.Lx = self.Nx*self.dx
        self.Ly = self.Ny*self.dy
        self.xau = -0.5*self.Lx + (np.arange(self.Nx)+0.5)*self.dx
        self.yau = - 0.5*self.Ly + (np.arange(self.Ny)+0.5)*self.dy

        if header["CRVAL3"] > 1e8: # when dnu is in Hz
            nu_max = header["CRVAL3"] # freq: max --> min
            dnu = header["CDELT3"]
            nu0 = nu_max + 0.5*dnu*(self.Nz-1)
            self.dv = - cst.c / 1e5 * dnu / nu0
        else:
            self.dv = header["CDELT3"]/1e3
        self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))

        if (self.dx < 0) or (self.xau[1] < self.xau[0]):
            raise Exception("Step in x-axis is negative")

        if (self.dy < 0) or (self.yau[1] < self.yau[0]):
            raise Exception("Step in y-axis is negative")

        if (self.dv < 0) or (self.vkms[1] < self.vkms[0]):
            raise Exception("Step in x-axis is negative")

        logger.info(f"fits file path: {file_path}")
        logger.info(f"pixel size[au]: {self.dx}  {self.dy}")
        logger.info(f"L[au]: {self.Lx} {self.Ly}")

#
#   Freeze: Revive in future version
#
#    def read_fits_PV(self, filepath):
#        pic = iofits.open(filepath)[0]
#        Ipv = pic.data
#        self.Ipv = copy.copy(self.Ippv_raw)
#        header = pic.header
#
#        self.Nx = header["NAXIS1"]
#        self.Nz = header["NAXIS2"]
#        self.dx = header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
#        Lx = self.Nx*self.dx
#        self.xau = -0.5*Lx + (np.arange(self.Nx)+0.5)*self.dx
#
#        if header["CRVAL2"] > 1e8:
#            nu_max = header["CRVAL2"] # freq: max --> min
#            dnu = header["CDELT2"]
#            nu0 = nu_max + 0.5*dnu*(self.Nz-1)
#            self.dv = - cst.c / 1e5 * dnu / nu0
#            self.vkms = (-0.5*(self.Nz-1)+np.arange(self.Nz)) * self.dv
#        else:
#            self.dv = header["CDELT2"]/1e3 # in m/s to in km/s
#            self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))
#        self.Ippv_raw = self.Ippv_raw[::-1,:]
#       # print(self.xau, self.dx, self.vkms, self.dv)
#
#        if ((self.dx < 0) or (self.xau[1] < self.xau[0])) \
#            or ((self.dv < 0) or (self.vkms[1] < self.vkms[0])):
#            raise Exception("reading axis is wrong.")


    def read_radmcdata(self, radmcdata):
       print(vars(radmcdata))

        pass


    #def convolve(self, beam_a_au, beam_b_au, v_width_kms, theta_deg=0, mode_convolve="fft", pointsource_test=False):
    #def convolve(self, conv_size, theta_deg=0, mode_convolve="fft", pointsource_test=False):
 #   def convolve(self, conv_info):
    def convolve(self, beam_a_au, beam_b_au, v_width_kms, beam_theta_deg, mode_convolve="fft", pointsource_test=False):
        # theta_deg : cclw is positive
        # enroll convolution info.
        self.conv = True
        self.beam_a_au = beam_a_au
        self.beam_b_au = beam_b_au
        self.v_width_kms = v_width_kms
        self.beam_theta_deg = beam_theta_deg
        self.mode_convolve = mode_convolve

        # setting
       # self.conv_info = conv_info
        self.convolution = True
        sigma_over_FWHM = 2*np.sqrt(2*np.log(2))
        convolver = {"normal": aconv.convolve, "fft":aconv.convolve_fft}[mode_convolve]
        option = {"allow_huge": True}

        if pointsource_test:
            self.pointsource_test = True
            Ippv = np.zeros_like(Ippv)
            Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1

        # making Kernel
        Kernel_xy2d = aconv.Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
                                             y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
                                             theta=theta_deg/180*np.pi)._array
        Kernel_v1d = aconv.Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM)._array
        Kernel_3d = np.multiply(Kernel_xy2d[np.newaxis,:,:], Kernel_v1d[:,np.newaxis,np.newaxis])

        # convolving
        self.Ippv = convolver(self.Ippv, Kernel_3d, **option)

        # flattening very small value
        self.Ippv = np.where(self.Ippv > np.max(self.Ippv)*1e-6, Ippv, -1e-100)
        self.Ippv_max = np.max(self.Ippv)


    def make_mom0_map(self, normalize="peak"):
        Ipp = integrate.simps(self.Ippv, axis=0)
        if normalize == "peak":
            Ipp /= np.max(Ipp)
        self.Imom0 = Ipp

    def position_line(self, PA_deg, poffset_au=0):
        PA_rad = PA_rad * np.pi/180
        pos_x = xau*np.cos(PA_rad) - poffset_au*np.sin(PA_rad)
        pos_y = xau*np.sin(PA_rad) + poffset_au*np.sin(PA_rad)
        return np.stack([pos_x, pos_y], axis=-1)

    def make_PV_map(self, pangle_deg=0, poffset_au=0):

        if self.Ippv.shape[1] > 1:
            posline = self.position_line(PA_deg=pangle_deg, poffset_au=poffset_au)
            points = [[(v, ply, plx) for plx, ply in zip(posline)] for v in self.vkms]
            Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), self.Ippv, points, bounds_error=False, fill_value=0)

        else:
            Ipv = self.Ippv[:,0,:]

        PV = PVmap(Ipv, self.xau, self.vkms, pangle_deg, poffset_au)
        PV.normalize()
        PV.save_fitsfile()

        self.PV_list.append(PV)


#    def __init__(self, dpath_radmc=None, dpath_fig=None, filename=None, dpc=0, convolution_pvdiagram=True,
#                 convolution_mom0map=True, posang_PV=0,
#                 beam_posang=-41, oplot_KeplerRotation=0,
#                 oplot_LocalPeak_Vax=0, oplot_LocalPeak_Pax=0,
#                 oplot_LocalPeak_2D=1, oplot_Maximums_2D=1,
#                 Mstar=0.18, beama_au=10, beamb_au=10,
#                 vwidth_kms=0.5, plotmode_PV='grid',
#                 convolve_PV_p=True, convolve_PV_v=True,
#                 convolver='normal', pointsource_test=False,
#                 P_offset_yau=0, f_crit=None,
#                 logcolor_PV=False, normalize=None, oplot_ire_fit=False, Imax=None,
#                 mass_estimation=True,
#                 ):


#    def save_fits(self, obj):
#        fp_fitsdata = self.dpath_fits+'/'+self.filename+'.fits'
#        if os.path.exists(fp_fitsdata):
#            msg("remove old fits file:",fp_fitsdata)
#            os.remove(fp_fitsdata)
#        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)
#        msg("Saved fits file:",fp_fitsdata)
#



#class FitsAnalyzer:
#    def __init__(self, dpath_radmc=None, dpath_fig=None, filename=None, dpc=0, convolution_pvdiagram=True,
#                 convolution_mom0map=True, posang_PV=0,
#                 beam_posang=-41, oplot_KeplerRotation=0,
#                 oplot_LocalPeak_Vax=0, oplot_LocalPeak_Pax=0,
#                 oplot_LocalPeak_2D=1, oplot_Maximums_2D=1,
#                 Mstar=0.18, beama_au=10, beamb_au=10,
#                 vwidth_kms=0.5, plotmode_PV='grid',
#                 convolve_PV_p=True, convolve_PV_v=True,
#                 convolver='normal', pointsource_test=False,
#                 P_offset_yau=0, f_crit=None,
#                 logcolor_PV=False, normalize=None, oplot_ire_fit=False, Imax=None,
#                 mass_estimation=True,
#                 ):
#
#        tools.set_arguments(self, locals(), printer=msg)
#
#        self.fits_file_path = dpath_radmc + '/' + filename + '.fits'
#        pic = iofits.open(self.fits_file_path)[0]
#        self.Ippv_raw = pic.data
#        header = pic.header
#        self.isconvolved = False
#        self.datatype="ppv"
#
#        if self.datatype=="ppv":
#            self.Nx = header["NAXIS1"]
#            self.Ny = header["NAXIS2"]
#            self.Nz = header["NAXIS3"]
#            self.dx = - header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
#            self.dy = + header["CDELT2"]*np.pi/180.0*self.dpc*cst.pc/cst.au
#            Lx = self.Nx*self.dx
#            Ly = self.Ny*self.dy
#            self.xau = -0.5*Lx + (np.arange(self.Nx)+0.5)*self.dx
#            self.yau = - 0.5*Ly + (np.arange(self.Ny)+0.5)*self.dy
#
#            if header["CRVAL3"] > 1e8:
#                nu_max = header["CRVAL3"] # freq: max --> min
#                dnu = header["CDELT3"]
#                nu0 = nu_max + 0.5*dnu*(self.Nz-1)
#                self.dv = - cst.c / 1e5 * dnu / nu0
#                self.vkms = (-0.5*(self.Nz-1)+np.arange(self.Nz)) * self.dv
#            else:
#                self.dv = header["CDELT3"]/1e3
#                self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))
#
#            if ((self.dx < 0) or (self.xau[1] < self.xau[0])) \
#                or ((self.dy < 0) or (self.yau[1] < self.yau[0])) \
#                or ((self.dv < 0) or (self.vkms[1] < self.vkms[0])):
#                raise Exception("reading axis is wrong.")
#            print("fits file path: {}".format(self.fits_file_path))
#            print("pixel size[au]: {} {}".format(self.dx, self.dy))
#            print("L[au]: {} {}".format(Lx, Ly))
#
#        elif self.datatype=="pv":
#
#            self.Nx = header["NAXIS1"]
#            self.Nz = header["NAXIS2"]
#            self.dx = header["CDELT1"]*np.pi/180.0*self.dpc*cst.pc/cst.au
#            Lx = self.Nx*self.dx
#            self.xau = -0.5*Lx + (np.arange(self.Nx)+0.5)*self.dx
#
#            if header["CRVAL2"] > 1e8:
#                nu_max = header["CRVAL2"] # freq: max --> min
#                dnu = header["CDELT2"]
#                nu0 = nu_max + 0.5*dnu*(self.Nz-1)
#                self.dv = - cst.c / 1e5 * dnu / nu0
#                self.vkms = (-0.5*(self.Nz-1)+np.arange(self.Nz)) * self.dv
#            else:
#                self.dv = header["CDELT2"]/1e3 # in m/s to in km/s
#                self.vkms = self.dv*(-0.5*(self.Nz-1) + np.arange(self.Nz))
#            self.Ippv_raw = self.Ippv_raw[::-1,:]
#           # print(self.xau, self.dx, self.vkms, self.dv)
#
#            if ((self.dx < 0) or (self.xau[1] < self.xau[0])) \
#                or ((self.dv < 0) or (self.vkms[1] < self.vkms[0])):
#                raise Exception("reading axis is wrong.")
#
#        self.pangle_rad = self.posang_PV/180.*np.pi
#
#    def position_line(self, poffset_au=0):
#        #print(poffset_au)
#        return np.array([[r*np.cos(self.pangle_rad) -poffset_au*np.sin(self.pangle_rad),
#                          r*np.sin(self.pangle_rad) +poffset_au*np.cos(self.pangle_rad)]
#                             for r in self.xau]).T
#
#    def chmap(self, n_lv=20):
#        Ippv = copy.copy(self.Ippv_raw)
#        if self.convolution_mom0map:
#            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
#                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
##        xx, yy = np.meshgrid(self.xau, self.yau)
#        cbmax = Ippv.max()
#
#        pltr = mp.Plotter(self.dpath_fig, x=self.xau, y=self.yau,
#                          xl="Position [au]", yl="Position [au]",
#                          cbl=r'Intensity [Jy pixel$^{-1}$ ]')
#        for i in range(self.Nz):
#            pltr.map(Ippv[i], out="chmap_{:0=4d}".format(i),
#                     n_lv=n_lv, ctlim=[0, cbmax], mode='grid',
#                     title="v = {:.3f} km/s".format(self.vkms[i]) , square=True)
#
#    def mom0map(self, n_lv=10):
#        Ippv = copy.copy(self.Ippv_raw)
#        if self.convolution_mom0map:
#            Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
#                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
#        Ipp = integrate.simps(Ippv, axis=0)
#        if self.normalize == "peak":
#            Ipp /= np.max(Ipp)
#
#        pltr = mp.Plotter(self.dpath_fig, x=self.xau, y=self.yau)
#        pltr.map(Ipp, out="mom0map",
#                xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
#                div=n_lv, mode='grid', ctlim=[0,Ipp.max()], square=True, save=False)
#
#        self.show_center_line(pltr.ax)
#        pline0 = self.position_line()
#        pline = self.position_line(poffset_au=self.P_offset_yau)
#        plt.plot(pline0[0], pline0[1], ls='--', c='w', lw=1)
#        plt.plot(pline[0], pline[1], c='w', lw=1)
#        self.show_beamsize(pltr, mode="mom0")
#        pltr.save("mom0map")
#
#
#    def pvdiagram(self, n_lv=5):
#        Ippv = copy.copy(self.Ippv_raw)
#        unit = r'[Jy pixel$^{-1}$]'
#        if self.pointsource_test:
#            Ippv = self.use_pointsource(Ippv)
#
#        if self.datatype=="ppv":
#
#            if self.convolution_pvdiagram:
#                Ippv = self._convolution(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au,
#                                     v_width_kms=self.vwidth_kms, theta_deg=self.beam_posang)
#
#                #Ippv = self._perpix_to_perbeam(Ippv, beam_a_au=self.beama_au, beam_b_au=self.beamb_au, v_width_kms=self.vwidth_kms)
#                #unit = r'[Jy beam$^{-1}$]'
#
#                #Ippv = self._perpix_to_pergunit(Ippv)
#                #unit = r'[Jy cm$^{-2}$ (km/s)$^{-1}$ ]'
#
#            if len(self.yau) > 1:
#                points = [[(v, ply, plx) for plx, ply in zip(*self.position_line(poffset_au=self.P_offset_yau))] for v in self.vkms]
#                print(np.shape(points))
#                Ipv = interpolate.interpn((self.vkms, self.yau, self.xau), Ippv, points, bounds_error=False, fill_value=0)
#            else:
#                Ipv = Ippv[:, 0, :]
#
#        elif self.datatype=="pv":
#            Ipv = self.Ippv_raw
#
#        if self.normalize == "peak":
#            if np.max(Ipv) > 0:
#                Ipv /= np.max(Ipv)
#                ctlim = [1/n_lv, 1]
#                n_lv -= 1
#            else:
#                ctlim = [0, 0.1]
#            unit = r'[$I_{\rm max}$]'
#        else:
#            ctlim = [0, np.max(Ipv)]  if not self.Imax else [0, self.Imax]
#
#        # Set ctlim
#        if self.logcolor_PV:
#            ctlim[1] = ctlim[0]*0.001
#
##        if output_fits:
#        if 1:
#            create_radmc3dImage(Ipv, self.xau, self.vkms, filename= self.dpath_fig+'/PV.fits')
#
#    #import astropy.io.fits as fits
#            #hdu = fits.PrimaryHDU(Ipv)
#            #hdulist = fits.HDUList([hdu])
#            #print(hdu, hdulist)
#            #hdulist.writeto('new.fits', overwrite=True)
#
#        xas = self.xau/self.dpc
#        pltr = mp.Plotter(self.dpath_fig, x=xas, y=self.vkms, xlim=[-5, 5], ylim=[-4, 4])
#
##        logoption = {"logcb":True, "ctlim":[np.max(Ipv)*1e-2,np.max(Ipv)]} if self.logcolor_PV else {"ctlim":[0, np.max(Ipv) for not self.Imax in self.Imax ]}
#        pltr.map(z=Ipv, mode=self.plotmode_PV, logcb=self.logcolor_PV, ctlim=ctlim, #ctlim=[np.max(Ipv)*1e-2,np.max(Ipv)],
#                 xl="Angular Offset [arcsec]", yl=r"Velocity [km s$^{-1}$]",
#                 cbl=r'Intensity '+unit,
#                 lw=1.5, #logx=True, logy=True, xlim=[0.1, 100], ylim=[0.1, 10],
#                 div=n_lv, save=False, clabel=True)
#
#        self.show_center_line(pltr.ax)
#        pltr.ax.minorticks_on()
#        pltr.ax.tick_params("both",direction="inout" )
#        pltr.ax2 = pltr.ax.twiny()
#        pltr.ax2.minorticks_on()
#        pltr.ax2.tick_params("both",direction="inout" )
#        pltr.ax2.set_xlabel("Angular Offset [au]")
#        pltr.ax2.set_xlim(np.array(pltr.ax.get_xlim())*self.dpc)
#
#        rCR = 200*cst.au
#        l = np.sqrt(cst.G*self.Mstar*cst.Msun*rCR)
#        a = (2*self.xau*cst.au/rCR)
#
#        if self.oplot_ire_fit:
#            pltr.ax.plot(xas, 2*l/rCR*(2*self.xau.clip(0)*cst.au/rCR)**(-2/3)/cst.kms, c="hotpink", ls=":", lw=1)
#
#        if self.oplot_KeplerRotation:
#            pltr.ax.plot(xas, np.sqrt(cst.G*self.Mstar*cst.Msun/ \
#                     (self.xau*cst.au))/cst.kms, c="cyan", ls=":", lw=1)
#
#        if self.oplot_LocalPeak_Pax:
#            for Iv, v_ in zip(Ipv.transpose(0, 1), self.vkms):
#                for xM in get_localpeak_positions(xas, Iv, threshold_abs=np.max(Ipv)*1e-10):
#                    pltr.ax.plot(xM, v_, c="red", markersize=1, marker='o')
#
#        if self.oplot_LocalPeak_Vax:
#            for Ip, x_ in zip(Ipv.transpose(1, 0), xas):
#                for vM in get_localpeak_positions(self.vkms, Ip, threshold_abs=np.max(Ipv)*1e-10):
#                    pltr.ax.plot(x_, vM, c="blue", markersize=1, marker='o')
#
#        if self.oplot_LocalPeak_2D:
#            for jM, iM in peak_local_max(Ipv, num_peaks=4, min_distance=10):#  min_distance=None):
#                pltr.ax.scatter(xas[iM], self.vkms[jM], c="k", s=20, zorder=10)
#                print("Local Max:   {:.1f}  au  , {:.1f}    km/s  ({}, {})".format(
#                    self.xau[iM], self.vkms[jM], jM, iM))
#
#            del jM, iM
#
#        def draw_cross_pointer(x, y, c):
#            pltr.ax.axhline(y=y, lw=2, ls=":", c=c, alpha=0.6)
#            pltr.ax.axvline(x=x, lw=2, ls=":", c=c, alpha=0.6)
#            pltr.ax.scatter(x, y, c=c, s=50 , alpha=1, linewidth=0, zorder=10)
#
#        if self.mass_estimation:
#            #vp_peaks = np.array([(self.vkms[jM], self.xau[iM]) for jM, iM
#            #                     in peak_local_max(Ipv[len(Ipv[0])//2:, len(Ipv[1])//2:], num_peaks=4)])
#            jmax, imax = Ipv.shape
#            #print(Ipv.shape, imax, jmax)
#            #print(self.vkms.shape, self.xau.shape)
#            jpeak, ipeak = peak_local_max( Ipv[jmax//2:, imax//2:], num_peaks=1)[0]
#            vkms_peak, xau_peak = self.vkms[jmax//2 + jpeak], self.xau[imax//2 + ipeak]
#
#
##            vp_peaks = [ (self.vkms[jM], self.xau[iM]) for jM, iM in peak_local_max( Ipv[imax//2:, jmax//2:], num_peaks=1) ]
##
##            i = np.argmax( np.sign(vp_peaks[:,0])*vp_peaks[:,0] * vp_peaks[:,1]  )
##            vkms_peak, xau_peak = vp_peaks[i]
#            M_CR = (abs(xau_peak) * cst.au * (vkms_peak*cst.kms)**2 )/(cst.G*cst.Msun)
#            draw_cross_pointer(xau_peak/self.dpc, vkms_peak, mp.c_def[1])
#
#            x_vmax, I_vmax = np.array([[ get_maximum_position(self.xau, Iv), np.max(Iv) ] for Iv in Ipv.transpose(0, 1)]).T
#            if (np.min(I_vmax) < self.f_crit*ctlim[1]) and (self.f_crit*ctlim[1] < np.max(I_vmax)):
#                print("Use {self.f_crit}*ctlim")
#                v_crit = tools.find_roots(self.vkms, I_vmax, self.f_crit*ctlim[1])
#            else:
#                print("Use {self.f_crit}*max Ippv")
#                v_crit = tools.find_roots(self.vkms, I_vmax, self.f_crit*np.max(Ippv) )
#            if len(v_crit) == 0 :
#                v_crit = [self.vkms[0]]
#
#            x_crit = tools.find_roots(x_vmax, self.vkms, v_crit[0])
#            M_CB = (abs(x_crit[0]) * cst.au * (v_crit[0]*cst.kms)**2 )/(2*cst.G*cst.Msun)
#            M_CR_vpeak = (abs(x_crit[0]) * cst.au * (v_crit[0]*cst.kms)**2 )/(np.sqrt(2)*cst.G*cst.Msun)
#            draw_cross_pointer(x_crit[0]/self.dpc, v_crit[0], mp.c_def[0])
#            txt = rf"$M_{{\rm ip}}$={M_CR:.3f}" + "\n"\
#                  +rf"$M_{{\rm vp,{self.f_crit*100}\%}}$={M_CB:.3f}"
#            plt.text(0.95, 0.05,txt,
#                     transform=pltr.ax.transAxes, ha="right", va="bottom", bbox=dict(fc="white", ec="black", pad=5))
#
#        self.show_beamsize(pltr, mode="PV")
#        pltr.save("pvd")
#
#    # theta : cclw is positive
#    def _convolution(self, Ippv, beam_a_au, beam_b_au, v_width_kms, theta_deg=0, ver="new"):
#        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
#        Ippv_conv = copy.copy(Ippv)
#        convolver = {"normal": aconv.convolve, "fft":aconv.convolve_fft}[self.convolver]
#        option = {"allow_huge": True}
#        if self.isconvolved==True:
#            return self.Ippv_conv
#        if ver=="new" and self.isconvolved==False:# super fast
#            Kernel_2d = aconv.Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
#                                               y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
#                                               theta=theta_deg/180*np.pi)._array
#            Kernel_1d = aconv.Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM)._array
#            Kernel_3d = np.multiply(Kernel_2d[np.newaxis,:,:], Kernel_1d[:,np.newaxis, np.newaxis])
#            #Kernel_3d /= np.sum(Kernel_3d)
#            Ippv_conv = convolver(Ippv_conv, Kernel_3d, **option)
#            #return  Ippv_conv.clip(np.max(Ippv_conv)*1e-3)
#            self.isconvolved = True
#            self.Ippv_conv = np.where( Ippv_conv > np.max(Ippv_conv)*1e-6, Ippv_conv, -1e-100)
#            return self.Ippv_conv
#
#        if ver=="old":
#            if self.convolve_PV_p:
#                Kernel_xy = aconv.Gaussian2DKernel(x_stddev=abs(beam_a_au/self.dx)/sigma_over_FWHM,
#                                          y_stddev=abs(beam_b_au/self.dy)/sigma_over_FWHM,
#                                         #x_size= len(self.xau) + 1,#int(abs(beam_a_au/self.dx)/sigma_over_FWHM)*12+1,
#            #                             y_size=4*len(self.yau) + 1,#int(abs(beam_b_au/self.dy)/sigma_over_FWHM)*12+1,
#                                         theta=theta_deg/180*np.pi)
#                for i in range(self.Nz):
#                    Ippv_conv[i] = convolver(Ippv_conv[i], Kernel_xy, **option)
#
#            if self.convolve_PV_v:
#                Kernel_v = aconv.Gaussian1DKernel(v_width_kms/self.dv/sigma_over_FWHM,)
#                             #           x_size=4*len(self.vkms) + 1) #int(v_width_kms/self.dv/sigma_over_FWHM)*12+1)
#                for j in range(self.Ny):
#                    for k in range(self.Nx):
#                        Ippv_conv[:, j, k] = convolver(Ippv_conv[:, j, k], Kernel_v, **option)
#
#            return np.where( Ippv_conv > np.max(Ippv_conv)*1e-8, Ippv_conv, 0)
#
#    def _perpix_to_perbeam(self, intensity_ppix, beam_a_au, beam_b_au, v_width_kms):
#        beam_area = np.pi * beam_a_au / 2.0 * beam_b_au / 2.0
#        pixel_in_beam = beam_area/(self.dx*self.dy)
#        return intensity_ppix * pixel_in_beam
#
#    def _perpix_to_pergunit(self, intensity):
#        return intensity*1./(self.dx*self.dy*self.dv)
##(self.dx*self.dy*self.dv)*cst.au**2*cst.kms
##(cst.au**2 * cst.km/s)
#
##        return intensity*(self.dx*self.dy*self.dv)
#
#    def use_pointsource(self, Ippv):
#        Ippv = np.zeros_like(Ippv)
#        Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1
#        return Ippv
#
#
#    def show_center_line(self,ax):
#        ax.axhline(y=0, lw=2, ls=":", c="k", alpha=1, zorder=1)
#        ax.axvline(x=0, lw=2, ls=":", c="k", alpha=1, zorder=1)
#        ax.scatter(0, 0, c="k", s=10 , alpha=1, linewidth=0,  zorder=1)
#
#    def show_beamsize(self, plot_cls, mode=None, with_box=False):
#        import matplotlib.patches as pat
#        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
#        if mode=="PV":
#            beam_crosslength_asec = (np.cos(self.beam_posang/180*np.pi)/self.beama_au**2 + np.sin(self.beam_posang/180*np.pi)/self.beamb_au**2 )**(-0.5)/self.dpc
#            beamx = beam_crosslength_asec
#            beamy = self.vwidth_kms
#        elif mode=="mom0":
#            beamx = self.beama_au
#            beamy = self.beamb_au
#
#        if with_box:
#            e1 = pat.Ellipse(xy=(0,0), width=beamx, height=beamy, lw=1, fill=True, ec="k", fc="0.6")
#        else:
#            e1 = pat.Ellipse(xy=(0,0), width=beamx, height=beamy, lw=0, fill=True, ec="k", fc="0.7", alpha=0.6)
#        box = AnchoredAuxTransformBox(plot_cls.ax.transData, loc='lower left', frameon=with_box, pad=0., borderpad=0.4)
#        box.drawing_area.add_artist(e1)
#        box.patch.set_linewidth(1)
#        plot_cls.ax.add_artist(box)
#
#    def save_fits(self, obj):
#        fp_fitsdata = self.dpath_fits+'/'+self.filename+'.fits'
#        if os.path.exists(fp_fitsdata):
#            msg("remove old fits file:",fp_fitsdata)
#            os.remove(fp_fitsdata)
#        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)
#        msg("Saved fits file:",fp_fitsdata)




#def create_radmc3dImage(Image2d, x1_ax, x2_ax, freq, unitx_cm=1, unity_cm=1 ):
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
# fig.savefig(dpath_fig+"emsv_img_plf.pdf")


if __name__ == '__main__':
    main()


