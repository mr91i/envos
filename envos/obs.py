#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import sys
import shutil
import glob
import numpy as np
import pandas as pd
import copy
import textwrap
from logging import INFO
import contextlib
import multiprocessing
from scipy import integrate, interpolate, signal
import dataclasses
from dataclasses import dataclass, asdict, field
import pathlib

import astropy
import astropy.io.fits as afits
import astropy.convolution as aconv
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca

from . import nconst as nc
from . import tools
from . import gpath
from .log import logger
from .radmc3d import RadmcController
import skimage
from scipy import optimize, interpolate

#logger = set_logger(__name__)
#####################################


def gen_radmc_cmd(
    mode="image",
    dpc=0,
    incl=0,
    phi=0,
    posang=0,
    npixx=32,
    npixy=32,
    zoomau=[],
    lam=None,
    iline=None,
    vc_kms=None,
    vhw_kms=None,
    nlam=None,
    option="",
):
    position = f"dpc {dpc} incl {incl} phi {phi} posang {posang}"
    camera = f"npixx {npixx} npixy {npixy} "
    camera += f"zoomau {zoomau[0]} {zoomau[1]} {zoomau[2]} {zoomau[3]}"
    if lam is not None:
        freq = f"lambda {lam}"
    elif iline is not None:
        freq = f"iline {iline} widthkms {vhw_kms:g} linenlam {nlam:d}"
        freq += f" vkms {vc_kms:g}" if vc_kms else ""
    cmd = " ".join(["radmc3d", f"{mode}", position, camera, freq, option])
    return cmd


class ObsSimulator:
    """
    This class returns observation data
    Basically, 1 object per 1 observation
    """

    def __init__(self, config=None, radmcdir=None, dpc=None, n_thread=1):

        self.radmc_dir = radmcdir or gpath.radmc_dir
        self.dpc = dpc
        self.n_thread = n_thread
        self.conv = False
        self.incl = None
        self.phi = None
        self.posang = None

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config):
        self.config = config
        self.dpc = config.dpc
        self.n_thread = config.n_thread
        self.incl = config.incl
        self.phi = config.phi
        self.posang = config.posang
        self.iline = config.iline
        self.molname = config.molname
        self.lineobs_option = config.lineobs_option

        self.set_resolution(
            sizex_au=config.sizex_au or config.size_au,
            sizey_au=config.sizey_au or config.size_au,
            pixsize_au=config.pixsize_au,
            #    npix=config.npix,
            #    npixx=config.npixx,
            #    npixy=config.npixy,
            vfw_kms=config.vfw_kms,
            dv_kms=config.dv_kms,
        )

        self.set_convolver(
            beam_maj_au=config.beam_maj_au,
            beam_min_au=config.beam_min_au,
            vreso_kms=config.vreso_kms,
            beam_pa_deg=config.beam_pa_deg,
            convmode=config.convmode,
        )

    def set_model(self, model, conf=None):
        logger.info("Setting model for observation")
        if conf is None:
            conf = self.config
        radmc = RadmcController(config=conf)
        radmc.clean_radmc_dir()
        radmc.set_model(model)
        radmc.set_temperature(model.Tgas)
        radmc.set_lineobs_inpfiles()

    def set_radmc_input(self, model, conf):
        radmc = RadmcController(**conf.__dict__)
        radmc.clean_radmc_dirs()
        radmc.set_model(model)
        radmc.set_temperature(model.Tgas)
        radmc.set_lineobs_inpfiles()

    def set_resolution(
        self,
        sizex_au,
        sizey_au=None,
        pixsize_au=None,
        # npix=None,
        # npixx=None,
        # npixy=None,
        vfw_kms=None,
        dv_kms=None,
    ):

        logger.info("Setting resolution of ObsSimulator")

        if (sizey_au is not None) and (sizex_au != sizey_au):
            logger.warning(
                "Use rectangle viewing mode. "
                "Currently (2020/12), a bug in the mode was fixed. "
                "Please check if your version is latest. "
                "If you do not want to use this, "
                "please set sizex_au = sizey_au."
            )

        self.zoomau_x = [-sizex_au / 2, sizex_au / 2]
        _sizey_au = sizey_au or sizex_au
        self.zoomau_y = [-_sizey_au / 2, _sizey_au / 2]
        Lx = self.zoomau_x[1] - self.zoomau_x[0]
        Ly = self.zoomau_y[1] - self.zoomau_y[0]

        if pixsize_au is not None:
            self.pixsize_au = pixsize_au
            npixx_float = Lx / pixsize_au
            self.npixx = int(round(npixx_float))
            npixy_float = Ly / pixsize_au
            self.npixy = int(round(npixy_float))
            """comment: // or int do not work to convert into int."""
        else:
            logger.error("No pixsize_au.")
            raise Exception
        #    self.npixx = npixx or npix
        #    self.npixy = npixy or npix
        #    self.pixsize_au = Lx / self.npixx

        self.dx_au = sizex_au / self.npixx
        self.dy_au = _sizey_au / self.npixy

        if self.dx_au != self.dy_au:
            raise Exception("dx is not equal to dy. Check your setting again.")

        if sizex_au == 0:
            self.zoomau_x = [-self.pixsize_au / 2, self.pixsize_au / 2]
        if sizey_au == 0:
            self.zoomau_y = [-self.pixsize_au / 2, self.pixsize_au / 2]

        self.vfw_kms = vfw_kms
        self.dv_kms = dv_kms

    def set_convolver(
        self,
        beam_maj_au,
        beam_min_au=None,
        vreso_kms=None,
        beam_pa_deg=0,
        convmode="fft",
    ):

        logger.info("Setting convolution function of ObsSimulator")

        self.conv = True
        self.convolve_config = {
            "beam_maj_au": beam_maj_au,
            "beam_min_au": beam_min_au,
            "beam_pa_deg": beam_pa_deg,
            "vreso_kms": vreso_kms,
        }
        self.convolver = Convolver(
            (self.dx_au, self.dy_au, self.dv_kms),
            **self.convolve_config,
            mode=convmode,
        )

    def observe_cont(self, lam_mic=None, freq=None, incl=None, phi=None, posang=None, star=False):
        if lam_mic is None and freq is not None:
            lam_mic = nc.c / freq * 1e4

        incl = incl or self.incl
        phi = phi or self.phi
        posang = phi or self.posang

        logger.info(f"Observing continum with wavelength of {lam_mic} micron")
        zoomau = np.concatenate([self.zoomau_x, self.zoomau_y])


        cmd = gen_radmc_cmd(
            mode="image",
            dpc=self.dpc,
            incl=incl,
            phi=phi,
            posang=posang,
            npixx=self.npixx,
            npixy=self.npixx,
            lam=lam_mic,
            zoomau=zoomau,
            option="noscat" + ("" if star else " nostar") ,
        )

        tools.shell(cmd, cwd=self.radmc_dir, error_keyword="ERROR", log_prefix="    ")

        self.data_cont = rmci.readImage(fname=f"{self.radmc_dir}/image.out")
        self.data_cont.dpc = self.dpc
        self.data_cont.freq0 = nc.c / ( lam_mic/1e4 )
        odat = read_radmcdata(self.data_cont)

        if self.conv:
            odat.data = self.convolver(odat.data)
            odat.set_conv_info(**self.convolve_config)

        return odat

    def observe_line_profile(
        self, zoomau=None, iline=None, molname=None, incl=None, phi=None, posang=None
    ):
        iline = iline or self.iline
        molname = molname or self.molname
        incl = incl or self.incl
        phi = phi or self.phi
        posang = poang or self.posang

        cmd = gen_radmc_cmd(
            mode="image",
            dpc=self.dpc,
            incl=incl,
            phi=phi,
            posang=posang,
            npixx=self.npixx,
            npixy=self.npixx,
            lam=lam,
            zoomau=zoomau,
            option="noscat nostar",
        )

    #    return

    def observe_line(self, iline=None, molname=None, incl=None, phi=None, posang=None, obsdust=False):
        iline = iline or self.iline
        molname = molname or self.molname
        incl = incl or self.incl
        phi = phi or self.phi
        posang = posang or self.posang

        logger.info(f"Observing line with {molname}")
        self.nlam = int(round(self.vfw_kms / self.dv_kms))  # + 1
        mol_path = f"{self.radmc_dir}/molecule_{molname}.inp"
        self.mol = rmca.readMol(fname=mol_path)
        logger.info(
            f"Total cell number is {self.npixx}x{self.npixy}x{self.nlam}"
            + f" = {self.npixx*self.npixy*self.nlam}"
        )

        common_cmd = {
            "mode": "image",
            "dpc": self.dpc,
            "incl": incl,
            "phi": phi,
            "posang": posang,
            "npixx": self.npixx,
            "npixy": self.npixy,
            "zoomau": [*self.zoomau_x, *self.zoomau_y],
            "iline": self.iline,
            "option": "noscat nostar "
            + self.lineobs_option
            + " ",  # + (" doppcatch " if ,
        }
        if not obsdust:
            common_cmd["option"] += "nodust "

        v_calc_points = np.linspace(-self.vfw_kms / 2, self.vfw_kms / 2, self.nlam)

        if self.n_thread >= 2:
            logger.info(f"Use OpenMP dividing processes into velocity directions.")
            logger.info(f"Number of threads is {self.n_thread}.")
            # Load-balance for multiple processing
            n_points = self._divide_nlam_by_threads(self.nlam, self.n_thread)
            vrange_list = self._calc_vrange_list(v_calc_points, n_points)
            v_center = [0.5 * (vmax + vmin) for vmin, vmax in vrange_list]
            v_hwidth = [0.5 * (vmax - vmin) for vmin, vmax in vrange_list]

            logger.info(f"All calc points: {format_array(v_calc_points)}")
            logger.info("Calc points in each thread:")
            _zipped = zip(v_center, v_hwidth, n_points)
            for i, (vc, vhw, ncp) in enumerate(_zipped):
                vax_nthread = np.linspace(vc - vhw, vc + vhw, ncp)
                logger.info(f" -- {i}th thread: {format_array(vax_nthread)}")

            def cmdfunc(i):
                return gen_radmc_cmd(
                    vc_kms=v_center[i],
                    vhw_kms=v_hwidth[i],
                    nlam=n_points[i],
                    **common_cmd,
                )

            args = [(i, cmdfunc(i)) for i in range(self.n_thread)]

            with multiprocessing.Pool(self.n_thread) as pool:

                results = pool.starmap(self._subcalc, args)

            self._check_multiple_returns(results)
            self.data = self._combine_multiple_returns(results)

        else:
            logger.info(f"Not use OpenMP.")
            cmd = gen_radmc_cmd(vhw_kms=self.vfw_kms / 2, nlam=self.nlam, **common_cmd)

            tools.shell(cmd, cwd=self.radmc_dir)
            self.data = rmci.readImage(fname=f"{self.radmc_dir}/image.out")

        if np.max(self.data.image) == 0:
            print(vars(self.data))
            logger.warning("Zero image !")
            raise Exception

        #        self.data.dpc = self.dpc
        self.data.dpc = self.dpc
        self.data.freq0 = self.mol.freq[iline - 1]
        #odat = Cube(datatype="line")
        #odat.read(radmcdata=self.data)
        odat = read_radmcdata(self.data)
        odat.set_sobs_info(
            iline=iline, molname=molname, incl=incl, phi=phi, posang=posang
        )

        if self.conv:
            odat.Ippv = self.convolver(odat.Ippv)
            odat.set_conv_info(**self.convolve_config)

        return odat

    @staticmethod
    def find_proper_nthread(n_thr, n_div):
        return max([i for i in range(n_thr, 0, -1) if n_div % i == 0])

    @staticmethod
    def _divide_nlam_by_threads(nlam, nthr):
        divided_nlam_list = [nlam // nthr] * nthr
        remainder = nlam % nthr
        for i in range(remainder):
            i_distribute = (nthr - 1 - i // 2) if i % 2 else i // 2
            divided_nlam_list[i_distribute] += 1
        return divided_nlam_list

    @staticmethod
    def _calc_vrange_list(whole_vrange, divided_nlam_list):
        vrange_list = []
        sum_nlam = 0
        for nlam in divided_nlam_list:
            i_start = sum_nlam
            i_end = sum_nlam + nlam - 1
            vrange = (whole_vrange[i_start], whole_vrange[i_end])
            vrange_list.append(vrange)
            sum_nlam += nlam
        return np.array(vrange_list)

    def _subcalc(self, p, cmd):
        dn = f"proc{p:d}"
        dpath_sub = f"{self.radmc_dir}/{dn}"
        os.makedirs(dpath_sub, exist_ok=True)
        for f in glob.glob(f"{self.radmc_dir}/*"):
            if re.search(r".*\.(inp|dat)$", f):
                shutil.copy2(f, f"{dpath_sub}/")

        log = logger.isEnabledFor(INFO) and p == 0
        tools.shell(
            cmd,
            cwd=dpath_sub,
            log=log,
            error_keyword="ERROR",
            log_prefix="    ",
        )
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            fname = f"{dpath_sub}/image.out"
            return rmci.readImage(fname=fname)

    def _check_multiple_returns(self, return_list):
        for i, r in enumerate(return_list):
            logger.debug(f"The {i}th return")
            for k, v in r.__dict__.items():
                if isinstance(v, (np.ndarray)):
                    vrange = f"[{np.min(v)}, {np.max(v)}]"
                    logger.debug(f"{k}: shape {v.shape}, range {vrange}")
                else:
                    logger.debug(f"{k}: {v}")

    def _combine_multiple_returns(self, return_list):
        data = return_list[0]
        for ret in return_list[1:]:
            data.image = np.append(data.image, ret.image, axis=-1)
            data.imageJyppix = np.append(data.imageJyppix, ret.imageJyppix, axis=-1)
            data.freq = np.append(data.freq, ret.freq, axis=-1)
            data.wav = np.append(data.wav, ret.wav, axis=-1)
            data.nfreq += ret.nfreq
            data.nwav += ret.nwav
        return data

    def output_fits(self, filepath):
        fp_fitsdata = filepath
        if os.path.exists(fp_fitsdata):
            logger.info(f"remove old fits file: {fp_fitsdata}")
            os.remove(fp_fitsdata)
        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)
        logger.info(f"Saved fits file: {fp_fitsdata}")


def format_array(array):
    if len(array) >= 2:
        delta = abs(array[1] - array[0])
    else:
        delta = 0
    msg = f"[{min(array):.2f}:{max(array):.2f}] "
    msg += f"with delta = {delta:.4g} and N = {len(array)}"
    return msg


def convolve(
    image, beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=0, mode="scipy"
):
    convolver = Convolver(
        (image.dx_au, image.dy_au, image.dv_kms),
        beam_maj_au,
        beam_min_au,
        vreso_kms,
        beam_pa_deg,
        mode,
    )
    return convolver(image.Ipv)


class Convolver:
    """
    shape of 3d-image is (ix_max, jy_max, kv_max)
    Kernel

    """

    def __init__(
        self,
        grid_size,
        beam_maj_au=None,
        beam_min_au=None,
        vreso_kms=None,
        beam_pa_deg=0,
        mode="scipy",
    ):
        # relation : standard deviation = 1/(2 sqrt(ln(2))) * FWHM of Gaussian
        # theta_deg : cclw is positive
        self.mode = mode
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        conv_size = [beam_maj_au + 1e-100, beam_min_au + 1e-100]
        if vreso_kms is not None:
            conv_size += [vreso_kms + 1e-100]
        stddev = np.array(conv_size) / np.array(grid_size) / sigma_over_FWHM
        beampa = np.radians(beam_pa_deg)
        self.Kernel_xy2d = aconv.Gaussian2DKernel(
            x_stddev=stddev[0], y_stddev=stddev[1], theta=beampa
        )._array
        if len(conv_size) == 3 and (conv_size[2] is not None):
            Kernel_v1d = aconv.Gaussian1DKernel(stddev[2])._array
            self.Kernel_3d = np.multiply(
                # self.Kernel_xy2d[np.newaxis, :, :],
                self.Kernel_xy2d[:, :, np.newaxis],
                Kernel_v1d[np.newaxis, np.newaxis, :],
            )

    def __call__(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:
            Kernel = self.Kernel_xy2d
            logger.info("Convolving image with 2d-Kernael")
        elif len(image.shape) == 3:
            Kernel = self.Kernel_3d
            logger.info("Convolving image with 3d-Kernael")
        else:
            raise Exception("Unknown data.image shape: ", image.shape)

        logger.info("Kernel shape is %s", Kernel.shape)
        logger.info("Image shape is %s", image.shape)

        if self.mode == "normal":
            return aconv.convolve(image, Kernel)
        elif self.mode == "fft":
            return aconv.convolve_fft(image, Kernel, allow_huge=True)
            # from scipy.fftpack import fft, ifft
            # return aconv.convolve_fft(image, Kernel, allow_huge=True, nan_treatment='interpolate', normalize_kernel=True, fftn=fft, ifftn=ifft)
        elif self.mode == "scipy":
            return signal.convolve(image, Kernel, mode="same", method="auto")

        elif self.mode == "null":
            return image
        else:
            raise Exception("Unknown convolve mode: ", self.mode)


#    if pointsource_test:
#        I = np.zeros_like(I)
#        I[I.shape[0]//2, I.shape[1]//2, I.shape[2]//2] = 1

#########################################################
# Observation Data Classes
#########################################################

class BaseObsData:
    def __str__(self):
        return tools.dataclass_str(self)

    def copy(self):
        return copy.deepcopy(self)

    def convolve_image(self):
        conv = Convolver(
            (self.dx, self.dy),
            beam_maj_au=self.obreso.beam_maj_au,
            beam_min_au=self.obreso.beam_min_au,
            beam_pa_deg=self.obreso.beam_pa_deg,
        )
        self.data = conv(self.data)

    def set_dpc(self, dpc):
        self.dpc = dpc

    def set_conv_info(
        self, beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=None
    ):
        self.conv_info = {
            "beam_maj_au":  beam_maj_au,
            "beam_min_au": beam_min_au,
            "vreso_kms": vreso_kms,
            "beam_pa_deg": beam_pa_deg,
        }
        for k, v in self.conv_info.items():
            setattr(self, k, v)

    def set_obs_resolution(
        self, beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=None,
              beam_maj_deg=None, beam_min_deg=None,
    ):
        self.obreso = Obreso( \
            obsdata=self,
            beam_maj_au=beam_maj_au,
            beam_min_au=beam_min_au,
            beam_maj_deg=beam_maj_deg,
            beam_min_deg=beam_min_deg,
            vreso_kms=vreso_kms,
            beam_pa_deg=beam_pa_deg,
            dpc=self.dpc,
        )
        #for k, v in self.conv_info.items():
        #    setattr(self, k, v)


    def set_sobs_info(self, iline=None, molname=None, incl=None, phi=None, posang=None):
        """
        sobs info is nit used for actual observation data.
        """
        self.sobs_info = {
            "iline":  iline,
            "molname": molname,
            "incl": incl,
            "phi": phi,
            "posang": posang,
        }
        for k, v in self.sobs_info.items():
            setattr(self, k, v)

    def save(self, filename=None, basename="obsdata", mode="pickle", dpc=None, filepath=None):

        if filepath is None:
            if filename is None:
                output_ext = {"joblib": "jb", "pickle": "pkl", "fits": "fits"}[mode]
                filename = basename + "." + output_ext
            filepath = os.path.join(gpath.run_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"remove old fits file: {filepath}")
                os.remove(filepath)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if mode == "joblib":
            self._save_joblib(filepath)
        elif mode == "pickle":
            self._save_instance(filepath)
        elif mode == "fits":
            self._save_fits(filepath)

    def _save_joblib(self, filepath):
        import joblib
        joblib.dump(self, filepath, compress=3)
        logger.info(f"Saved joblib file: {filepath}")

    def _save_instance(self, filepath):
        pd.to_pickle(self, filepath)
        logger.info(f"Saved pickle file: {filepath}")

    def _save_fits(self, filename):
        save_fits(self, filename)
        logger.info(f"Saved fits file: {filename}")

    def save_instance(self, filename="obsdata.pkl", filepath=None):
        logger.warning(
            'The function "save_instance"  will be imcompatible in a future version. Instead please use a function:\n'
            '    save(basename="obsdata", mode="pickle", dpc=None, filepath=None) ,\n'
            '    selecting a mode that you want use among {"pickle"(default), "joblib", "fits"}.'
        )
        if filepath is None:
            filepath = os.path.join(gpath.run_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.to_pickle(self, filepath)

    def move_position(self, center_pos, ax="x", unit="au"): # ra=False, decl=False, deg=False, au=False):
        fac = 3600 * self.dpc

        if unit == "ra":
            d = self.convert_ra_to_deg(*center_pos) * fac
        elif unit == "decl":
            self.convert_decl_to_deg(*center_pos) * fac
        elif unit == "deg":
            d = center_pos * fac
        elif unit == "au":
            d = center_pos

        if ax == "x":
            self.xau -= d
        elif ax == "y":
            self.yau -= d

    def _reset_positive_axes_2(self):
        axnames = ["xau", "yau", "vkms"]
        daxnames = ["dx", "dy", "dv"]
        axLnames = ["Lx", "Ly", "Lv"]
        for _Ln, _axn in zip(axLnames, axnames):
            if hasattr(self, _axn):
                ax = getattr(self, _axn)
                if ax[1] < ax[0] and len(ax) >= 2:
                    setattr(self, _axn, ax[::-1])
                    setattr(self, _daxn, ax[1] - ax[0])
                    setattr(self, "image",  np.flip(getattr(self, "image"), i) )


    def _reset_positive_axes(self, img, axs):
        for i, ax in enumerate(axs):
            if (len(ax) >= 2) and (ax[1] < ax[0]):
                axs[i] = ax[::-1]
                img = np.flip(img, i)


    def convert_Jyppix_to_brightness_temperature(self, imageJyppix, freq0=None, lam0=None):
        if lam0 is not None:
            freq0 = nc.c / lam0
        # freq = tools.vkms_to_freq(self.vkms, freq0)
        lam0 = nc.c / freq0
        conv = 2.3504431e-11 * self.dx * self.dy / self.dpc**2 * 1e23
        Inu = (imageJyppix/conv).clip(0.0)
        Tb_RJ = Inu * lam0**2/(2 * nc.kB)
        Tb =  nc.h * freq0/nc.kB * 1./np.log(1. + 2.*nc.h*freq0**3/(Inu*nc.c**2) )
        Tb_app =  nc.h * freq0/nc.kB * np.nan_to_num( 1./np.log(1. + np.nan_to_num( 2.*nc.h*freq0**3/(Inu*nc.c**2) ) ) )
        # x =  2.*nc.h*freq0**3/(Inu*nc.c**2)
        #Tb_app =  nc.h * freq0/nc.kB * Inu*nc.c**2/2.*nc.h*freq0**3 * (1 + x/2 - x**2/12 + 1/24*x**3-19/720*x**4+3/160*x**5)
        # Tb_app = Inu*lam0**2/(2.*nc.kB) * (1. + x/2 - 1/12*x**2 + 1/24*x**3 - 19/720*x**4 + 3/160*x**5)
        #Tb_app =  nc.h * freq0/nc.kB * 1./np.log(1. + 2.*nc.h*freq0**3/(Inu*nc.c**2) )
        # 1e-6 - 1 * 0.1**2 /   1e-16  *
        print(f"Tb_app is {np.max(Tb)} K , Tb is {np.max(Tb)} K, Tb_RJ is {np.max(Tb_RJ)} K")
        if np.isnan(np.max(Tb_app)):
            raise Exception
        return Tb_app



    def get_Iname(self):
        return {"Cube":"Ippv", "Image":"Ipp", "Image2":"data", "PVmap":"Ipv" }[type(self).__name__]

    def get_I(self):
        return getattr(self, self.get_Iname())

    def set_I(self, _I):
        return setattr(self, self.get_Iname(), _I)

    def get_axes(self):
        #dtype = self.dtype() # type(self).__name__
       # print(dtype)
        if self.dtype == "Cube":
            return [self.xau, self.yau, self.vkms]
        elif self.dtype == "Image":
            return [self.xau, self.yau]
        elif self.dtype == "Image2":
            return [self.xau, self.yau]
        elif self.dtype == "PVmap":
            return [self.xau, self.vkms]
        else:
            raise Exception("Unknown class type")

#    def dtype(self):
#        return type(self).__name__

    def trim(self, xlim=None, ylim=None, vlim=None):
        if xlim is not None:
            nax = self._axorder["xau"]
            imin, imax = minmaxargs(self.xau, xlim)
            self.set_I(np.take(self.get_I(), range(imin,imax), axis=nax))
            self.xau = self.xau[imin:imax]

        if ylim is not None and hasattr(self, "yau"):
            nax = self._axorder["yau"]
            jmin, jmax = minmaxargs(self.yau, ylim)
            self.set_I(np.take(self.get_I(), range(jmin,jmax), axis=nax))
            self.yau = self.yau[jmin:jmax]

        if vlim is not None and hasattr(self, "vkms"):
            nax = self._axorder["vkms"]
            jmin, jmax = minmaxargs(self.vkms, vlim)
            self.set_I(np.take(self.get_I(), range(kmin,kmax), axis=nax))
            self.vkms = self.vkms[kmin:kmax]


    def copy_info_from_obsdata(self, obsdata):
        if hasattr(obsdata, "conv_info"):
            self.set_conv_info(
                beam_maj_au=obsdata.beam_maj_au,
                beam_min_au=obsdata.beam_min_au,
                vreso_kms=obsdata.vreso_kms,
                beam_pa_deg=obsdata.beam_pa_deg,
            )
            print("Here will be modified in near future")

        if hasattr(obsdata, "sobs_info"):
            self.set_sobs_info(
                iline=obsdata.iline,
                molname=obsdata.molname,
                incl=obsdata.incl,
                phi=obsdata.phi,
                posang=obsdata.posang,
            )

@dataclasses.dataclass
class Obreso:
    obsdata: dataclasses.InitVar[BaseObsData] = None
    beam_maj_deg: float = None
    beam_min_deg: float = None
    beam_pa_deg: float = None
    vreso_kms: float = None
    dpc: float = None
    beam_maj_au: float = None
    beam_min_au: float = None

    def __str__(self):
        return tools.dataclass_str(self)

    def __post_init__(self, obsdata):
        self.set_dpc_from_obsdata(obsdata)
        if self.beam_maj_au is None:
            self.set_beamsize_au()
        elif beam_maj_deg is None:
            self.set_beamsize_deg()

    def set_beamsize_au(self):
        if self.dpc is None:
            raise Exception("dpc is not set.")
        self.beam_maj_au = np.deg2rad(self.beam_maj_deg) * self.dpc * nc.pc/nc.au
        self.beam_min_au = np.deg2rad(self.beam_min_deg) * self.dpc * nc.pc/nc.au

    def set_beamsize_deg(self):
        if self.dpc is None:
            raise Exception("dpc is not set.")
        self.beam_maj_deg = np.rad2deg(self.beam_maj_au * nc.au / (self.dpc * nc.pc) )
        self.beam_min_deg = np.rad2deg(self.beam_min_au * nc.au / (self.dpc * nc.pc) )

    def set_dpc_from_obsdata(self, obsdata):
        if (obsdata is not None) and \
           (hasattr(obsdata, "dpc")) and \
           (obsdata.dpc is not None):
            self.dpc = obsdata.dpc
        else:
            print("Something wrong in obsdata")


#obj = Cube(Ippv, xau, yau, vkms, Iunit=Iunit, dpc=dpc, freq0=freq0)
@dataclasses.dataclass
class Cube(BaseObsData):
    """
    Usually, obsdata is made by doing obsevation.
    But one can generate obsdata by reading fitsfile or radmcdata
    """
    Ippv: np.ndarray
    xau: np.ndarray
    yau: np.ndarray
    vkms: np.ndarray
    Nx: int = None
    Ny: int = None
    Nv: int = None
    dx: float = None
    dy: float = None
    dv: float = None
    Lx: float = None
    Ly: float = None
    Lv: float = None
    dpc: float = None
#    beam_maj_au: float = None
#    beam_min_au: float = None
#    vreso_kms: float = None
#    beam_pa_deg: float = None
    conv_info: dict=None
    sobs_info: dict = None
    Iunit: str=None
    freq0: float = None
    #datatype: str = None
   #  _axorder: dict = field(default_factory={"xau":0, "yau":1, "vkms":2})
    dtype = "Cube"
    _axorder = {"xau":0, "yau":1, "vkms":2}

    def __post_init__(self):
        self._check_data_shape()
        self._reset_positive_axes(self.Ippv, [self.xau, self.yau, self.vkms])
        self._process_axinfo()

    def _check_data_shape(self):
        if self.Ippv.shape != (len(self.xau), len(self.yau), len(self.vkms)):
           logger.info("Data type error")
           raise Exception

    def _process_axinfo(self):
        self.Nx = len(self.xau)
        self.Ny = len(self.yau)
        self.Nv = len(self.vkms)
        self.dx = self.xau[1] - self.xau[0]
        self.dy = self.yau[1] - self.yau[0]
        self.dv = self.vkms[1] - self.vkms[0]
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.yau[-1] - self.yau[0]
        self.Lv = self.vkms[-1] - self.vkms[0]

    def get_mom0_map(self, normalize="peak", method="sum", vlim=None):
        if self.Ippv.shape[2] == 1:
            _Ipp = self.Ippv[...,0]
        else:
            if (vlim is not None) and (len(vlim) == 2):
                if vlim[0] < vlim[1]:
                    cond = np.where((vlim[0] < self.vkms) & (vkms < vlim[1]), True, False)
                    _Ipp = self.Ippv[..., cond]
                    _vkms = self.vkms[cond]
                else:
                    raise Exception

            if method == "sum":
                _Ipp = np.sum(self.Ippv, axis=-1) * (self.vkms[1] - self.vkms[0])

            elif method == "integrate":
                _Ipp = integrate.simps(self.Ippv, self.vkms, axis=-1)

        if normalize == "peak":
            _Ipp /= np.max(_Ipp)

        img = Image2(_Ipp, xau=self.xau, yau=self.yau, dpc=self.dpc)
        img.copy_info_from_obsdata(self)

        return img


    def get_pv_map(self, pangle_deg=0, poffset_au=0, Inorm="max", save=False):
    # Inorm: None, "max", float
        if self.Ippv.shape[1] > 1:
            posline = self.position_line(
                self.xau, PA_deg=pangle_deg, poffset_au=poffset_au
            )
            points = [[(pl[0], pl[1], v) for v in self.vkms] for pl in posline]
            Ipv = interpolate.interpn(
                (self.xau, self.yau, self.vkms),
                self.Ippv,
                points,
                bounds_error=False,
                method='linear',
                fill_value=0,
            )
        else:
            Ipv = self.Ippv[:, 0, :]

        pv = PVmap(Ipv, self.xau, self.vkms, self.dpc, pangle_deg, poffset_au)
        pv.copy_info_from_obsdata(self)
        pv.normalize(Inorm)
        if save:
            pv.save_fitsfile()
        # self.pv_list.append(pv)
        return pv

    def position_line(self, xau, PA_deg, poffset_au=0):
        PA_rad = (PA_deg + 90) * np.pi / 180
        pos_x = xau * np.cos(PA_rad) - poffset_au * np.sin(PA_rad)
        pos_y = xau * np.sin(PA_rad) + poffset_au * np.sin(PA_rad)
        return np.stack([pos_x, pos_y], axis=-1)

    #def move_center(self, center_pos, ra=False, decl=False, deg=False, au=False):


@dataclasses.dataclass
class Image(BaseObsData):
    Ipp: np.ndarray
    xau: np.ndarray
    yau: np.ndarray
    dpc: float = None
    Iunit: str = r"[Jy km s$^{-1}$ pixel$^{-1}$]"
#    beam_maj_au: float = None
#    beam_min_au: float = None
#    vreso_kms: float = None
#    beam_pa_deg: float = None
    conv_info: dict=None
    sobs_info: dict = None
    fitsfile: str = None
    freq0:float = None
    dtype = "Image"
    _axorder = {"xau":0, "yau":1}

    def __post_init__(self):
        self._check_data_shape()
        self._reset_positive_axes(self.Ipp, [self.xau, self.yau])
        self._process_axinfo()

    def _check_data_shape(self):
        if self.Ipp.shape != (len(self.xau), len(self.yau)):
            logger.info("Data type error")
            raise Exception

    def _process_axinfo(self):
        self.Nx = len(self.xau)
        self.Ny = len(self.yau)
        self.dx = self.xau[1] - self.xau[0]
        self.dy = self.yau[1] - self.yau[0]
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.yau[-1] - self.yau[0]

    def get_peak_position(self, interp=True):
        ip, jp = skimage.feature.peak_local_max(self.Ipp, num_peaks=1)[0]
        xau_peak, yau_peak = self.xau[ip], self.yau[jp]

        if interp:
            fun = interpolate.RectBivariateSpline(self.xau, self.yau, self.Ipp)
            dx = self.xau[1] - self.xau[0]
            dy = self.yau[1] - self.yau[0]
            res = optimize.minimize(
                lambda _x: 1 / fun(_x[0], _x[1])[0, 0],
                [xau_peak, yau_peak],
                bounds=[
                    (xau_peak - 4 * dx, xau_peak + 4 * dx),
                    (yau_peak - 4 * dy, yau_peak + 4 * dy),
                ],
            )
            return res.x[0], res.x[1]

        return self.xau[ip], self.yau[jp]

    def convert_perbeam_to_perpixel(self): # this could go base
        bmaj = self.ob["beam_maj_au"]
        bmin = self.conv_info["beam_min_au"]
        A_beam = np.pi * bmaj * bmin
        A_pix = self.dx * self.dy
        self.Ipp *= A_pix / A_beam
        self.Iunit = "Jy/pix"

@dataclasses.dataclass
class Image2(BaseObsData):
    data: np.ndarray
    ra: np.ndarray = None # in rad
    dec: np.ndarray = None # in rad
    ra0: float = 0
    dec0: float = 0
    #ra_deg: np.ndarray = None # in deg
    #dec_deg: np.ndarray = None # in deg
    #dec0_deg: float = 0
    #ra0_deg: float = 0
    xau: np.ndarray = None
    yau: np.ndarray = None
    dpc: float = None
    Iunit: str = r"[Jy pixel$^{-1}$]"
#    beam_maj_au: float = None
#    beam_min_au: float = None
#    vreso_kms: float = None
#    beam_pa_deg: float = None
    obreso: dict=None
    sobs_info: dict = None
    fitsfile: str = None
    freq0:float = None
    dtype = "Image"
    _axorder = {"ra":0, "dec":1, "xau":0, "yau": 1}
    radec_deg: dataclasses.InitVar[tuple] = None

    def __post_init__(self, radec_deg):
        if radec_deg:
            self.set_radec_rad(*radec_deg)
        self._complement_coord()
        self._check_data_shape()
        self._reset_positive_axes(self.data, [self.xau, self.yau])
        self._process_axinfo()

    def _complement_coord(self):
        if self.ra is not None and self.dec is not None:
            self.calc_radec_to_stdcoord()
       # elif self.rad_deg is not None and self.dec_deg is not None:
       #     self.update_radec_rad()
       #     self.calc_radec_to_stdcoord()
        elif self.xau is not None and self.yau is not None:
            self.calc_stdcoord_to_radec()
        else:
            raise Exception("Incomplete data")

    def calc_radec_to_stdcoord(self):
        self.xau =  -self.dpc * nc.pc/nc.au * np.cos(self.dec0) * self.ra
        self.yau = self.dpc * nc.pc/nc.au * self.dec

    def calc_stdcoord_to_radec(self):
        if self.dpc is None:
            self.dpc = 1
            print("Please set dpc")
        self.dec0 = 0.5 * (self.yau[-1] + self.yau[0] ) * nc.au/(self.dpc * nc.pc)
        self.ra0 = -0.5 * (self.xau[-1] + self.xau[0] ) * nc.au/(self.dpc * nc.pc * np.cos(self.dec0))
        self.dec = self.yau * nc.au/(self.dpc * nc.pc)
        self.ra = -self.xau * nc.au/(self.dpc * nc.pc * np.cos(self.dec0))

    def _check_data_shape(self):
        if self.data.shape != (len(self.ra), len(self.dec)):
            logger.info("Data type error")
            raise Exception

    def _process_axinfo(self):
        self.Nx = self.Nra = len(self.ra)
        self.Ny = self.Ndec = len(self.dec)
        self.dra = self.ra[1] - self.ra[0]
        self.ddec = self.dec[1] - self.dec[0]
        self.dx = self.xau[1] - self.xau[0]
        self.dy = self.yau[1] - self.yau[0]
        self.Lra = self.ra[-1] - self.ra[0]
        self.Ldec = self.dec[-1] - self.dec[0]
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.xau[-1] - self.xau[0]

    def get_peak_position(self, interp=True):
        #ip, jp = skimage.feature.peak_local_max(self.data, num_peaks=1)[0]
        ip, jp = np.unravel_index(np.argmax(self.data), self.data.shape)
        xau_peak, yau_peak = self.xau[ip], self.yau[jp]

        if interp:
            fun = interpolate.RectBivariateSpline(self.xau, self.yau, self.data)
            dx = self.xau[1] - self.xau[0]
            dy = self.yau[1] - self.yau[0]
            res = optimize.minimize(
                lambda _x: 1 / fun(_x[0], _x[1])[0, 0],
                [xau_peak, yau_peak],
                bounds=[
                    (xau_peak - 4 * dx, xau_peak + 4 * dx),
                    (yau_peak - 4 * dy, yau_peak + 4 * dy),
                ],
            )
            return res.x[0], res.x[1]

        return self.xau[ip], self.yau[jp]

    def reset_center(self, xyau=None, radecdeg=None):
        if xyau is not None:
            self.xau -= xyau[0]
            self.yau -= xyau[1]
            self.calc_stdcoord_to_radec()

        elif radecdeg is not None:
            self.dec0 -= np.deg2rad( decdeg[0] )
            self.rad0 -= np.deg2rad( decdeg[1] )
            self.dec -= np.deg2rad( decdeg[0] )
            self.rad -= np.deg2rad( decdeg[1] )
            self.calc_radec_to_stdcoord()

    def reset_center_to_maximum(self):
        xc, yc = self.get_peak_position(interp=False)
        self.reset_center(xyau=(xc, yc))


    def convert_perbeam_to_perpixel(self): # this could go base
        bmaj = self.obreso.beam_maj_au # = full width half maximum along major axis
        bmin = self.obreso.beam_min_au
        A_beam = np.pi * bmaj * bmin/(4*np.log(2))
        A_pix = self.dx * self.dy
        self.data *= A_pix / A_beam
        self.Iunit = "Jy/pix"

    def set_radec_deg(self):
        """
        Set ra-dec coordinate in degree. This is useful sometimes.
        """
        self.dec0_deg = np.rad2deg(self.dec0)
        self.rad0_deg = np.rad2deg(self.rad0)
        self.dec_deg = np.rad2deg(self.dec)
        self.rad_deg = np.rad2deg(self.rad)

    def set_radec_rad(self, ra0, dec0, ra, dec):
        self.dec0 = np.deg2rad(dec0)
        self.ra0 = np.deg2rad(ra0)
        self.dec = np.deg2rad(dec)
        self.ra = np.deg2rad(ra)


@dataclasses.dataclass
class PVmap(BaseObsData):
    Ipv: np.ndarray = None
    xau: np.ndarray = None
    vkms: np.ndarray = None
    dpc: float = None
    pangle_deg: float = None
    poffset_au: float = None
    Iunit: str = r"[Jy pixel$^{-1}$]"
#    beam_maj_au: float = None
#    beam_min_au: float = None
#    vreso_kms: float = None
#    beam_pa_deg: float = None
    conv_info: dict=None
    sobs_info: dict = None
    freq0:float = None
    #_axorder: dict = field(default_factory={"xau":0, "vkms":1})
    _axorder ={"xau":0, "vkms":1}
    dtype = "PVmap"

    def __post_init__(
        self,
    ):
        self._check_data_shape()
        self._reset_positive_axes(self.Ipv, [self.xau, self.vkms])
        self._process_axinfo()
    def _check_data_shape(self):
        if self.Ipv.shape != (len(self.xau), len(self.vkms)):
            logger.info("Data type error")
            raise Exception


    def _process_axinfo(self):
        self.Nx = len(self.xau)
        self.Nv = len(self.vkms)
        self.dx = self.xau[1] - self.xau[0]
        self.dv = self.vkms[1] - self.vkms[0]
        self.Lx = self.xau[-1] - self.xau[0]
        self.Lv = self.vkms[-1] - self.vkms[0]


    def normalize(self, Ipv_norm=None):
        if Ipv_norm is None:
            self.Ipv /= 1
            self.Iunit = r"[Jy pixel$^{{-1}}$]"  # rf"[{Ipv_norm} Jy pixel$^{{-1}}$]]"
        elif Ipv_norm == "max":
            self.Ipv /= np.max(self.Ipv)
            self.Iunit = r"[$I_{\rm max}$]"
        elif isinstance(Ipv_norm, float):
            self.Ipv /= Ipv_norm
            self.Iunit = f"[{Ipv_norm}" + r"Jy pixel$^{-1}$]"
        else:
            return None

    def reverse_x(self):
        self.Ipv = self.Ipv[:, ::-1]
        self.xau = -self.xau[::-1]

    def reverse_v(self):
        self.Ipv = self.Ipv[::-1, :]
        self.vkms = -self.vkms[::-1]

    def reverse_xv(self):
        self.reverse_x()
        self.reverse_v()

    def reversed_x(self):
        pv = self
        pv.Ipv = self.Ipv[:, ::-1]
        pv.xau = -self.xau[::-1]
        return pv

    def reversed_v(self):
        pv = self
        pv.Ipv = self.Ipv[::-1, :]
        pv.vkms = -self.vkms[::-1]
        return pv

    def reversed_xv(self):
        pv = self.reversed_x()
        return pv.reversed_v()

    def offset_x(self, dx):
        # self.Ipv = self.Ipv[:,::-1]
        self.xau += dx

    def offset_v(self, dv):
        self.vkms += dv

    def trim(self, range_x=None, range_v=None):
        if range_x is not None:
            imax, imin = [np.abs(self.xau - xl).argmin() for xl in range_x]
            self.Ipv = self.Ipv[:, imin:imax+1]
            self.xau = self.xau[imin:imax+1]

        if range_v is not None:
            jmin, jmax = [np.abs(self.vkms - vl).argmin() for vl in range_v]
            self.Ipv = self.Ipv[jmin:jmax+1, :]
            self.vkms = self.vkms[jmin:jmax+1]




#########################################################
# Save & Read Functions
#########################################################
def save_fits(od, filename):
    """
    save the obsdata as a fitsfile
    see IAU manual for variables used in fits:
         https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
    """
    filepath = gpath.run_dir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    hdu = afits.PrimaryHDU(od.get_I().T)
    ## I may need to change ppv into vyx , ...Ivyx?
    dtype = type(od).__name__
    hd = {
        "DTYPE": dtype,
        "DPC": od.dpc,
        "BTYPE": "Intensity",
        "BUNIT": "Jy/pixel",
    }


    def axis_dict(Nax, nax, dax, axname):
        if (axname == "x") or (axname == "y"):
            dic = {
                f"CTYPE{nax}": "ANGLE",
                f"NAXIS{nax}": Nax,
                f"CRVAL{nax}": 0.0,
                f"CRPIX{nax}": (Nax + 1.) / 2.,
                f"CDELT{nax}": au2deg(dax, od.dpc), #np.rad2deg( np.pi*dax/(648000* od.dpc) ),
                f"CUNIT{nax}": "deg"
            }
        elif axname=="v":
            dic = {
                f"CTYPE{nax}": "VRAD",
                f"NAXIS{nax}": od.Nv,
                f"CRVAL{nax}": 0.0,
                f"CRPIX{nax}": (od.Nv + 1) / 2,
                f"CDELT{nax}": od.dv * 1e3,
                f"CUNIT{nax}": "m/s"
            }
        return dic

    if dtype == "Cube":
        hd.update({"NAXIS": 3})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Ny, 2, od.dy, "y"))
        hd.update(axis_dict(od.Nv, 3, od.dv, "v"))
    elif dtype == "Image":

        print(od)
        hd.update({"NAXIS": 2})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Ny, 2, od.dy, "y"))
    elif dtype == "PVmap":
        hd.update({"NAXIS": 2})
        hd.update(axis_dict(od.Nx, 1, od.dx, "x"))
        hd.update(axis_dict(od.Nv, 2, od.dv, "v"))

    if od.beam_maj_au:
        hd.update({
            "BMAJ": au2deg(od.beam_maj_au, od.dpc),
            "BMIN": au2deg(od.beam_min_au, od.dpc),
            "BPA": od.beam_pa_deg,
            "VRESO":od.vreso_kms,
        })

    infonames = {"iline":"iline", "molname":"molname", "incl":"incl", "phi":"phi", "posang":"posang", "freq0":"resfrq"}
    hd.update( (_fitsn, getattr(od, _odn)) for _odn, _fitsn in infonames.items() if hasattr(od, _odn) )
    hdu.header.update(hd)
    hdulist = afits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)

# Readers to be deleted
def read_obsdata(path, mode=None):
    path = str(path)
    if (".pkl" in path) or (mode == "pickle"):
        return tools.read_pickle(path)

    elif (".jb" in path) or (mode == "joblib"):
        import joblib

        return joblib.load(path)

    elif (".fits" in path) or (mode == "fits"):
        #logger.error("Still constructing...Sorry...")
        #sys.exit(1)
        print("do fits")
        return None

    else:
        logger.error("Still constructing...Sorry...")
        sys.exit(1)
        # raise Exception("Still constructing...Sorry")
        return Cube(filepath=path)

def read_radmcdata(data):
    if len(data.image.shape) == 2:
        Ipp = data.imageJyppix/ data.dpc**2
        dtype="image"
    elif len(data.image.shape) == 3 and data.image.shape[2] == 1:
        Ipp = data.imageJyppix[:, :, 0] / data.dpc**2
        dtype="image"
    elif len(data.image.shape) == 3:
        Ippv = data.imageJyppix / data.dpc**2  # .transpose(2, 1, 0)
        dtype="cube"
    Iunit = "Jy/pixel"

    dpc = data.dpc  # or 100
    Nx = data.nx
    Ny = data.ny
    Nv = data.nfreq
    Nf = data.nfreq
    xau = data.x / nc.au
    yau = data.y / nc.au
    freq = data.freq
    freq0 = data.freq0
    vkms = tools.freq_to_vkms_array(freq, freq0)
    dx = data.sizepix_x / nc.au
    dy = data.sizepix_y / nc.au
    dv = vkms[1] - vkms[0] if len(vkms) > 1 else 0
    # Check
    #if data.nx != len(xau) or data.ny != len(yau)  or data.nfreq != len(vkms) or dx != (xau[1]-xau[0]) or  dy != (yau[1]-yau[0]) or dv != (vkms[1] - vkms[0]):
    #    print("Something might be wrong in reading a radmcdata")
    #    exit(1)

    print(Cube)
    if dtype == "cube":
        obj = Cube(Ippv, xau, yau, vkms, Iunit=Iunit, dpc=dpc, freq0=freq0)

    elif dtype == "image":
        obj = Image2(Ipp, xau=xau, yau=yau, Iunit=Iunit, dpc=dpc, freq0=freq0)

    return obj

def read_image_fits(
    filepath, unit1="au", unit2="au", dpc=1, unit1_cm=None, unit2_cm=None
):
    return read_fits(filepath, "image", dpc=dpc)

def read_pv_fits(
    filepath, unit1="au", unit2="kms", dpc=1, unit1_cm=None, unit2_cms=None, v0_kms=0
):
    return read_fits(filepath, "pv", dpc=dpc)

def read_cube_fits(
    filepath, unit1="au", unit2="au", unit3="kms", dpc=1, unit1_cm=None, unit2_cm=None, unit3_cms=None, v0_kms=0
):
    return read_fits(filepath, "cube", dpc=dpc)

#!Constracting!

from astropy.nddata import CCDData

def read_fits(filepath, fitstype, dpc, unit1=None, unit2=None, unit3=None, unit1_fac=None, unit2_fac=None, unit3_fac=None, Iunit=None):
    logger.info(f"Reading fits file: {filepath}")
    hdul = afits.open(filepath)[0]
    header = hdul.header
    data = hdul.data.T ## .T is due to data shape

#    print(header)

    naxis = header["NAXIS"]
    wcs = astropy.wcs.WCS(header, naxis=naxis)
#    logger.debug("header:\n" + textwrap.fill(str(header), 80) + "\n")
    centerpix = wcs.wcs.crpix # should be improved into the center pixcoord
    axref =  wcs.wcs_pix2world([centerpix], 0)[0]

    Iunit = Iunit if Iunit is not None else (header["BUNIT"] if "BUNIT" in header else None)
    unit1 = unit1 if unit1 is not None else ( header["CUNIT1"].rstrip() if "CUNIT1" in header else None )
    unit2 = unit2 if unit2 is not None else ( header["CUNIT2"].rstrip() if "CUNIT2" in header else None )
    unit3 = unit3 if unit3 is not None else ( header["CUNIT3"].rstrip() if "CUNIT3" in header else None )

    def _get_ax(axnum):
        n = header[f"NAXIS{axnum:d}"]
        pixcoord = np.tile(centerpix, (n,1))

#        print(pixcoord)
#        ax_2 = wcs.array_index_to_world_values(pixcoord, 0)[:,axnum-1]
#        print(axnum, ax_2[1] - ax_2[0] )

        pixcoord[:,axnum-1] = np.arange(n)
#        print(pixcoord)
        ax = wcs.wcs_pix2world(pixcoord, 0)[:,axnum-1]
        axc =  wcs.wcs_pix2world([centerpix], 0)[:,axnum-1]
        print(axnum, ax[1] - ax[0])
        ax_1 = wcs.p4_pix2foc(pixcoord, 0)[:,axnum-1]
#        axc_1 =  wcs.all_pix2world([centerpix], 0)[:,axnum-1]
        print(axnum, ax_1)
        return ax - axc

    def _add_beam_info(obj, dpc, vkms=None):
        if not "BMAJ" in header:
            return
        beam_maj_deg = header["BMAJ"] #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        beam_min_deg = header["BMIN"] #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        beam_pa_deg = header["BPA"]

        if (vkms is not None) and (len(vkms) >= 2):
            vreso_kms = vkms[1] - vkms[0]
        else:
            vreso_kms = None
        #obj.set_conv_info(
        #    beam_maj_au,
        #    beam_min_au,
        #    vreso_kms,
        #    beam_pa_deg,
        #)
        obj.set_obs_resolution(
            beam_maj_deg=beam_maj_deg,
            beam_min_deg=beam_min_deg,
            vreso_kms=vreso_kms,
            beam_pa_deg=beam_pa_deg,
        )

    def _data_shape_convert(data, ndim, transpose=True):
        if len(np.shape(data)) != ndim:
            nshape = [i for i in np.shape(data) if i!=1]
            if len(nshape) == ndim:
                ndata = np.reshape(data, nshape)
            else:
                logger.error(f"Wait, something wrong in the data! nshape = {nshape}")
                raise Exception
        else:
            ndata = data
        #print(ndata.shape)
        #if transpose:
        #    ndata = ndata.T
        return ndata

    def _get_lenscale(unit_name=None, unit_cm=None):
        if unit_cm is not None:
            logger.info(
                "   1st axis is interpreted as POSITION "
                f"with user-defined unit (unit = {unit_cm} cm)."
            )
            fac = unit_cm / nc.au
        elif unit_name.lower() in ["degree", "deg"]:
            fac = 3600 * dpc
        elif unit_name.lower() in ["au"]:
            fac = 1.0
        else:
            raise Exception("Unknown datatype in 1st axis")
        return fac

    def _get_velscale(unit_name=None, unit_cms=None):
        if unit_cms:
            logger.info(
                "   **nd axis is interpreted as VELOCITY "
                f"with user-defined unit (unit = {unit_cms} cm/s)."
            )
            fac = unit_cms / 1e5

        elif unit_name.lower() in ["ms", "m/s"]:
            fac = 1e-3

        elif unit_name.lower() in ["kms", "km/s"]:
            fac = 1.0

        elif unit_name.lower() in ["Hz"]:  # when dnu is in Hz
            nu0 = header["RESTFRQ"]  # freq: max --> min
            fac = nc.c / 1e5 / nu0
            logger.info(
                f"Here nu0 is supposed to be {nu0/1e9} GHz. If it is wrong, please let S.M know, because it should be a bug."
            )

        else:
            raise Exception("Unknown datatype in 2nd axis")

        return fac

    freq0 = header[f"RESTFRQ"] if "RESTFREQ" in header else None

    if fitstype.lower()=="image":
        ax1 = _get_ax(1) #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2) #* _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        data = _data_shape_convert(data, 2)
        #print(data.shape, _get_ax(1).shape, xau.shape, yau.shape)
        if unit1 == "deg":
            radec_deg=(axref[0], axref[1], ax1, ax2)
            im = Image2(data, radec_deg=radec_deg, dpc=dpc, Iunit=Iunit, freq0=freq0)
        elif unit1 == "au":
            im = Image2(data, xau=ax1, yau=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)
        _add_beam_info(im, dpc)
        obj = im
        """
        xau = _get_ax(1) * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        yau = _get_ax(2) * _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        data = _data_shape_convert(data, 2)
        #print(data.shape, _get_ax(1).shape, xau.shape, yau.shape)
        im = Image(data, xau, yau, dpc=dpc, Iunit=Iunit, freq0=freq0)
        _add_beam_info(im, dpc)
        obj = im
        """

    elif fitstype.lower()=="pv":
        xau = _get_ax(1) * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        vkms = _get_ax(2) * _get_velscale(unit_name=unit2, unit_cms=unit2_fac)
        data = _data_shape_convert(data, 2)
        pv = PVmap(data, xau, vkms, dpc=dpc, Iunit=Iunit, freq0=freq0)
        _add_beam_info(pv, dpc, vkms)
        obj = pv

    elif fitstype.lower()=="cube":
        xau = _get_ax(1) * _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        yau = _get_ax(2) * _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        vkms = _get_ax(3) * _get_velscale(unit_name=unit3, unit_cms=unit3_fac)
        data = _data_shape_convert(data, 3)
        cube = Cube(data, xau, yau, vkms, dpc=dpc, Iunit=Iunit, freq0=freq0)
        _add_beam_info(cube, dpc, vkms)
        obj = cube

    return obj





#########################################################
# Conversion functions
#########################################################

def au2deg(length_au, dpc):
     return np.rad2deg( np.pi*length_au/(648000*dpc) )

def deg_to_ra(deg):
    hour = int(deg/15)
    minu = int((deg/15 - hour)*60)
    seco = (deg/15 - hour - minu/60 )*3600
    return hour, minu, seco

def deg_to_decl(deg):
    _deg = int(deg)
    arcmin = int(60*(deg - _deg))
    arcsec = 3600*(deg - _deg - arcmin/60)
    return _deg, arcmin, arcsec

def ra_to_deg(hour, minu, sec):
    return 15*(hour + minu/60 + sec/(3600))

def decl_to_deg(deg, arcmin, arcsec):
    return deg + arcmin/60 + arcsec/(3600)

#########################################################
# Tools
#########################################################

def minmaxargs(array, lim):
    imin, imax = np.take(np.argwhere((array>lim[0]) & (array<lim[-1])), (0,-1))
    return imin, imax+1


if __name__=="__main__":
    obj = read_fits(
        "/home/smori/my-envos/ShareMori/260G_spw1_C3H2_v.fits",
        "cube",
        dpc=140,
        unit1="deg",
        unit2="deg",
        unit3="m/s",
    )
    print(obj)
