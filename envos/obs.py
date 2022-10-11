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
    au2pc = nc.au / nc.pc
    fac_deg2au = np.pi / 180 * nc.pc / nc.au

    def __str__(self):
        return tools.dataclass_str(self)

    def _reset_positive_axes(self, img, axs):
        for i, ax in enumerate(axs):
            if (len(ax) >= 2) and (ax[1] < ax[0]):
                axs[i] = ax[::-1]
                img = np.flip(img, i)


    def _check_data_shape(self):
        lens = tuple([ len(ax) for ax in self.get_axes() ])
        if self.get_I().shape != lens:
            logger.info("Data type error")
            raise Exception

    """
    Dynamical variables
    """

    def __getattr__(self, vname):
        dyv =  self._dynamic_vars(vname)
        if dyv is not None:
            return dyv
        else:
            raise AttributeError(vname)

    def _dynamic_vars(self, key):
        if key == "dx":
            return self.xau[1] - self.xau[0]
        elif key == "dy":
            return self.yau[1] - self.yau[0]
        elif key == "dv":
            return self.vkms[1] - self.vkms[0]
        elif key == "Lx":
            return self.xau[-1] - self.xau[0]
        elif key == "Ly":
            return self.yau[-1] - self.yau[0]
        elif key == "Lv":
            return self.vkms[-1] - self.vkms[0]
        elif key == "Nx":
            return len(self.xau)
        elif key == "Ny":
            return len(self.yau)
        elif key == "Nv":
            return len(self.vkms)
        else:
            return None

    def copy(self):
        return copy.deepcopy(self)

    def convolve_image(self): ## !! need to be changed
        conv = Convolver(
            (self.dx, self.dy),
            beam_maj_au=self.obreso.beam_maj_au,
            beam_min_au=self.obreso.beam_min_au,
            beam_pa_deg=self.obreso.beam_pa_deg,
        )
        self.data = conv(self.data)

    """
    Get and set I and axes
    """

    def get_Iname(self):
        return self._Iname

    def get_I(self):
        return getattr(self, self.get_Iname())

    def set_I(self, _I):
        return setattr(self, self.get_Iname(), _I)

    def get_axes(self):
        return [getattr(self, _axn) for _axn in self._axnames if hasattr(self, _axn)]

    def set_axes(self, axes):
        for ax, _axn in zip(axes, self._axnames):
            if hasattr(self, _axn):
               setattr(self, _axn, ax)
        #self.calc_stdcoord_to_radec()

    """
    Functions setting important attributes
    """
    def set_dpc(self, dpc):
        self.dpc = dpc

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

    def copy_info_from_obsdata(self, obsdata):
        if hasattr(obsdata, "obreso"):
            self.obreso = obsdata.obreso

        if hasattr(obsdata, "sobs_info") and (obsdata.sobs_info is not None):
            self.set_sobs_info(
                iline=obsdata.iline,
                molname=obsdata.molname,
                incl=obsdata.incl,
                phi=obsdata.phi,
                posang=obsdata.posang,
            )
    """
    Functions for saving this object
    """

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
            import joblib
            joblib.dump(self, filepath, compress=3)
            logger.info(f"Saved joblib file: {filepath}")

        elif mode == "pickle":
            pd.to_pickle(self, filepath)
            logger.info(f"Saved pickle file: {filepath}")

        elif mode == "fits":
            save_fits(self, filename)
            logger.info(f"Saved fits file: {filename}")

    """
    Functions operating I and axes
    """
    def norm_I(self, norm="max"):
        I = self.get_I()
        if norm is None:
            pass

        elif norm == "max":
            self.set_I(I/np.max(I))
            #self.Ipv /= np.max(self.Ipv)
            self.Iunit = r"[$I_{\rm max}$]"

        elif isinstance(norm, float):
            self.set_I(I/norm)
            self.Iunit = f"[{norm:.2g}" + r"Jy pixel$^{-1}$]"

        else:
            return None


    def trim(self, xlim=None, ylim=None, vlim=None):
        if xlim is not None:
            nax = self._axorder["xau"]
            imin, imax = minmaxargs(self.xau, xlim)
            self.set_I(np.take(self.get_I(), range(imin,imax), axis=nax))
            self.xau = self.xau[imin:imax]
            #self.calc_stdcoord_to_radec()

        if ylim is not None and hasattr(self, "yau"):
            nax = self._axorder["yau"]
            jmin, jmax = minmaxargs(self.yau, ylim)
            self.set_I(np.take(self.get_I(), range(jmin,jmax), axis=nax))
            self.yau = self.yau[jmin:jmax]
            #self.calc_stdcoord_to_radec()

        if vlim is not None and hasattr(self, "vkms"):
            nax = self._axorder["vkms"]
            kmin, kmax = minmaxargs(self.vkms, vlim)
            self.set_I(np.take(self.get_I(), range(kmin,kmax), axis=nax))
            self.vkms = self.vkms[kmin:kmax]

    def reverse_ax(self, nums): # reverese_ax(1,2)
        if isinstance(nums, int):
            nums = [nums]
        for _num in nums:
            axn = self._axnames[_num]
            ax = getattr(self, axn)
            axrev = ax[::-1]
            setattr(self, axn, axrev)
            I = self.get_I()
            Irev = np.flip(I, axis=_num)
            self.set_I(Irev)

    def reversed_ax_data(self, nums):
        if isinstance(nums, int):
            nums = [nums]
        obj = self.copy()
        for _num in nums:
            axn = obj._axnames[_num]
            ax = getattr(obj, axn)
            axrev = ax[::-1]
            setattr(obj, axn, axrev)
            I = obj.get_I()
            Irev = np.flip(I, axis=_num)
            obj.set_I(Irev)
        return obj

    """
    Functions for conversion
    """
    def convert_Jyppix_to_brightness_temperature(self, imageJyppix, freq0=None, lam0=None):
        if lam0 is not None:
            freq0 = nc.c / lam0
        lam0 = nc.c / freq0
        conv = 2.3504431e-11 * self.dx * self.dy / self.dpc**2 * 1e23
        Inu = (imageJyppix/conv).clip(0.0)
        Tb_RJ = Inu * lam0**2/(2 * nc.kB)
        Tb_app =  nc.h * freq0/nc.kB * np.nan_to_num( 1./np.log(1. + np.nan_to_num( 2.*nc.h*freq0**3/(Inu*nc.c**2) ) ) )
        print(f"Tb_app is {np.max(Tb)} K , Tb is {np.max(Tb)} K, Tb_RJ is {np.max(Tb_RJ)} K")
        if np.isnan(np.max(Tb_app)):
            raise Exception
        return Tb_app

    """
    Functions for observational coordinates

    """

    def ra(self):
        return -self.xau * self.au2pc /(self.dpc * np.cos(self.refpos.dec0*nc.deg2rad)) + self.refpos.ra0

    def dec(self):
        return self.yau * nc.au2pc/self.dpc + self.refpos.dec0

    def radec(self):
        return self.ra(), self.dec()

    def freq(self):
        return tools.vkms_to_freq(self.vkms, self.refpos.freq0)

    def set_coord_from_radec(self, ra0_deg, dec0_deg, ra_deg, dec_deg):
        self.refpos.ra0 = ra0_deg
        self.refpos.dec0 = dec0_deg
        self.yau = self.deg2au(dec_deg - dec0_deg)
        self.xau = -np.cos(dec0_deg*nc.deg2rad) * self.deg2au(ra_deg - ra0_deg)

    def set_coord_from_radecSIN(self, ra0_deg, dec0_deg, ra_deg, dec_deg):
        self.refpos.ra0 = ra0_deg
        self.refpos.dec0 = dec0_deg
        self.yau =  self.deg2au(dec_deg - dec0_deg)
        self.xau = -self.deg2au(ra_deg - ra0_deg)

    def deg2au(self, v_deg):
        return v_deg * 3600 * self.dpc #self.fac_deg2au * self.dpc

    """
    set coorfdinate functions
    To be marged
    """
    def move_position(self, center_pos, ax="x", unit="au"): # ra=False, decl=False, deg=False, au=False):
        deg2au = 3600 * self.dpc

        if unit == "ra":
            d = self.convert_ra_to_deg(*center_pos) * deg2au
        elif unit == "dec":
            d = self.convert_decl_to_deg(*center_pos) * deg2au
        elif unit == "deg":
            d = center_pos * deg2au
        elif unit == "au":
            d = center_pos
        elif unit == "kms":
            d = center_pos
        elif unit == "ms":
            d = center_pos * 1e-3

        if ax == "x":
            self.xau -= d
        elif ax == "y":
            self.yau -= d
        elif ax == "v":
            self.vkms -= d

    """
    def move_center(self, xyau=None, radecdeg=None, v0_kms=None):
        if xyau is not None:
            self.move_position(xyau[0], "x", "au")
            self.move_position(xyau[1], "y", "au")
            self.refpos.dec0 += xyau[1] * nc.au/nc.pc/self.dpc * 180 / np.pi
            self.refpos.ra0 -= xyau[0] * nc.au/nc.pc/self.dpc/np.cos(self.refpos.dec0) * 180 / np.pi

        ""
        elif radecdeg is not None:
            self.dec0 -= np.deg2rad( decdeg[0] )
            self.rad0 -= np.deg2rad( decdeg[1] )
            self.dec -= np.deg2rad( decdeg[0] )
            self.rad -= np.deg2rad( decdeg[1] )
            self.calc_radec_to_stdcoord()
        ""

        if v0_kms is not None:
            self.move_position(v0_kms, "v", "kms")
            #self.vkms -= v0_kms
            # self.freq0 -= *** : freq is not changed here, for now
    """

    def reset_radec_center(self, ra0, dec0):
        self.refpos.ra0 = ra0
        self.refpos.dec = dec0

    def set_center_pos(self, radec_deg=None):
        if radec_deg is not None:
            dxdeg = (radec_deg[0] - self.refpos.ra0) * np.cos(self.refpos.dec0*nc.deg2rad)
            dydeg = radec_deg[1] - self.refpos.dec0
            self.move_position(dxdeg, "x", "deg")
            self.move_position(dydeg, "y", "deg")

    def move_center_pos(self, xy_au=None):
        if xy_au is not None:
            self.move_position(xy_au[0], "x", "au")
            self.move_position(xy_au[1], "y", "au")

    "1. Just change ra0 dec0"
    "2. Change origin of the coordinate but not change the reference point"





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

@dataclasses.dataclass
class RefPos:
    ra0: float = 0
    dec0: float = 0
    freq0: float = 0
    vkms0: float = 0
    def radec(self):
        return np.array([self.ra0, self.dec0])


@dataclasses.dataclass
class Cube(BaseObsData):
    """
    Usually, obsdata is made by doing obsevation.
    But one can generate obsdata by reading fitsfile or radmcdata

    Coordinate:
        A standard coordinate respect to the reference position (RA, Dec, freq0).
        One can generate RA--Dec coordinate by using Cube.ra, Cube.dec, or Cube.radec.
        The refrence position is contained as refpos.
        The origin of coordinate is set to be the same as the FITS file,
        but the origin can be moved arbitrary.
        # Actually I have considered to use, e.g., x in arcsecond as the basic coordinate.
        # But I have not been sure if it is clearer than xau, so I gave up thinking...
        # Also, historically, this class has had the two coordinate, xau and ra.
        # It was very confusing, produced many bugs, and destroyed my confidence...
        # Please tell me your thoughts...
        # For now, I am partially convinced because directly obtained coordinate
        # from synthetic observation to models is physical coordinate rather than
        # observational coordinate. So this choice of the basic coordinate is somewhat reasonable.
    """

    Ippv: np.ndarray
    xau: np.ndarray = None
    yau: np.ndarray = None
    vkms: np.ndarray = None
    dpc: float = None
    obreso: Obreso = None
    sobs_info: dict = None
    Iunit: str = r"Jy pixel$^{-1}$"
    refpos: RefPos = RefPos() #list[float, float, float] = [0,0,0]
    radec_deg: dataclasses.InitVar[tuple] = None
    radecSIN_deg: dataclasses.InitVar[tuple] = None
    freq0: dataclasses.InitVar[float] = None
    ##
    dtype = "Cube"
    _axorder = {"xau":0, "yau": 1, "vkms": 2}
    _Iname = "Ippv"
    _axnames = ["xau", "yau", "vkms"]

    def __post_init__(self, radec_deg, radecSIN_deg, freq0):
        if radec_deg:
            self.set_coord_from_radec(*radec_deg)
        elif radecSIN_deg:
            self.set_coord_from_radecSIN(*radecSIN_deg)
        if freq0:
            self.refpos.freq0 = freq0
        #self._complement_coord()
        self._check_data_shape()
        self._reset_positive_axes(self.Ippv, [self.xau, self.yau, self.vkms])


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

        img = Image(_Ipp, xau=self.xau, yau=self.yau, dpc=self.dpc)
        img.copy_info_from_obsdata(self)

        return img


    def get_pv_map(self, length=[-700, 700], pangle_deg=0, poffset_au=0, norm=None, save=False):
    # norm: None, "max", float
        if self.Ippv.shape[1] > 1:
            Nl = np.sqrt(self.Nx**2 + self.Ny**2)
            posax = np.linspace(length[0], length[1], int(Nl))
            posline = self.position_line(
                posax, PA_deg=pangle_deg, poffset_au=poffset_au
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
            posax = self.xau # ???
            Ipv = self.Ippv[:, 0, :]

        pv = PVmap(Ipv, xau=posax, vkms=self.vkms, dpc=self.dpc, pangle_deg=pangle_deg, poffset_au=poffset_au)
        pv.copy_info_from_obsdata(self)
        if norm is not None:
            pv.norm_I(norm)
        if save:
            pv.save_fitsfile()
        # self.pv_list.append(pv)
        return pv

    def position_line(self, xau, PA_deg, poffset_au=0):
        PA_rad = (PA_deg + 90) * np.pi / 180
        pos_x = xau * np.cos(PA_rad) - poffset_au * np.sin(PA_rad)
        pos_y = xau * np.sin(PA_rad) + poffset_au * np.sin(PA_rad)
        return np.stack([pos_x, pos_y], axis=-1)

@dataclasses.dataclass
class Image(BaseObsData):
    data: np.ndarray
    xau: np.ndarray = None
    yau: np.ndarray = None
    refpos: RefPos = RefPos()
    freq0: float = None
    dpc: float = None
    obreso: Obreso = None
    sobs_info: dict = None
    Iunit: str = r"Jy pix$^{-1}$"
    #fitsfile: str = None
    radec_deg: dataclasses.InitVar[tuple] = None
    radecSIN_deg: dataclasses.InitVar[tuple] = None
    ##
    dtype = "Image"
    _axorder = {"xau":0, "yau": 1}
    _Iname = "data"
    _axnames = ["xau", "yau"]

    def __post_init__(self, radec_deg, radecSIN_deg):
        if radec_deg:
            self.set_coord_from_radec(*radec_deg)
        elif radecSIN_deg:
            self.set_coord_from_radecSIN(*radecSIN_deg)
        self._check_data_shape()
        self._reset_positive_axes(self.data, [self.xau, self.yau])

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
            return  np.array([ res.x[0], res.x[1] ])

        return np.array([ self.xau[ip], self.yau[jp] ])

    def offset_center_to_maximum(self):
        xc, yc = self.get_peak_position(interp=False)
        self.reset_center(xyau=(xc, yc))

    def convert_perbeam_to_perpixel(self): # this could go base
        bmaj = self.obreso.beam_maj_au # = full width half maximum along major axis
        bmin = self.obreso.beam_min_au
        A_beam = np.pi * bmaj * bmin/(4*np.log(2))
        A_pix = self.dx * self.dy
        self.data *= A_pix / A_beam
        self.Iunit = "Jy/pix"


@dataclasses.dataclass
class PVmap(BaseObsData):
    Ipv: np.ndarray
    xau: np.ndarray = None
    vkms: np.ndarray = None
    refpos: RefPos = RefPos()
    dpc: float = None
    " Information for how to have made this PV "
    pangle_deg: float = None
    poffset_au: float = None
    Iunit: str = r"[Jy pixel$^{-1}$]"
    conv_info: dict=None
    obreso: Obreso = None
    freq0: float = None
    xrad: dataclasses.InitVar[float] = None
    ##
    dtype = "PVmap"
    _axorder ={"xau":0, "vkms":1}
    _Iname = "Ipv"
    _axnames = ["xau", "vkms"]

    def __post_init__(
        self,
        xrad
    ):
        if xrad is not None:
            self.xau = xrad * self.dpc * nc.pc / nc.au
        self._check_data_shape()
        self._reset_positive_axes(self.Ipv, [self.xau, self.vkms])


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

    if dtype == "cube":
        obj = Cube(Ippv, xau, yau, vkms, Iunit=Iunit, dpc=dpc, freq0=freq0)

    elif dtype == "image":
        obj = Image(Ipp, xau=xau, yau=yau, Iunit=Iunit, dpc=dpc, freq0=freq0)

    return obj

def read_image_fits(
    filepath, unit1="au", unit2="au", dpc=1, unit1_cm=None, unit2_cm=None
):
    return read_fits(filepath, "image", dpc=dpc, unit1=unit1, unit2=unit2)

def read_pv_fits(
    filepath, unit1="au", unit2="kms", dpc=1, unit1_cm=None, unit2_cms=None, v0_kms=0
):
    return read_fits(filepath, "pv", dpc=dpc, unit1=unit1, unit2=unit2, v0_kms=v0_kms)

def read_cube_fits(
    #filepath, unit1=None, unit2=None, unit3=None, dpc=1, unit1_cm=None, unit2_cm=None, unit3_cms=None, v0_kms=0
    filepath, unit1=None, unit2=None, unit3=None, dpc=1, v0_kms=0
):
    #return read_fits(filepath, "cube", dpc=dpc)
#    return read_fits(filepath, "cube", dpc=dpc)
    return read_fits(filepath, "cube", dpc=dpc, unit1=unit1, unit2=unit2,  unit3=unit3, v0_kms=v0_kms) #, unit1_fac=unit1_cm, unit2_fac=unit2_cm, unit3_fac=unit3_cms)

#!Constracting!

from astropy.nddata import CCDData

def read_fits(filepath, fitstype, dpc, unit1=None, unit2=None, unit3=None, unit1_fac=None, unit2_fac=None, unit3_fac=None, Iunit=None, v0_kms=0):
    logger.info(f"Reading fits file: {filepath}")
    hdul = afits.open(filepath)[0]
    header = hdul.header
    data = hdul.data.T ## .T is due to data shape
#    print(header)
    naxis = header["NAXIS"]

    print("Below is warning in reading fits (see wcs in astropy):")
    print("**** Warning start ****")
    wcs = astropy.wcs.WCS(header, naxis=naxis)
    print("**** Warning end ****")
#    logger.debug("header:\n" + textwrap.fill(str(header), 80) + "\n")
    origin = 0
    centerpix = wcs.wcs.crpix + origin - 1 #- 1 # should be improved into the center pixcoord
    print(f"crpix in fits = {wcs.wcs.crpix}, origin = {origin}, crpix in code {centerpix}")
    axref =  wcs.wcs_pix2world([centerpix], origin)[0]
    print("Taking centerpix as ",centerpix, " gives reference position ",  axref)
    print("This should be equal to CRVAL* in FITS")

    Iunit = Iunit if Iunit is not None else (header["BUNIT"] if "BUNIT" in header else None)
    unit1 = unit1 if unit1 is not None else ( header["CUNIT1"].rstrip() if "CUNIT1" in header else None )
    unit2 = unit2 if unit2 is not None else ( header["CUNIT2"].rstrip() if "CUNIT2" in header else None )
    unit3 = unit3 if unit3 is not None else ( header["CUNIT3"].rstrip() if "CUNIT3" in header else None )

    def _get_ax(axnum, sub=True):
        n = header[f"NAXIS{axnum:d}"]
        pixcoord = np.tile(centerpix, (n,1))
        pixcoord[:,axnum-1] = np.arange(origin, n+origin) # pixcoord in python starts from 0 but that in fits from 1 ??
        ax = wcs.wcs_pix2world(pixcoord, origin)[:,axnum-1]
        if sub:
            axc =  wcs.wcs_pix2world([centerpix], origin)[:,axnum-1]
            return ax - axc
        else:
            return ax


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
        return ndata

    def _lfac_to_cm(unit_name=None, unit_cm=None):
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
        #print(fac)
        return fac

    def _vfac_to_kms(unit_name=None, unit_cms=None):
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
        ax1 = _get_ax(1,sub=0) #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2,sub=0) #* _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        data = _data_shape_convert(data, 2)

        if unit1 == "deg" and unit2 == "deg":
            radec_deg=(axref[0], axref[1], ax1, ax2)
            #obj = Image(data, radec_deg=radec_deg, dpc=dpc, Iunit=Iunit, freq0=freq0)
            obj = Image(data, radecSIN_deg=radec_deg, dpc=dpc, Iunit=Iunit, freq0=freq0)

        elif unit1 == "au" and unit2 == "au":
            obj = Image(data, xau=ax1, yau=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)

    elif fitstype.lower()=="pv":
        ax1 = _get_ax(1,sub=0) #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2,sub=0) * _vfac_to_kms(unit_name=unit2, unit_cms=unit2_fac) #* _get_velscale(unit_name=unit2, unit_cms=unit2_fac)
        ax2 -= v0_kms
        data = _data_shape_convert(data, 2)
        if  unit1 == "deg":
            #radec_deg=(axref[0], axref[1], ax1, ax2)
            #print(radec_deg)
            obj = PVmap(data, xrad=np.deg2rad(ax1),  vkms=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)
        elif unit1 == "au":
            obj = PVmap(data, xau=ax1, vkms=ax2, dpc=dpc, Iunit=Iunit, freq0=freq0)

    elif fitstype.lower()=="cube":
        ax1 = _get_ax(1,sub=0) #* _get_lenscale(unit_name=unit1, unit_cm=unit1_fac)
        ax2 = _get_ax(2,sub=0) #* _get_lenscale(unit_name=unit2, unit_cm=unit2_fac)
        ax3 = _get_ax(3, sub=0) * _vfac_to_kms(unit_name=unit3, unit_cms=unit3_fac)
        ax3 -= v0_kms
        data = _data_shape_convert(data, 3)
        if unit1 == "deg":
            radec_deg=(axref[0], axref[1], ax1, ax2)
            #obj = Cube(data, radec_deg=radec_deg, vkms=ax3, dpc=dpc, Iunit=Iunit, freq0=freq0)
            obj = Cube(data, radecSIN_deg=radec_deg, vkms=ax3, dpc=dpc, Iunit=Iunit, freq0=freq0)
        elif unit1 == "au":
            obj = Cube(data, xau=ax1, yau=ax2, vkms=ax3, dpc=dpc, Iunit=Iunit, freq0=freq0)
    _add_beam_info(obj, dpc)
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

def deg_to_dec(deg):
    _deg = int(deg)
    arcmin = int(60*(deg - _deg))
    arcsec = 3600*(deg - _deg - arcmin/60)
    return _deg, arcmin, arcsec

def ra_to_deg(hour, minu, sec):
    return 15*(hour + minu/60 + sec/(3600))

def dec_to_deg(deg, arcmin, arcsec):
    return deg + arcmin/60 + arcsec/(3600)

#########################################################
# Tools
#########################################################

def minmaxargs(array, lim):
    #print(array, lim)
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
