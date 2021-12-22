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
from dataclasses import dataclass, asdict

import astropy.io.fits as iofits
import astropy.convolution as aconv
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca

import envos.nconst as nc
from envos import tools
from envos import gpath
from envos.log import set_logger
from envos.radmc3d import RadmcController

logger = set_logger(__name__)
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
        # self.view = False
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

    def observe_cont(self, lam, incl=None, phi=None, posang=None):
        incl = incl or self.incl
        phi = phi or self.phi
        posang = phi or self.posang

        logger.info(f"Observing continum with wavelength of {lam} micron")
        zoomau = np.concatenate([self.zoomau_x, self.zoomau_y])
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

        tools.shell(cmd, cwd=self.radmc_dir, error_keyword="ERROR", log_prefix="    ")

        self.data_cont = rmci.readImage(fname=f"{self.radmc_dir}/image.out")
        self.data_cont.freq0 = nc.c / (lam * 1e4)
        odat = Image(radmcdata=self.data_cont, datatype="continum")

        if self.conv:
            odat.Ipp = self.convolver(odat.Ipp)
            odat.add_convolution_info(**self.convolve_config)

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

    def observe_line(self, iline=None, molname=None, incl=None, phi=None, posang=None):
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
            "option": "noscat nostar nodust "
            + self.lineobs_option
            + " ",  # + (" doppcatch " if ,
        }

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
                # exit()

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
        odat = ObsData3D(datatype="line")
        odat.read(radmcdata=self.data)
        odat.add_obs_info(
            iline=iline, molname=molname, incl=incl, phi=phi, posang=posang
        )

        if self.conv:
            odat.Ippv = self.convolver(odat.Ippv)
            odat.add_convolution_info(**self.convolve_config)

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
    image, beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=0, mode="fft"
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
        mode="fft",
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
#        Ippv = np.zeros_like(Ippv)
#        Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1


#############################
#############################
class BaseObsData:
    def __str__(self):
        return tools.dataclass_str(self)

    def set_dpc(self, dpc):
        self.dpc = dpc

    def add_convolution_info(
        self, beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=None
    ):
        self.convolve = True
        self.beam_maj_au = beam_maj_au
        self.beam_min_au = beam_min_au
        self.vreso_kms = vreso_kms
        self.beam_pa_deg = beam_pa_deg

    def add_obs_info(self, iline=None, molname=None, incl=None, phi=None, posang=None):
        self.obsinfo_flag = True
        self.iline = iline
        self.molname = molname
        self.incl = incl
        self.phi = phi
        self.posang = posang

    #    def read_instance(self, filepath):
    #        cls = pd.read_pickle(filepath)
    #        for k, v in cls.__dict__.items():
    #            setattr(self, k, v)

    def save(self, basename="obsdata", mode="pickle", dpc=None, filepath=None):
        if filepath is None:
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
            self._save_fits(filepath, dpc=dpc)

    def _save_joblib(self, filepath):
        import joblib

        joblib.dump(self, filepath, compress=3)
        logger.info(f"Saved joblib file: {filepath}")

    def _save_instance(self, filepath):
        pd.to_pickle(self, filepath)
        logger.info(f"Saved pickle file: {filepath}")

    def _save_fits(self, filepath=None, dpc=None):
        self.data.writeFits(fname=filepath, dpc=dpc)
        logger.info(f"Saved fits file: {filepath}")

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

    def convert_ra_to_deg(self, hour, minu, sec):
        return hour*360/24 + minu*360/24*1/60 + sec*360/24*1/(60*60)

    def convert_decl_to_deg(self, deg, arcmin, arcsec):
        return deg + arcmin/60 + arcsec/(60*60)

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


@dataclass
class ObsData3D(BaseObsData):
    """
    Usually, obsdata is made by doing obsevation.
    But one can generate obsdata by reading fitsfile or radmcdata
    """
    Ippv: np.ndarray = None
    Nx: int = None
    Ny: int = None
    Nv: int = None
    xau: np.ndarray = None
    yau: np.ndarray = None
    vkms: np.ndarray = None
    dx: float = None
    dy: float = None
    dv: float = None
    Lx: float = None
    Ly: float = None
    Lv: float = None
    dpc: float = None
    convolve: bool = False
    beam_maj_au: float = None
    beam_min_au: float = None
    vreso_kms: float = None
    beam_pa_deg: float = None
    datatype: str = None
    obsinfo_flag: bool = False

    def __post_init__(self):
        pass

    def read(
        self,
        fitsfile=None,
        radmcdata=None,
        pklfile=None,
        dpc=None,
    ):
        if fitsfile is not None:
            print("wait...")
            exit(1)
            pass
            #self.read_fits(fitsfile)
            #self = copy.deepcopy( fits.read_cube_fits(fitsfile, dpc) )

        elif radmcdata is not None:
            self.read_radmcdata(radmcdata)

        # elif pklfile is not None:
        #    self.read_instance(pklfile)

        else:
            logger.info("No input.")

    def trim(self, range_x=None, range_y=None, range_v=None):
        if range_x is not None:
            
            #for i, r in enumerate(np.array([self.xau, np.abs(self.xau - range_x[1]) ]).T) :
            #    print(i, r)
            imin, imax = [np.abs(self.xau - xl).argmin() for xl in range_x]
            #print(imin, imax)
            self.Ippv = self.Ippv[imin:imax+1, :, :]
            self.xau = self.xau[imin:imax+1]

        if range_y is not None:
            jmin, jmax = [np.abs(self.yau - yl).argmin() for yl in range_y]
            self.Ippv = self.Ippv[:, jmin:jmax+1, :]
            self.yau = self.yau[jmin:jmax+1]

        if range_v is not None:
            kmin, kmax = [np.abs(self.vkms - vl).argmin() for vl in range_v]
            self.Ippv = self.Ippv[:, :, kmin:kmax+1]
            self.vkms = self.vkms[kmin:kmax+1]

    def get_mom0_map(self, normalize="peak", method="sum", vrange=None):
        if self.Ippv.shape[2] == 1:
            Ipp = self.Ippv[...,0]
        else:    
            Ippv = self.Ippv
            vkms = self.vkms
            if (vrange is not None) and (len(vrange) == 2):
                if vrange[0] < vrange[1]:
                    cond = np.where((vrange[0] < vkms) & (vkms < vrange[1]), True, False)  
                    Ippv = Ippv[..., cond]
                    vkms = vkms[cond]
                else:
                    raise Exception
    
            if method == "sum":
                dv = vkms[1] - vkms[0] # if len(vkms) >= 2 else 1
                Ipp = np.sum(Ippv, axis=-1) * dv
    
            elif method == "integrate":
                Ipp = integrate.simps(Ippv, vkms, axis=-1)
        

        if normalize == "peak":
            Ipp /= np.max(Ipp)

        img = Image(Ipp, self.xau, self.yau, self.dpc)
        img.add_convolution_info(
            self.beam_maj_au,
            self.beam_min_au,
            self.vreso_kms,
            self.beam_pa_deg,
        )
        if self.obsinfo_flag:
            img.add_obs_info(
                iline=self.iline,
                molname=self.molname,
                incl=self.incl,
                phi=self.phi,
                posang=self.posang,
            )

        return img


    def get_pv_map(self, pangle_deg=0, poffset_au=0, Inorm="max", save=False):

        if self.Ippv.shape[1] > 1:
            posline = self.position_line(
                self.xau, PA_deg=pangle_deg, poffset_au=poffset_au
            )
            points = [[(pl[0], pl[1], v) for v in self.vkms] for pl in posline]
            #import pprint
            #pprint.pprint(points )
            #exit()

            if 0:
                Ipv = interpolate.RegularGridInterpolator((self.xau, self.yau, self.vkms),
                      self.Ippv, method='linear', bounds_error=False, fill_value=0)(points)
    
            else:
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

        pv.add_convolution_info(
            self.beam_maj_au,
            self.beam_min_au,
            self.vreso_kms,
            self.beam_pa_deg,
        )
        if self.obsinfo_flag:
            pv.add_obs_info(
                iline=self.iline,
                molname=self.molname,
                incl=self.incl,
                phi=self.phi,
                posang=self.posang,
            )
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

    def read_radmcdata(self, data):
        # if len(data.image.shape) == 2:
        #    self.Ipp = data.image
        # elif len(data.image.shape) == 3 and data.image.shape[2] == 1:
        #    self.Ipp = data.image[:, :, 0]
        # elif len(data.image.shape) == 3:
        self.Ippv = data.image  # .transpose(2, 1, 0)
        self.dpc = data.dpc  # or 100
        self.Nx = data.nx
        self.Ny = data.ny
        self.Nv = data.nfreq
        self.Nf = data.nfreq
        self.xau = data.x / nc.au
        self.yau = data.y / nc.au
        self.freq = data.freq
        self.freq0 = data.freq0
        self.vkms = tools.freq_to_vkms_array(self.freq, self.freq0)
        self.dx = data.sizepix_x / nc.au
        self.dy = data.sizepix_y / nc.au
        self.dv = self.vkms[1] - self.vkms[0]  # if len(self.vkms) > 1 else 0
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.yau[-1] - self.yau[0]
        self.Lv = self.vkms[-1] - self.vkms[0]  # if len(self.vkms) > 1 else 0

    def set(self, image, xau, yau, vkms, dpc):
        self.Ippv = image  # .transpose(2, 1, 0)
        self.dpc = dpc  # or 100

        self.xau = xau
        self.yau = yau
        self.vkms = vkms
        self.Nx = len(self.xau)
        self.Ny = len(self.yau)
        self.Nv = len(self.vkms)

        self.dx = self.xau[1] - self.xau[0]
        self.dy = self.yau[1] - self.yau[0]
        self.dv = self.vkms[1] - self.vkms[0] if len(self.vkms) >= 2 else None 
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.yau[-1] - self.yau[0]
        self.Lv = self.vkms[-1] - self.vkms[0]  # if len(self.vkms) > 1 else 0
        self.reset_positive_axes()

    def check_data_shape(self):
        if self.Ippv.shape != (len(self.xau), len(self.yau), len(self.vkms)):
            logger.info("Data type error")
            raise Exception  

    #def move_center(self, center_pos, ra=False, decl=False, deg=False, au=False):

    def reset_positive_axes(self):
        self.xau, self.dx = self.flip_ax(self.xau, self.dx, 0)
        self.yau, self.dy = self.flip_ax(self.yau, self.dx, 1)
        self.vkms, self.dv = self.flip_ax(self.vkms, self.dv, 2)

    def flip_ax(self, ax, dax, num):
        if (len(ax) >= 2) and (ax[1] < ax[0]):
            ax = ax[::-1]
            dax = ax[1] - ax[0]
            np.flip(self.Ippv, num)
        return ax, dax        


@dataclass
class Image(BaseObsData):
    Ipp: np.ndarray = None
    xau: np.ndarray = None
    yau: np.ndarray = None
    dpc: float = None
    unit_I: str = r"[Jy km s$^{-1}$ pixel$^{-1}$]"
    beam_maj_au: float = None
    beam_min_au: float = None
    vreso_kms: float = None
    beam_pa_deg: float = None
    fitsfile: str = None
    def read_radmcdata(self, data):
        if len(data.image.shape) == 2:
            self.Ipp = data.image
        elif len(data.image.shape) == 3 and data.image.shape[2] == 1:
            self.Ipp = data.image[:, :, 0]
        elif len(data.image.shape) == 3:
            self.Ippv = data.image  # .transpose(2, 1, 0)
        self.dpc = data.dpc or 100
        self.Nx = data.nx
        self.Ny = data.ny
        self.Nv = data.nfreq
        self.Nf = data.nfreq
        self.xau = data.x / nc.au
        self.yau = data.y / nc.au
        self.freq = data.freq
        self.freq0 = data.freq0
        self.vkms = tools.freq_to_vkms_array(self.freq, self.freq0)
        self.dx = data.sizepix_x / nc.au
        self.dy = data.sizepix_y / nc.au
        self.Lx = self.xau[-1] - self.xau[0]
        self.Ly = self.yau[-1] - self.yau[0]
        self.dv = self.vkms[1] - self.vkms[0] if len(self.vkms) > 1 else 0
        self.Lv = self.vkms[-1] - self.vkms[0] if len(self.vkms) > 1 else 0

    def check_data_shape(self):
        if self.Ipp.shape != (len(self.xau), len(self.yau)):
            logger.info("Data type error")
            raise Exception 

    def get_peak_position(self, interp=True):
        import skimage
        from scipy import optimize, interpolate
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

@dataclass
class PVmap(BaseObsData):
    Ipv: np.ndarray = None
    xau: np.ndarray = None
    vkms: np.ndarray = None
    dpc: float = None
    pangle_deg: float = None
    poffset_au: float = None
    unit_I: str = r"[Jy pixel$^{-1}$]"
    beam_maj_au: float = None
    beam_min_au: float = None
    vreso_kms: float = None
    beam_pa_deg: float = None
    fitsfile: str = None

    def __post_init__(
        self,
    ):
        pass
        # if self.fitsfile is not None:
        #    self.read_fitsfile(self.fitsfile)

    def normalize(self, Ipv_norm=None):
        if Ipv_norm is None:
            self.Ipv /= 1
            self.unit_I = r"[Jy pixel$^{{-1}}$]"  # rf"[{Ipv_norm} Jy pixel$^{{-1}}$]]"
        elif Ipv_norm == "max":
            self.Ipv /= np.max(self.Ipv)
            self.unit_I = r"[$I_{\rm max}$]"
        elif isinstance(Ipv_norm, float):
            self.Ipv /= Ipv_norm
            self.unit_I = f"[{Ipv_norm}" + r"Jy pixel$^{-1}$]"
        else:
            return None

    def reverse_x(self):
        self.Ipv = self.Ipv[:, ::-1]
        self.xau = -self.xau[::-1]

    def reverse_v(self):
        self.Ipv = self.Ipv[::-1, :]
        self.vkms = -self.vkms[::-1]

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

    def save_fitsfile(self, filename="pvimage.fits", filepath=None):
        """
        save the obsdata as a fitsfile
        see IAU manual for variables used in fits:
             https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
        """
        if filepath is None:
            filepath = os.path.join(gpath.run_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        Np = len(self.xau)
        Nv = len(self.vkms)
        dx = self.xau[1] - self.xau[0]
        dv = self.vkms[1] - self.vkms[0]
        hdu = iofits.PrimaryHDU(self.Ipv)
        # x = x0(crval) + dx(cdelt) * (i - i0(crpix))
        # i = 1, 2, 3 ...
        new_header = {
            "NAXIS": 2,
            "CTYPE1": "ANGLE",
            "NAXIS1": Np,
            "CRVAL1": 0.0,
            "CRPIX1": (Np + 1) / 2,
            "CDELT1": np.radians(dx * nc.au / self.dpc),
            "CUNIT1": "deg",
            "CTYPE2": "VRAD",
            "NAXIS2": Nv,
            "CRVAL2": 0.0,
            "CRPIX2": (Nv + 1) / 2,
            "CDELT2": dv * 1e3,
            "CUNIT2": "m/s",
            "BTYPE": "Intensity",
            "BUNIT": "Jy/pixel",
        }
        if self.beam_maj_au:
            new_header.update(
                {
                    "BMAJ": self.beam_maj_au,
                    "BMIN": self.beam_min_au,
                    "BPA": self.beam_pa_deg,
                }
            )
        hdu.header.update(new_header)
        hdulist = iofits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)

    def check_data_shape(self):
        if self.Ipv.shape != (len(self.xau), len(self.vkms)):
            logger.info("Data type error")
            raise Exception



# Readers
def read_obsdata(path, mode=None):
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
        return ObsData3D(filepath=path)


def read_image_fits(
    filepath, unit1="au", unit2="au", dpc=1, unit1_cm=None, unit2_cm=None
):
    fr = FitsReader(filepath)
    fr.read_image(unit1=unit1, unit2=unit2, dpc=dpc, unit1_cm=unit1_cm, unit2_cm=unit2_cm)
    return fr.image


def read_pv_fits(
    filepath, unit1="au", unit2="kms", dpc=1, unit1_cm=None, unit2_cms=None, v0_kms=0
):
    fr = FitsReader(filepath)
    fr.read_pv(unit1=unit1, unit2=unit2, dpc=dpc, unit1_cm=unit1_cm, unit2_cms=unit2_cms, v0_kms=v0_kms)
    return fr.pv


def read_cube_fits(filepath, unit1="au", unit2="au", unit3="kms", dpc=1, unit1_cm=None, unit2_cm=None, unit3_cms=None, v0_kms=0):
    fr = FitsReader(filepath)
    fr.read_cube(unit1=unit1, unit2=unit2, unit3=unit3, dpc=dpc, unit1_cm=unit1_cm, unit2_cm=unit2_cm, unit3_cms=unit3_cms, v0_kms=v0_kms)
    logger.info(fr.cube)
    #exit()
    return fr.cube


class FitsReader:
    def __init__(self, filepath):
        self.path = filepath
        self.image = None
        self.header = None
        self.naxis = None
        self.image = None
        self.pv = None
        self.cube = None
        self.load()

    def load(self):
        logger.info(f"Reading fits file: {self.path}")
        pic = iofits.open(self.path)[0]
        if pic.data.shape[0] == 1:
            self.image = pic.data[0]
        else:
            self.image = pic.data

        self.header = pic.header
        self.naxis = self.header[f"NAXIS"]
        logger.debug("header:")
        logger.debug(textwrap.fill(str(self.header), 80) + "\n")

        #print(x, wcs.wcs_pix2world([[0,0,0,0], [1,0,0,0]], 0) )

#        #a = wcs.sub(2)
#        #print(a, np.array(a))
#        x = np.arange(512)#np.linspace(1, 512, 512)
#        y = np.arange(1) #np.linspace(1, 512, 512)
#        z = np.arange(1)#np.linspace(1, 66, 66)
#        w = np.arange(1) #np.linspace(1, 1, 1)
#        xx, yy, zz, ww = np.meshgrid(x, y, z, w)
#        a = wcs.wcs_pix2world(xx, yy, zz, ww, 0)
#        print(np.shape(a))
#        print(np.array(a)[0,0,:,0, 0], np.array(a)[1,0,:,0, 0], np.array(a)[2,0,:,0, 0],)
#        exit()

#        #pixcrd = np.array([[0, 0, 0, 0], [24, 38, 100, 100], [45, 98, 100, 100]], dtype=np.float64)
#        a = np.linspace(0, 100, 101)
#        x = np.arange(512)#np.linspace(1, 512, 512)
#        y = np.arange(512) #np.linspace(1, 512, 512)
#        z = np.arange(66)#np.linspace(1, 66, 66)
#        w = np.arange(1) #np.linspace(1, 1, 1)
#        xx, yy, zz, ww = np.meshgrid(x, y, z, w)
#        pixargs = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), ww.reshape(-1) ]).T
#        print(pixargs, pixargs.shape)
#        a, b, c, d = wcs.wcs_pix2world(xx, yy, zz, ww, 0)
#        print(a[:,0,0,0])
#        #a, b, c, d = wcs.all_pix2world(pixargs, 0).T
#        #print(a.shape, b.shape, c.shape)
#
#        exit()


        #pixcrd = np.array([list(a) ]*4).T
        #print(np.shape(pixcrd))
        #world = wcs.wcs_pix2world(pixcrd, 0)
#        world =  wcs.pixel_to_world(   )   
#        print(world)
#        exit()
#        import astropy.coordinates
#        from astropy.coordinates import SkyCoord
#
#        maxind = np.unravel_index(np.nanargmax(pic.data), pic.data.shape)
#        maxworld = wcs.wcs_pix2world([maxind[::-1]], 0) 
#        print(maxworld)
#        galx, galy = maxworld[0][:2]
#        coord = astropy.coordinates.SkyCoord(galx, galy, frame='galactic', unit='deg')
#        print(coord)
#
#        from astropy.coordinates import SkyCoord
#        coord = SkyCoord('00h00m00s +00d00m00s', frame='galactic')
#        pixels = wcs.world_to_pixel(coord)  
#        print(pixels)
#
#        exit()

    def data_shape_convert(self, data, ndim, transpose=True):
        if len(np.shape(data)) != ndim:
            nshape = [i for i in np.shape(data) if i!=1]
            if len(nshape) == ndim:
                ndata = np.reshape(data, nshape)
            else:            
                logger.error(f"Wait, something wrong in the data! nshape = {nshape}")
                raise Exception
                exit(1)
        else:
            ndata = data
        if transpose:
            ndata = ndata.T

        #elif len(data.shape) == 3:
        #    ndata = data

        #else:
        #    logger.info("Data type looks weired.")
        #    sys.exit(1)


        return ndata


    def read_image(
        self, unit1="au", unit2="au", dpc=1, unit1_cm=None, unit2_cm=None
    ):
        header = self.header
        self.dpc = dpc
        xau = self.get_ax(1)
        xau *= self.get_length_scale(unit_name=unit1, unit_cm=unit1_cm)
        yau = self.get_ax(2)
        yau *= self.get_length_scale(unit_name=unit2, unit_cm=unit2_cm)

        self.image = self.data_shape_convert(self.image, 2)
        self.image, xau, yau = self.set_axes_positive(self.image, (xau, yau))
        self.image = Image(self.image, xau, yau, dpc)
        self.image.check_data_shape()
        self.add_beam_info(self.image, dpc)

    def read_pv(
        self, unit1="au", unit2="kms", dpc=1, unit1_cm=None, unit2_cms=None, v0_kms=0
    ):
        header = self.header
        self.dpc = dpc
        xau = self.get_ax(1)
        xau *= self.get_length_scale(unit_name=unit1, unit_cm=unit1_cm)
        vkms = self.get_ax(2) 
        vkms *= self.get_velocity_scale(unit_name=unit2, unit_cms=unit2_cms)
        vkms -= v0_kms
        self.image = self.data_shape_convert(self.image, 2) # transpose=True)
        self.image, xau, vkms = self.set_axes_positive(self.image, (xau, vkms))

        self.pv = PVmap(self.image, xau, vkms, dpc)
        self.pv.check_data_shape()
        self.add_beam_info(self.pv, dpc, vkms)


    def read_cube(self, unit1="au", unit2="au", unit3="kms", dpc=1, unit1_cm=None, unit2_cm=None, unit3_cms=None, v0_kms=0, center=None):
        header = self.header
        self.dpc = dpc
        xau = self.get_ax(1)
        xau *= self.get_length_scale(unit_name=unit1, unit_cm=unit1_cm)
        yau = self.get_ax(2)
        yau *= self.get_length_scale(unit_name=unit2, unit_cm=unit2_cm)
        #yau = self.slide_to_center(yau)
        vkms = self.get_ax(3)
        vkms *= self.get_velocity_scale(unit_name=unit3, unit_cms=unit3_cms)
        vkms -= v0_kms
        self.cube = ObsData3D()
        self.image = self.data_shape_convert(self.image, 3)
        self.image, xau, yau, vkms = self.set_axes_positive(self.image, (xau, yau, vkms))
        #exit()
        self.cube.set(self.image, xau, yau, vkms, dpc)
        self.cube.check_data_shape()
        self.add_beam_info(self.cube, dpc, vkms)

    def add_beam_info(self, obj, dpc, vkms=None):
        if "BMAJ" in self.header:
            beam_maj_au = self.header["BMAJ"] * dpc
            beam_min_au = self.header["BMIN"] * dpc
            beam_pa_deg = self.header["BPA"]
            if (vkms is not None) and (len(vkms) >= 2):
                vreso_kms =  vkms[1] - vkms[0]
            else:
                vreso_kms = None
            obj.add_convolution_info(
                beam_maj_au,
                beam_min_au,
                vreso_kms,
                beam_pa_deg,
            )

    def get_ax(self, axnum): 
        import astropy.io.fits
        wcs = astropy.wcs.WCS(self.header, naxis=self.naxis)
        n = self.header[f"NAXIS{axnum:d}"]
        x = np.arange(n)
        zeros = np.zeros((self.naxis-1, n)) 
        ind = np.insert(zeros, axnum-1, x, axis=0)
        ret = wcs.wcs_pix2world(ind.T, 0)
        return ret[:, axnum-1]

    def get_axis(self, axis_number, neglect_crval=False, ra=False):
        naxis = self.header[f"NAXIS{axis_number:d}"]
        crval = self.header[f"CRVAL{axis_number:d}"] if not neglect_crval else 0
        crpix = self.header[f"CRPIX{axis_number:d}"]
        cdelt = self.header[f"CDELT{axis_number:d}"]
        #if ra:
        #    cdelt /=  
        ax = crval + cdelt * (np.arange(naxis) + 1 - crpix)
        #  if axis_number == 1:
        #      deg = 4*360/24 + 39*360/24*1/60 + 53.89*360/24*1/60*1/60
        #      print(ax- deg)
        #      ax -= deg
        #  if axis_number == 2:
        #      deg = 26 + 3*1/60 + 9.8/60/60
        #      print(ax-deg)
        #      ax -= deg
        
        #ax = cdelt * (np.arange(naxis) + 1 - crpix)
        #print(axis_number, ax, naxis, crval, crpix, cdelt)
        #exit(1)
        return ax # crval + cdelt * (np.arange(naxis) + 1 - crpix)

    def set_axes_positive(self, img, axes):
        new_axes = []
        for i, _ax in enumerate(axes):
            if (len(_ax) >= 2) and (_ax[1] < _ax[0]):
                _ax = _ax[::-1]
                img = np.flip(img, i)
            new_axes.append(_ax)
        return img, *new_axes 

    def convert_ra_to_deg(self, hour, minu, sec):
        return 15*(hour + minu/60 + sec/(3600))

    def convert_decl_to_deg(self, deg, arcmin, arcsec):
        return deg + arcmin/60 + arcsec/(3600)

    def get_length_scale(self, unit_name=None, unit_cm=None):
        if unit_cm is not None:
            logger.info(
                "   1st axis is interpreted as POSITION "
                f"with user-defined unit (unit = {unit_cm} cm)."
            )
            fac = unit_cm / nc.au

        elif unit_name in ["degree", "deg"]:
            #print("Use deg")
            fac = 3600 * self.dpc

        elif unit_name in ["au"]:
            fac = 1.0

        else:
            raise Exception("Unknown datatype in 1st axis")

        return fac

    def get_velocity_scale(self, unit_name=None, unit_cms=None):
        if unit_cms:
            logger.info(
                "   **nd axis is interpreted as VELOCITY "
                f"with user-defined unit (unit = {unit_cms} cm/s)."
            )
            fac = unit_cms / 1e5

        elif unit_name in ["ms", "m/s"]:
            fac = 1e-3

        elif unit_name in ["kms", "km/s"]:
            fac = 1.0

        elif unit_name in ["Hz"]:  # when dnu is in Hz
            nu0 = self.header["RESTFRQ"]  # freq: max --> min
            fac = nc.c / 1e5 / nu0
            logger.info(
                f"Here nu0 is supposed to be {nu0/1e9} GHz. If it is wrong, please let S.M know, because it should be a bug."
            )

        else:
            raise Exception("Unknown datatype in 2nd axis")

        return fac


    def slide_to_center(self, position_ax):
        return position_ax - 0.5*(position_ax[-1] + position_ax[0])



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

