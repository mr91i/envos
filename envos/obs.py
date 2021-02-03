#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import copy
import logging
import contextlib
import multiprocessing
from scipy import integrate, interpolate

import astropy.io.fits as iofits
import astropy.convolution as aconv
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca

import envos.nconst as nc
from envos import tools
import envos.global_paths as gpath
from envos.log import set_logger

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
    vw_kms=None,
    nlam=None,
    option="",
):
    position = f"dpc {dpc} incl {incl} phi {phi} posang {posang}"
    camera = f"npixx {npixx} npixy {npixy} "
    camera += f"zoomau {zoomau[0]} {zoomau[1]} {zoomau[2]} {zoomau[3]}"
    if lam is not None:
        freq = f"lambda {lam}"
    elif iline is not None:
        freq = f"iline {iline} widthkms {vw_kms:g} linenlam {nlam:d}"
        freq += f" vkms {vc_kms:g}" if vc_kms else ""
    cmd = " ".join(["radmc3d", f"{mode}", position, camera, freq, option])
    return cmd


class ObsSimulator:
    """
    This class returns observation data
    Basically, 1 object per 1 observation
    """

    def __init__(
        self, config=None, radmcdir=None, dpc=None, omp=True, nthread=1
    ):

        self.radmc_dir = radmcdir or gpath.radmc_dir
        self.dpc = dpc
        self.omp = omp
        self.nthread = nthread
        self.view = False
        self.conv = False
        self.incl = None
        self.phi = None
        self.posang = None

        if config is not None:
            oconf = config.obs
            self.dpc = oconf.dpc
            self.omp = oconf.omp
            self.nthread = oconf.nthread
            self.incl = oconf.incl
            self.phi = oconf.phi
            self.posang = oconf.posang
            self.iline = oconf.iline
            self.molname = oconf.molname

            self.set_resolution(
                sizex_au=oconf.sizex_au,
                sizey_au=oconf.sizey_au,
                pixsize_au=oconf.pixsize_au,
                npix=oconf.npix,
                npixx=oconf.npixx,
                npixy=oconf.npixy,
                vwidth_kms=oconf.vwidth_kms,
                dv_kms=oconf.dv_kms,
            )

            self.set_convolver(
                beam_maj_au=oconf.beam_maj_au,
                beam_min_au=oconf.beam_min_au,
                vreso_kms=oconf.vreso_kms,
                beam_pa_deg=oconf.beam_pa_deg,
                convmode=oconf.convmode,
            )

    def set_resolution(
        self,
        sizex_au,
        sizey_au=None,
        pixsize_au=None,
        npix=None,
        npixx=None,
        npixy=None,
        vwidth_kms=None,
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
            self.npixx = npixx or npix
            self.npixy = npixy or npix
            self.pixsize_au = Lx / self.npixx

        self.dx_au = sizex_au / self.npixx
        self.dy_au = _sizey_au / self.npixy

        if self.dx_au != self.dy_au:
            raise Exception("dx is not equal to dy. Check your setting again.")

        if sizex_au == 0:
            self.zoomau_x = [-self.pixsize_au / 2, self.pixsize_au / 2]
        if sizey_au == 0:
            self.zoomau_y = [-self.pixsize_au / 2, self.pixsize_au / 2]

        self.vwidth_kms = vwidth_kms
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
        self.convolver = Convolver(
            (beam_maj_au, beam_min_au, vreso_kms),
            (self.dx_au, self.dy_au, self.dv_kms),
            beam_pa_deg=beam_pa_deg,
            mode=convmode,
        )

    def observe_cont(self, lam, incl=0, phi=0, posang=0):
        if self.incl is not None:
            incl = self.incl
        if self.phi is not None:
            phi = self.phi
        if self.posang is not None:
            posang = self.posang

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

        tools.shell(
            cmd, cwd=self.radmc_dir, error_keyword="ERROR", log_prefix="    "
        )

        self.data_cont = rmci.readImage(fname=f"{self.radmc_dir}/image.out")

        self.data_cont.freq0 = nc.c / (lam * 1e4)
        convolve = False
        if self.conv:
            # self.data_cont.conv_image = self.convolver(self.data_cont.image)
            self.data_cont.image = self.convolver(self.data_cont.image)
            convolve = True

        odat = ObsData(
            radmcdata=self.data_cont, datatype="continum", convolve=convolve
        )
        return odat

    def observe_line(self, iline=None, molname=None, incl=0, phi=0, posang=0):
        if self.iline is not None:
            iline = self.iline
        if self.molname is not None:
            molname = self.molname
        if self.incl is not None:
            incl = self.incl
        if self.phi is not None:
            phi = self.phi
        if self.posang is not None:
            posang = self.posang

        logger.info(f"Observing line with {molname}")
        if self.omp and (self.nthread > 1):
            logger.info(
                f"Use OpenMP dividing processes into velocity directions."
            )
            logger.info(f"Number of threads is {self.nthread}.")
        else:
            logger.info(f"Not use OpenMP.")

        # self.obs_mode = "line"
        self.iline = iline
        # self.vwidth_kms = vwidth_kms
        # self.dv_kms = dv_kms
        self.nlam = int(round(2 * self.vwidth_kms / self.dv_kms)) + 1
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
            "option": "noscat nostar nodust",
        }

        v_calc_points = np.linspace(
            -self.vwidth_kms, self.vwidth_kms, self.nlam
        )

        if self.omp and (self.nthread > 1):
            # Load-balance for multiple processing
            n_points = self._divide_nlam_by_threads(self.nlam, self.nthread)
            vrange_list = self._calc_vrange_list(v_calc_points, n_points)
            v_center = [0.5 * (vmax + vmin) for vmin, vmax in vrange_list]
            v_width = [0.5 * (vmax - vmin) for vmin, vmax in vrange_list]

            logger.info(f"All calc points: {format_array(v_calc_points)}")
            logger.info("Calc points in each thread:")
            _zipped = zip(v_center, v_width, n_points)
            for i, (vc, vw, ncp) in enumerate(_zipped):
                vax_nthread = np.linspace(vc - vw, vc + vw, ncp)
                logger.info(f" -- {i}th thread: {format_array(vax_nthread)}")

            def cmdfunc(i):
                return gen_radmc_cmd(
                    vc_kms=v_center[i],
                    vw_kms=v_width[i],
                    nlam=n_points[i],
                    **common_cmd,
                )

            args = [(i, cmdfunc(i)) for i in range(self.nthread)]

            with multiprocessing.Pool(self.nthread) as pool:
                results = pool.starmap(self._subcalc, args)

            self._check_multiple_returns(results)
            self.data = self._combine_multiple_returns(results)

        else:
            cmd = gen_radmc_cmd(
                vw_kms=self.vwidth_kms, nlam=self.nlam, **common_cmd
            )

            tools.shell(cmd, cwd=self.radmc_dir)
            self.data = rmci.readImage(fname=f"{self.radmc_dir}/image.out")

        if np.max(self.data.image) == 0:
            print(vars(self.data))
            logger.warning("Zero image !")
            raise Exception

        #        self.data.dpc = self.dpc
        self.data.freq0 = self.mol.freq[iline - 1]
        if self.conv:
            self.data.image = self.convolver(self.data.image)
        return ObsData(radmcdata=self.data, datatype="line")

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
        # logger.info("execute: " + cmd)
        dpath_sub = f"{self.radmc_dir}/{dn}"
        os.makedirs(dpath_sub, exist_ok=True)
        os.system(f"cp {self.radmc_dir}/{{*.inp,*.dat}} {dpath_sub}/")
        log = logger.isEnabledFor(logging.INFO) and p == 0
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
            data.imageJyppix = np.append(
                data.imageJyppix, ret.imageJyppix, axis=-1
            )
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
    delta = abs(array[1] - array[0])
    msg = f"[{min(array):.2f}:{max(array):.2f}] "
    msg += f"with delta = {delta:.4g} and N = {len(array)}"
    return msg


class Convolver:
    def __init__(self, conv_size, grid_size, beam_pa_deg=0, mode="fft"):
        # relation : standard deviation = 1/(2 sqrt(ln(2))) * FWHM of Gaussian
        # theta_deg : cclw is positive
        self.mode = mode
        sigma_over_FWHM = 2 * np.sqrt(2 * np.log(2))
        stddev = np.array(conv_size) / np.array(grid_size) / sigma_over_FWHM
        beampa = np.radians(beam_pa_deg)
        self.Kernel_xy2d = aconv.Gaussian2DKernel(
            stddev[0], stddev[1], beampa
        )._array
        if len(conv_size) == 3 and (conv_size[2] is not None):
            Kernel_v1d = aconv.Gaussian1DKernel(stddev[2])._array
            self.Kernel_3d = np.multiply(
                self.Kernel_xy2d[np.newaxis, :, :],
                Kernel_v1d[:, np.newaxis, np.newaxis],
            )

    def __call__(self, image):
        if len(image.shape) == 2:
            Kernel = self.Kernel_xy2d
        elif len(image.shape) == 3:
            Kernel = self.Kernel_3d
        else:
            raise Exception("Unknown data.image shape: ", image.shape)

        if self.mode == "normal":
            return aconv.convolve(image, Kernel)
        elif self.mode == "fft":
            return aconv.convolve_fft(image, Kernel, allow_huge=True)
        elif self.mode == "null":
            return image
        else:
            raise Exception("Unknown convolve mode: ", self.mode)


#    if pointsource_test:
#        Ippv = np.zeros_like(Ippv)
#        Ippv[Ippv.shape[0]//2, Ippv.shape[1]//2, Ippv.shape[2]//2] = 1


#############################
#############################


class ObsData:
    """
    Usually, obsdata is made by doing obsevation.
    But one can generate obsdata by reading fitsfile or radmcdata
    """

    def __init__(
        self,
        fitsfile=None,
        radmcdata=None,
        pklfile=None,
        datatype=None,
        dpc=None,
        convolve=False,
    ):
        self.datatype = datatype
        self.Ippv = None
        self.Ippv_raw = None
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
        self.dpc = dpc

        self.convolve = convolve
        self.beam_maj_au = None
        self.beam_min_au = None
        self.vreso_kms = None
        self.beam_pa_deg = None
        self.convolver = None
        self.PV_list = []

        if fitsfile is not None:
            self.read_fits(fitsfile)
        elif radmcdata is not None:
            self.read_radmcdata(radmcdata)
        elif pklfile is not None:
            self.read_instance(pklfile)
        else:
            logger.info("No input.")

    def read_fits(self, file_path, dpc):
        pic = iofits.open(file_path)[0]
        self.Ippv_raw = pic.data
        self.Ippv = copy.copy(self.Ippv_raw)
        header = pic.header

        self.Nx = header["NAXIS1"]
        self.Ny = header["NAXIS2"]
        self.Nz = header["NAXIS3"]
        self.dx = -header["CDELT1"] * np.pi / 180.0 * self.dpc * nc.pc / nc.au
        self.dy = +header["CDELT2"] * np.pi / 180.0 * self.dpc * nc.pc / nc.au
        self.Lx = self.Nx * self.dx
        self.Ly = self.Ny * self.dy
        self.xau = -0.5 * self.Lx + (np.arange(self.Nx) + 0.5) * self.dx
        self.yau = -0.5 * self.Ly + (np.arange(self.Ny) + 0.5) * self.dy

        if header["CRVAL3"] > 1e8:  # when dnu is in Hz
            nu_max = header["CRVAL3"]  # freq: max --> min
            dnu = header["CDELT3"]
            nu0 = nu_max + 0.5 * dnu * (self.Nz - 1)
            self.dv = -nc.c / 1e5 * dnu / nu0
        else:
            self.dv = header["CDELT3"] / 1e3
        self.vkms = self.dv * (-0.5 * (self.Nz - 1) + np.arange(self.Nz))

        if (self.dx < 0) or (self.xau[1] < self.xau[0]):
            raise Exception("Step in x-axis is negative")

        if (self.dy < 0) or (self.yau[1] < self.yau[0]):
            raise Exception("Step in y-axis is negative")

        if (self.dv < 0) or (self.vkms[1] < self.vkms[0]):
            raise Exception("Step in x-axis is negative")

        logger.info(f"fits file path: {file_path}")
        logger.info(f"pixel size[au]: {self.dx}  {self.dy}")
        logger.info(f"L[au]: {self.Lx} {self.Ly}")

    def read_radmcdata(self, data):
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

    def set_dpc(self, dpc):
        self.dpc = dpc

    def make_mom0_map(self, normalize="peak"):
        Ipp = integrate.simps(self.Ippv, axis=0)
        if normalize == "peak":
            Ipp /= np.max(Ipp)
        self.Imom0 = Ipp

    def position_line(self, xau, PA_deg, poffset_au=0):
        PA_rad = PA_deg * np.pi / 180
        pos_x = xau * np.cos(PA_rad) - poffset_au * np.sin(PA_rad)
        pos_y = xau * np.sin(PA_rad) + poffset_au * np.sin(PA_rad)
        return np.stack([pos_x, pos_y], axis=-1)

    def make_PV_map(self, pangle_deg=0, poffset_au=0):
        if self.Ippv.shape[1] > 1:
            posline = self.position_line(
                self.xau, PA_deg=pangle_deg, poffset_au=poffset_au
            )
            #points = [[(v, pl[1], pl[0]) for pl in posline] for v in self.vkms]
            points = [[(pl[0], pl[1], v) for pl in posline] for v in self.vkms]
            Ipv = interpolate.interpn(
                #(self.vkms, self.yau, self.xau),
                (self.xau, self.yau, self.vkms),
                self.Ippv,
                points,
                bounds_error=False,
                fill_value=0,
            )
        else:
            Ipv = self.Ippv[:, 0, :]

        PV = PVmap(Ipv, self.xau, self.vkms, self.dpc, pangle_deg, poffset_au)
        PV.add_conv_info(
            self.beam_maj_au,
            self.beam_min_au,
            self.vreso_kms,
            self.beam_pa_deg,
        )
        PV.normalize()
        PV.save_fitsfile()
        self.PV_list.append(PV)
        return PV

    def save_instance(self, filename="obsdata.pkl", filepath=None):
        if filepath is None:
            filepath = os.path.join(gpath.run_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.to_pickle(self, filepath)

    def save_fits(self, filename="obsdata.fits", dpc=None, filepath=None):
        if filepath is None:
            filepath = os.path.join(gpath.run_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            logger.info(f"remove old fits file: {filepath}")
            os.remove(filepath)

        self.data.writeFits(fname=filepath, dpc=dpc)
        logger.info(f"Saved fits file: {filepath}")


class PVmap:
    def __init__(
        self,
        Ipv=None,
        xau=None,
        vkms=None,
        dpc=None,
        pangle_deg=None,
        poffset_au=None,
        fitsfile=None,
    ):
        self.Ipv = Ipv
        self.Ipv_raw = Ipv
        self.xau = xau
        self.vkms = vkms
        self.dpc = dpc
        self.pangle_deg = pangle_deg
        self.poffset_au = poffset_au
        self.unit_I = r"[Jy pixel$^{-1}$]"
        self.beam_maj_au = None
        self.beam_min_au = None
        self.vreso_kms = None
        self.beam_pa_deg = None

        if fitsfile is not None:
            self.read_fits_PV(fitsfile)

    def add_conv_info(self, beam_maj_au, beam_min_au, vreso_kms, beam_pa_deg):
        self.beam_maj_au = beam_maj_au
        self.beam_min_au = beam_min_au
        self.vreso_kms = vreso_kms
        self.beam_pa_deg = beam_pa_deg

    def normalize(self, Ipv_norm=None):
        if Ipv_norm is None:
            self.Ipv /= np.max(self.Ipv)
            self.unit_I = r"[$I_{\rm max}$]"
        else:
            self.Ipv /= Ipv_norm
            self.unit_I = rf"[{Ipv_norm} Jy pixel$^{{-1}}$]]"

    def trim(self, range_x=None, range_v=None):
        if range_x is not None:
            imax, imin = [np.abs(self.xau - xl).argmin() for xl in range_x]
            self.Ipv = self.Ipv[:, imin:imax]
            self.xau = self.xau[imin:imax]

        if range_v is not None:
            jmin, jmax = [np.abs(self.vkms - vl).argmin() for vl in range_v]
            self.Ipv = self.Ipv[jmin:jmax, :]
            self.vkms = self.vkms[jmin:jmax]

    def save_fitsfile(self, filename="PVimage.fits", filepath=None):
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

    def read_fitsfile(
        self, filepath, unit1_in_au=None, unit2_in_kms=None, unit1="", unit2=""
    ):
        logger.info(f"Reading fits file: {filepath}")
        pic = iofits.open(filepath)[0]
        self.Ipv = pic.data
        header = pic.header

        self.Nx = header["NAXIS1"]

        if unit1_in_au:
            logger.info(
                "   1st axis is interpreted as POSITION "
                f"with user-defined unit (unit1 = {unit1_in_au} au)."
            )
            dx_au = header["CDELT1"] * unit1_in_au

        elif "ANGLE" in header["CYTYPE"] and unit1 == "deg":  # x in deg
            logger.info("   1st axis is interpreted as POSITION [deg].")
            dx_au = header["CDELT1"] * 3600 * self.dpc  # dx in au
        else:
            raise Exception("Unknown datatype in 1st axis")

        self.xau = (np.arange(self.Nx) - 0.5 * (self.Nx - 1)) * dx_au

        self.Nv = header["NAXIS2"]
        if unit2_in_kms:
            logger.info(
                "    2nd axis is interpreted as VELOCITY "
                f"with user-defined unit (unit2 = {unit2_in_kms} km/s)."
            )
            dv_kms = header["CDELT2"] * unit2_in_kms

        elif "VRAD" in header["CYTYPE"] and unit2 == "m/s":  # v in m/s
            logger.info("    2nd axis is interpreted as VELOCITY [m/s].")
            dv_kms = header["CDELT2"] / 1e3  # in m/s to in km/s

        else:
            raise Exception("Unknown datatype in 2nd axis")

        self.vkms = (np.arange(self.Nv) - 0.5 * (self.Nv - 1)) * dv_kms

        if "BMAJ" in header:
            self.beam_maj_au = header["BMAJ"] * self.dpc
            self.beam_min_au = header["BMIN"] * self.dpc
            self.beam_pa_deg = header["BPA"]
            self.vreso_kms = dv_kms

        self.bunit = header["BUNIT"]

        if ((self.dx < 0) or (self.xau[1] < self.xau[0])) or (
            (self.dv < 0) or (self.vkms[1] < self.vkms[0])
        ):
            raise Exception("reading axis is wrong.")


def read_pkl_obsdata(self, filepath):
    return pd.read_pickle(filepath)


def read_fits_PV(
    cls, filepath, unit1_in_au=None, unit2_in_kms=None, unit1="", unit2=""
):
    return PVmap(
        fitsfile=filepath,
        unit1_in_au=unit1_in_au,
        unit2_in_kms=unit2_in_kms,
        unit1=unit1,
        unit2=unit2,
    )
