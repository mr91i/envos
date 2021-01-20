#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from multiprocessing import Pool
from contextlib import redirect_stdout
from scipy import integrate, interpolate
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import astropy.io.fits as iofits
import astropy.convolution as aconv
import radmc3dPy.image as rmci
import radmc3dPy.analyze as rmca
from envos import tools
from envos import nconst as nc
from envos import config

#####################################


def run_and_capture(cmd):
    """
    :param cmd: str 実行するコマンド.
    :rtype: str
    :return: 標準出力.
    """
    # ここでプロセスが (非同期に) 開始する.
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    buf = []

    while True:
        # バッファから1行読み込む.
        line = proc.stdout.readline().decode("utf-8")
        # print(line)
        buf.append(line)
        sys.stdout.write(line)

        # バッファが空 + プロセス終了.
        if not line and proc.poll() is not None:
            break

    return "\n".join(buf)


"""
def main():
    osim = ObsSimulator(radmc_dir, dpc=inp.obs.dpc, sizex_au=inp.obs.sizex_au, sizey_au=inp.obs.sizey_au, pixsize_au=inp.obs.pixsize_au, incl=inp.obs.incl, phi=inp.obs.phi, posang=inp.obs.posang, omp=inp.obs.omp, n_thread=inp.obs.n_thread)
    # obsdata = osim.observe_cont(1249)
    iout = 1
    calc = 1
    if iout:
        if calc:
            obsdata2 = osim.observe_line(iline=inp.obs.iline, vwidth_kms=inp.obs.vwidth_kms, dv_kms=inp.obs.dv_kms, ispec=inp.radmc.mol_name)
            obsdata2.save_instance("obs.pkl")
        else:
            obsdata2 = ObsData(pklfile="obs.pkl") # when read
    else: # or
        if calc:
            osim.observe_line(iline=inp.obs.iline, vwidth_kms=inp.obs.vwidth_kms, dv_kms=inp.obs.dv_kms, ispec=inp.radmc.mol_name)
            osim.output_fits("obs.fits")
        # osim.output_instance("obs.pkl")
        else:
            obsdata2 = ObsData(fitsfile="obs.fits")

    obsdata2.convolve(inp.vis.beama_au, inp.vis.beamb_au, inp.vis.vreso_kms, inp.vis.beam_posang)
#    obsdata2.make_mom0_map()
#    plot_mom0_map(obs)

    PV = obsdata2.make_PV_map()
    from plot_example import plot_pvdiagram
    from header import dpath_fig
    plot_pvdiagram(PV, dpath_fig=dpath_fig, n_lv=5, Ms_Msun=0.2, rCR_au=150, f_crit=0.1)
"""


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
    camera = f"npixx {npixx} npixy {npixy} zoomau {zoomau[0]} {zoomau[1]} {zoomau[2]} {zoomau[3]}"
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

    #    def __init__(self, radmc_dir=None, dpc=None,
    #                 sizex_au=None, sizey_au=None, pixsize_au=None,
    #                 npix=None, npixx=None, npixy=None,
    #                 incl=0, phi=0, posang=0, omp=True, n_thread=1, inp=None,
    #                 beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=None, convmode="fft")
    #        self.radmc_dir = radmc_dir
    #        self.dpc = dpc
    #        self.omp = omp
    #        self.n_thread = n_thread
    #        self.incl = incl
    #        self.phi = phi
    #        self.posang = posang

    def __init__(self, radmcdir=None, dpc=None, omp=True, n_thread=1):
        # beam_maj_au=None, beam_min_au=None, vreso_kms=None, beam_pa_deg=None, convmode="fft")

        self.radmc_dir = radmcdir or config.dp_radmc
        self.dpc = dpc
        self.omp = omp
        self.n_thread = n_thread
        # self.incl = incl
        # self.phi = phi
        # self.posang = posang

        self.view = False
        self.conv = False
        # self.set_camera(sizex_au, sizey_au, pixsize_au=pixsize_au, npix=npix, npixx=npixx, npixy=npixy)

    #    def set_from_inp(self, inp):
    #        self.radmc_dir = inp.radmc_dir
    #        self.dpc = inp.dpc
    #        self.omp = inp.omp
    #        self.n_thread = inp.n_thread
    #        self.incl = inp.incl_deg
    #        self.phi = inp.phi_deg
    #        self.posang = inp.posang_deg
    #
    #        self.beam_maj_au = inp.beam_maj_au
    #        self.beam_min_au = inp.beam_min_au
    #        self.vreso_kms = inp.vreso_kms
    #        self.beam_pa_deg = inp.beam_pa_deg
    #        self.convmode = inp.convmode
    #        #self.set_camera(inp.sizex_au, inp.sizey_au, pixsize_au=inp.pixsize_au, npix=inp.npix, npixx=inp.npixx, npixy=inp.npixy)
    #        self.set_camera(inp.sizex_au, inp.sizey_au, pixsize_au=inp.pixsize_au)
    #        """

    def exe(self, cmd, wdir, log=False):

        os.chdir(wdir)
        if log:
            out = run_and_capture(cmd)
            if "ERROR" in out:
                raise Exception(out)
        else:
            return subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

    # def set_resolution(self, sizex_au, sizey_au=None, pixsize_au=None, npix=100, npixx=None, npixy=None, incl=0, phi=0, posang=0, vwidth_kms=None, dv_kms=None):
    def set_resolution(
        self,
        sizex_au,
        sizey_au=None,
        pixsize_au=None,
        npix=100,
        npixx=None,
        npixy=None,
        vwidth_kms=None,
        dv_kms=None,
    ):
        # self.reso = True
        # self.incl = incl
        # self.phi = phi
        # self.posang = posang

        if sizex_au != sizey_au:
            logger.info(
                "Use rectangle view mode. Currently (2020/12), a bug in this mode was fixed."
            )
            logger.info(
                "Please check if your version is latest. If you do not want to use this, please set sizex_au = sizey_au."
            )

        self.zoomau_x = [
            -sizex_au / 2,
            sizex_au / 2,
        ]  # if self.sizex_au == 0 else [-self.pixsize_au/2, self.pixsize_au/2]
        _sizey_au = sizey_au or sizex_au
        self.zoomau_y = [
            -_sizey_au / 2,
            _sizey_au / 2,
        ]  # if self.sizey_au == 0 else [-self.pixsize_au/2, self.pixsize_au/2]

        if pixsize_au is not None:
            self.pixsize_au = pixsize_au
            self.npixx = int(
                round((self.zoomau_x[1] - self.zoomau_x[0]) / pixsize_au)
            )  # // or int do not work to convert into int.
            self.npixy = int(
                round((self.zoomau_y[1] - self.zoomau_y[0]) / pixsize_au)
            )  # // or int do not work to convert into int.
        else:
            self.npixx = npixx or npix
            self.npixy = npixy or npix
            self.pixsize_au = (
                self.zoomau_x[1] - self.zoomau_x[0]
            ) / self.npixx

        self.dx_au, self.dy_au = sizex_au / self.npixx, sizey_au / self.npixy

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
        self.conv = True
        self.convolver = Convolver(
            (beam_maj_au, beam_min_au, vreso_kms),
            (self.dx_au, self.dy_au, self.dv_kms),
            beam_pa_deg=beam_pa_deg,
            mode=convmode,
        )

    def observe_cont(self, lam, incl=0, phi=0, posang=0):
        self.obs_mode = "cont"
        self.lam = lam
        cmd = gen_radmc_cmd(
            mode="image",
            dpc=self.dpc,
            incl=incl,
            phi=phi,
            posang=posang,
            npixx=self.npixx,
            npixy=self.npixx,
            lam=lam,
            zoomau=[*self.zoomau_x, *self.zoomau_y],
            option="noscat nostar",
        )

        self.exe(cmd, self.radmc_dir, log=True)
        self.data_cont = rmci.readImage()
        self.data_cont.freq0 = nc.c / (lam * 1e4)
        if self.conv:
            self.data_cont.image = self.convolver(self.data_cont.image)

        odat = ObsData(radmcdata=self.data_cont, datatype=self.obs_mode)
        return odat

    def observe_line(self, iline, ispec, incl=0, phi=0, posang=0):
        self.obs_mode = "line"
        self.iline = iline
        # self.vwidth_kms = vwidth_kms
        # self.dv_kms = dv_kms
        self.nlam = 2 * int(round(self.vwidth_kms / self.dv_kms)) + 1
        self.mol = rmca.readMol(fname=f"{self.radmc_dir}/molecule_{ispec}.inp")
        logger.info(
            f"Total cell number is {self.npixx} x {self.npixy} x {self.nlam} = {self.npixx*self.npixy*self.nlam}"
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
        vseps = np.linspace(
            -self.vwidth_kms - self.dv_kms / 2,
            self.vwidth_kms + self.dv_kms / 2,
            self.nlam + 1,
        )

        if self.omp and (self.n_thread > 1):
            n_points = self.divide_threads(self.n_thread, self.nlam)
            v_thread_seps = self.calc_thread_seps(v_calc_points, n_points)
            v_center = [
                0.5 * (v_range[1] + v_range[0]) for v_range in v_thread_seps
            ]
            v_width = [
                0.5 * (v_range[1] - v_range[0]) for v_range in v_thread_seps
            ]

            logger.info(f"All calc points: {format_array(v_calc_points)}")
            logger.info("Calc points in each thread:")
            for i, (vc, vw, ncp) in enumerate(
                zip(v_center, v_width, n_points)
            ):
                logger.info(
                    f"    {i}th thread: {format_array(np.linspace(vc-vw, vc+vw, ncp))}"
                )

            args = [
                (
                    p,
                    gen_radmc_cmd(
                        vc_kms=v_center[p],
                        vw_kms=v_width[p],
                        nlam=n_points[p],
                        **common_cmd,
                    ),
                )
                for p in range(self.n_thread)
            ]
            rets = Pool(self.n_thread).starmap(self._subcalc, args)
            self._check_multiple_returns(rets)
            self.data = self._combine_multiple_returns(rets)

        else:
            cmd = gen_radmc_cmd(
                vw_kms=self.vwidth_kms, nlam=self.nlam, **common_cmd
            )
            logger.info(f"command is {cmd}")

            cwd = os.getcwd()
            os.chdir(self.radmc_dir)
            subprocess.call(cmd, shell=True)
            self.data = rmci.readImage()
            os.chdir(cwd)

        if np.max(self.data.image) == 0:
            logger.warning("Zero image !")

        #        self.data.dpc = self.dpc
        self.data.freq0 = self.mol.freq[iline - 1]
        if self.conv:
            self.data.image = self.convolver(self.data.image)
        return ObsData(radmcdata=self.data, datatype=self.obs_mode)

    @staticmethod
    def find_proper_nthread(n_thr, n_div):
        return max([i for i in range(n_thr, 0, -1) if n_div % i == 0])

    @staticmethod
    def divide_threads(n_thr, n_div):
        ans = [n_div // n_thr] * n_thr
        rem = n_div % n_thr
        nlist = [
            (n_thr - i // 2 if i % 2 else i // 2) for i in range(n_thr)
        ]  # [ ( i if i%2==0 else n_thread -(i-1) ) for i in range(n_thread) ]
        for i in range(rem):
            ii = (n_thr - 1 - i // 2) if i % 2 else i // 2
            ans[ii] += 1
        return ans

    @staticmethod
    def calc_thread_seps(calc_points, thread_divs):
        ans = []
        sum_points = 0
        for npoints in thread_divs:
            ans.append(
                [
                    calc_points[sum_points],
                    calc_points[sum_points + npoints - 1],
                ]
            )
            sum_points += npoints
        return np.array(ans)

    def _subcalc(self, p, cmd):
        dn = f"proc{p:d}"
        logger.info("execute: " + cmd)
        dpath_sub = f"{self.radmc_dir}/{dn}"
        os.makedirs(dpath_sub, exist_ok=True)
        os.system(f"cp {self.radmc_dir}/{{*.inp,*.dat}} {dpath_sub}/")
        self.exe(
            cmd, dpath_sub, log=(logger.isEnabledFor(logging.DEBUG) and p == 1)
        )
        with redirect_stdout(open(os.devnull, "w")):
            return rmci.readImage()

    def _check_multiple_returns(self, return_list):
        for i, r in enumerate(return_list):
            logger.debug(f"The{i}th return")
            for k, v in r.__dict__.items():
                if isinstance(v, (np.ndarray)):
                    logger.debug(
                        "{}: shape is {}, range is [{}, {}]".format(
                            k, v.shape, np.min(v), np.max(v)
                        )
                    )
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

    def output_instance(self, filepath):
        pd.to_pickle(self, filepath)

    def read_instance(self, filepath):
        instance = pd.read_pickle(filepath)
        for k, v in instance.__dict__.items():
            setattr(self, k, v)


def format_array(array):
    return f"[{min(array):.2f}:{max(array):.2f}] with delta = {abs(array[1]-array[0]):.4g} and N = {len(array)}"


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

        if self.mode == "normal":
            return aconv.convolve(image, Kernel)
        elif self.mode == "fft":
            return aconv.convolve_fft(image, Kernel, allow_huge=True)


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

        self.conv = False
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

    #
    #   Freeze: Revive in future version
    #

    def set_dpc(self, dpc):
        self.dpc = dpc

    def read_radmcdata(self, data):
        # self.Ippv_raw = data.image.transpose(2, 1, 0)
        self.Ippv = data.image.transpose(2, 1, 0)
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

    #    def convolve(self, beam_maj_au, beam_min_au, vreso_kms, beam_pa_deg, mode="fft", pointsource_test=False):

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
        # pangle_deg = pangle_deg if pangle_deg else self
        if self.Ippv.shape[1] > 1:
            posline = self.position_line(
                self.xau, PA_deg=pangle_deg, poffset_au=poffset_au
            )
            points = [[(v, pl[1], pl[0]) for pl in posline] for v in self.vkms]
            Ipv = interpolate.interpn(
                (self.vkms, self.yau, self.xau),
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
            filepath = os.path.join(config.dp_run, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pd.to_pickle(self, filepath)

    def save_fits(self, filename="obsdata.fits", dpc=None, filepath=None):
        if filepath is None:
            filepath = os.path.join(config.dp_run, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if os.path.exists(filepath):
            logger.info(f"remove old fits file: {filepath}")
            os.remove(filepath)
        self.data.writeFits(fname=filepath, dpc=dpc)
        logger.info(f"Saved fits file: {fp_fitsdata}")


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
            read_fits_PV(self, fitsfile)
            # self.read_fitsfile(filepath=fitsfile)

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

    def trim(range_x=None, range_v=None):
        if range_x is not None:
            imax, imin = [np.abs(self.xau - xlim).argmin() for xl in range_x]
            self.Ipv = self.Ipv[:, imin:imax]
            self.xau = self.xau[imin:imax]

        if ramge_v is not None:
            jmin, jmax = [np.abs(self.vkms - vlim).argmin() for vl in range_v]
            self.Ipv = self.Ipv[jmin:jmax, :]
            self.vkms = self.vkms[jmin:jmax]

    def save_fitsfile(
        self, filename="PVimage.fits", unitp="au", unitv="km/s", filepath=None
    ):
        # see IAU manual : https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
        if filepath is None:
            filepath = os.path.join(config.dp_run, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        Np = len(self.xau)
        Nv = len(self.vkms)
        dx = self.xau[1] - self.xau[0]
        dv = self.vkms[1] - self.vkms[0]
        hdu = iofits.PrimaryHDU(self.Ipv)
        new_header = {
            "NAXIS": 2,
            "CTYPE1": "ANGLE",
            "CUNIT1": unitp,
            "NAXIS1": Np,
            "CRVAL1": 0.0,
            "CRPIX1": (Np + 1) / 2,
            "CDELT1": np.radians(dx * nc.au / self.dpc),
            "CUNIT1": "deg",
            "CTYPE2": "VRAD",
            "CUNIT2": unitv,
            "NAXIS1": Nv,
            "CRVAL2": 0.0,
            "CRPIX2": (Nv + 1) / 2,
            "CDELT2": dv * 1e3,
            "CUNIT2": "m/s",
            "BTYPE": "Intensity",
            "BUNIT": "Jy/pixel",
        }
        if self.beam_maj_au:
            nue_header.update(
                {
                    "BMAJ": self.beam_maj_au,
                    "BMIN": self.beam_min_au,
                    "BPA": self.beam_pa_deg,
                }
            )
        hdu.header.update(new_header)
        hdulist = iofits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)

    # SIMPLE  =                    T                                                  BITPIX  =                  -32                                                  NAXIS   =                    2                                                  NAXIS1  =                  122                                                  NAXIS2  =                   56                                                  CTYPE1  = 'ANGLE   '                                                            CRVAL1  =  0.0000000000000E+00                                                  CRPIX1  =  6.2000000000000E+01                                                  CDELT1  =  3.4722222324035E-05                                                  CTYPE2  = 'VRAD    '                                                            CRVAL2  =  1.4886976186320E+03                                                  CRPIX2  = -2.0000000000000E+00                                                  CDELT2  =  1.4693057132590E+02                                                  BUNIT   = 'JY/BEAM '                                                            OBJECT  = 'IRAS04368+2557'                                                      DATE-OBS= '2012-08-10T10:49:13.824000'                                          OBSRA   =  6.9974458333330E+01                                                  OBSDEC  =  2.6052694444440E+01                                                  TELESCOP= 'ALMA    '                                                            EQUINOX =  2.0000000000000E+03                                                  BTYPE   = 'Intensity'                                                           BMAJ    =  2.1805740065040E-04                                                  BMIN    =  1.8803813391260E-04                                                  BPA     =  1.7929582214360E+02                                                  VELREF  =                  257                                                  DATAMIN = -6.8105190992355E-02                                                  DATAMAX =  6.2522120773792E-02

    def read_fitsfile(self, filepath, unit1_in_au=None, unit2_in_kms=None):
        logger.info(f"Reading fits file: {filepath}")
        pic = iofits.open(filepath)[0]
        self.Ipv = pic.data
        print("Ipv_max : ", np.max(self.Ipv))
        header = pic.header

        self.Nx = header["NAXIS1"]
        if unit1_in_cm:
            logger.info(
                f"   1st axis is interpreted as POSITION with user-defined unit (unit1 = {unit1_in_cm} cm)."
            )
            self.dx = header["CDELT1"] * unit1_in_au
        elif "ANGLE" in header["CYTYPE"] and unit1 == "deg":  # x in deg
            logger.info(f"   1st axis is interpreted as POSITION [deg].")
            self.dx = (
                header["CDELT1"] * np.pi / 180.0 * self.dpc * nc.pc / nc.au
            )
        else:
            raise Exception("Unknown datatype in 1st axis")
        self.xau = (
            -0.5 * self.Nx * self.dx + (np.arange(self.Nx) + 0.5) * self.dx
        )

        self.Nv = header["NAXIS2"]
        if unit2_in_kms:
            logger.info(
                f"    2nd axis is interpreted as VELOCITY with user-defined unit (unit2 = {unit2_in_cm} cm)."
            )
            self.dv = header["CDELT2"] * unit2_in_kms
        elif "VRAD" in header["CYTYPE"] and unit2 == "m/s":  # v in m/s
            logger.info(f"    2nd axis is interpreted as VELOCITY [m/s].")
            self.dv = header["CDELT2"] / 1e3  # in m/s to in km/s
        else:
            raise Exception("Unknown datatype in 2nd axis")
        self.vkms = (-0.5 * (self.Nv - 1) + np.arange(self.Nv)) * self.dv

        if "BMAJ" in header:
            self.beam_maj_au = header["BMAJ"] * self.dpc
            self.beam_min_au = header["BMIN"] * self.dpc
            self.beam_pa_deg = header["BPA"]
            self.vreso_kms = self.dv

        self.bunit = header["BUNIT"]

        if ((self.dx < 0) or (self.xau[1] < self.xau[0])) or (
            (self.dv < 0) or (self.vkms[1] < self.vkms[0])
        ):
            raise Exception("reading axis is wrong.")


def read_pkl_obsdata(self, filepath):
    return pd.read_pickle(filepath)


def read_fits_PV(cls, filepath, unit1_in_au=None, unit2_in_kms=None):
    logger.info(f"Reading fits file: {filepath}")
    pic = iofits.open(filepath)[0]
    cls.Ipv = pic.data
    print("Ipv_max : ", np.max(cls.Ipv))
    header = pic.header

    cls.Nx = header["NAXIS1"]
    if unit1_in_cm:
        logger.info(
            f"   1st axis is interpreted as POSITION with user-defined unit (unit1 = {unit1_in_cm} cm)."
        )
        cls.dx = header["CDELT1"] * unit1_in_au
    elif "ANGLE" in header["CYTYPE"] and unit1 == "deg":  # x in deg
        logger.info(f"   1st axis is interpreted as POSITION [deg].")
        cls.dx = header["CDELT1"] * np.pi / 180.0 * cls.dpc * nc.pc / nc.au
    else:
        raise Exception("Unknown datatype in 1st axis")
    cls.xau = -0.5 * cls.Nx * cls.dx + (np.arange(cls.Nx) + 0.5) * cls.dx

    cls.Nv = header["NAXIS2"]
    if unit2_in_kms:
        logger.info(
            f"    2nd axis is interpreted as VELOCITY with user-defined unit (unit2 = {unit2_in_cm} cm)."
        )
        cls.dv = header["CDELT2"] * unit2_in_kms
    elif "VRAD" in header["CYTYPE"] and unit2 == "m/s":  # v in m/s
        logger.info(f"    2nd axis is interpreted as VELOCITY [m/s].")
        cls.dv = header["CDELT2"] / 1e3  # in m/s to in km/s
    else:
        raise Exception("Unknown datatype in 2nd axis")
    cls.vkms = (-0.5 * (cls.Nv - 1) + np.arange(cls.Nv)) * cls.dv

    if "BMAJ" in header:
        cls.beam_maj_au = header["BMAJ"] * cls.dpc
        cls.beam_min_au = header["BMIN"] * cls.dpc
        cls.beam_pa_deg = header["BPA"]
        cls.vreso_kms = cls.dv

    cls.bunit = header["BUNIT"]

    if ((cls.dx < 0) or (cls.xau[1] < cls.xau[0])) or (
        (cls.dv < 0) or (cls.vkms[1] < cls.vkms[0])
    ):
        raise Exception("reading axis is wrong.")


if __name__ == "__main__":
    main()
