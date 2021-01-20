#!/usr/bin/env python
import astropy.io.fits as fits
import numpy as np
import glob
import re

# import myplot5 as myp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, optimize, interpolate
import nconst as nc

#########################################################################
plabel_PV1 = "CM_T30_CR31.7_M0.2_cav45_incl30"
param_name = "bs0.125"
param_dict = {
    "bs0.125": ["./data9-2", "CM_bs0.125_T30_CR*_M*_cav45"],
}[param_name]
mode_get_table = 2  # 1: from PVchi, 2:from pvcor
method_corel = {1: "SSD", 2: "ZNCC"}[2]
calc_mass_method = {1: "vpeak", 2: "ipeak"}[1]

#########################################################################
data_dir = param_dict[0]
re_number = r"([\d\.e\+]+)"
PV2_PARAM_PATTERN = param_dict[1].replace("*", re_number)
# "_".join(["".join(map(str, pl)) for pl in param_list]).replace("*", re_number)
fp_PV1_data = f"{data_dir}/fig_{plabel_PV1}/PV.fits"
fp_PV1_fitsdata = f"{data_dir}/fitsdata/{plabel_PV1}.fits"
CR0, M0 = map(
    float, re.search(r"CR([\d\.e\+]+)_M([\d\.e\+]+)", plabel_PV1).groups()
)
pat_PV2 = re.compile(PV2_PARAM_PATTERN)
fits_dir = data_dir + "/fitsdata"
out_dir = data_dir + "/out"
#########################################################################


def looper():
    global method_corel, calc_mass_method, param_name
    param_tag = PV2_PARAM_PATTERN

    for cmm in ["vpeak", "ipeak"]:
        calc_mass_method = cmm
        data_tag = param_name
        tag = data_tag + "_" + cmm
        # print(tag)
        main_est(tag=tag)


def main():
    points = {1: read_data_from_PVchi2(), 2: data_from_pvcor()}[mode_get_table]
    {"SSD": plot_map_SSD, "ZNCC": plot_map_ZNCC}[method_corel](points)


#############################################################
class PVdiagram:
    # def __init__(self, hdu, x1style="au", x2style="km/s"):
    def __init__(self, data, header, x1style="au", x2style="km/s"):
        self.I = data
        # self.I = hdu.data
        # header = hdu.header
        if x1style == "au":
            self.Nx = header["NAXIS1"]
            self.dx = header["CDELT1"]
            self.Lx = self.Nx * self.dx
            self.xau = -0.5 * self.Lx + (np.arange(self.Nx) + 0.5) * self.dx

        if x2style == "km/s":
            self.Nv = header["NAXIS2"]
            self.dv = header["CDELT2"]
            self.Lv = self.Nv * self.dv
            self.vkms = -0.5 * self.Lv + (np.arange(self.Nv) + 0.5) * self.dv

        self.nan = np.any(np.isnan(self.I))
        if self.nan:
            print("data is nan")


def data_from_pvcor(ranking=True):
    dplist_PV2 = [
        path for path in glob.glob(data_dir + "/fig_*") if pat_PV2.search(path)
    ]
    pvd1 = gen_pvd(fp_PV1_data)
    pvd_list = make_pvdlist_corr(pvd1, dplist_PV2)
    if len(pvd_list) == 0:
        raise Exception("Failed to generate table.")
    if ranking:
        order = {"ZNCC": -1, "SSD": 1}[method_corel]
        pvd_list_nonan = [pvd for pvd in pvd_list if not pvd.nan]
        pvd_list_sorted = sorted(pvd_list_nonan, key=lambda x: x.corr)
        for i, pvd in enumerate(pvd_list_sorted[:10]):
            print(f"{i+1}: CR={pvd.CR} , M={pvd.Ms} -->  {pvd.corr}")
    return pvd_list


import psutil
import copy


def gen_pvd(filename, mode="fits"):
    if mode == "fits":
        with fits.open(filename) as hdulist:
            hdu = hdulist[0]
            pvd = PVdiagram(
                copy.copy(hdu.data), hdu.header, x1style="au", x2style="km/s"
            )
            del hdu.data
        return pvd


def make_pvdlist_corr(pvd1, dp_list):
    def gen_data(dp):
        print("dpath:", dp)
        pvd2 = gen_pvd(dp + "/PV.fits")
        pvd2.Ms, pvd2.CR = get_param_from_inputfile(dp)
        pvd2.corr = float(
            calc_correlation(pvd1.I, pvd2.I, method=method_corel)
        )
        return pvd2

    return np.array([gen_data(dp) for dp in dp_list])


def get_param_from_inputfile(dirpath):
    inpfile = glob.glob(dirpath + "/*.in.*")
    with open(inpfile[0]) as f:
        for l in f.readlines():
            if len(l) <= 1:
                continue
            words = l.split()
            if words[0] == "model.Ms_Msun":
                M = words[2]
            if words[0] == "model.rCR_au":
                CR = words[2]

    return float(M), float(CR)


########################################################################################


def calc_correlation(
    im1, im2_original, method="ZNCC", threshold=0, with_noise=False
):
    jmax1, imax1 = im1.shape
    jmax2, imax2 = im2_original.shape
    im2 = ndimage.zoom(im2_original, (jmax1 / jmax2, imax1 / imax2), order=1)
    im1 = np.where(im1 > 0, im1, 0)
    im2 = np.where(im2 > 0, im2, 0)
    if with_noise:
        noise_level = 0.33
        im1 += np.random.uniform(-noise_level, noise_level, size=im1.shape)
    # Noise cut
    im1 = np.where(im1 > threshold, im1, 0)
    im1 = np.where(im1 / np.max(im1) > 0.00, im1, 0)
    im2 = np.where(im2 / np.max(im2) > 0.00, im2, 0)
    return {"ZNCC": calc_ZNCC(im1, im2), "SSD": calc_SSD(im1, im2)}[method]


def calc_SSD(im1, im2):
    return np.sum((im1 - im2) ** 2)


def calc_ZNCC(im1, im2):
    imax1, jmax1 = im1.shape
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1 ** 2)
    sumBsq = np.sum(im2 ** 2)
    print(imax1, jmax1, sumAB, sumA, sumB)
    ZNCC_AB = (
        (imax1 * jmax1 * sumAB - sumA * sumB)
        / np.sqrt(imax1 * jmax1 * sumAsq - sumA ** 2)
        / np.sqrt(imax1 * jmax1 * sumBsq - sumB ** 2)
    )
    return ZNCC_AB


#########################################################################################
def main_est(tag=""):
    pvdl = make_pvdl_with_estimation_from_image()
    plot_map_Mest(pvdl, tag=tag)


####


def make_pvdl_with_estimation_from_image():
    dplist = [
        path for path in glob.glob(data_dir + "/fig_*") if pat_PV2.search(path)
    ]
    pvd_list = make_pvd_list(dplist)
    if len(pvd_list) == 0:
        raise Exception("Failed to generate table.")
    return pvd_list


def make_pvd_list(dp_list):
    def gen_data(dp):
        print(f"Parsing {dp}")
        pvd = gen_pvd(dp + "/PV.fits")
        pvd.Ms, pvd.CR = get_param_from_inputfile(dp)
        pvd.Mest = calc_Mass(pvd, method=calc_mass_method)
        if isinstance(pvd.CR, str):
            print(vars(pvd))
            exit()
        return pvd

    return np.array([gen_data(dp) for dp in dp_list])


import tools
from scipy import interpolate, optimize
from skimage.feature import peak_local_max


def get_maximum_position(x, y):
    return find_local_peak_position(x, y, np.argmax(y))


def find_local_peak_position(x, y, i):
    if 2 <= i <= len(x) - 3:
        grad_y = interpolate.InterpolatedUnivariateSpline(
            x[i - 2 : i + 3], y[i - 2 : i + 3]
        ).derivative(1)
        return optimize.root(grad_y, x[i]).x[0]
    else:
        return np.nan


def calc_Mass(pvd, threshold=0, f_crit=0.2, method="vpeak"):
    im = pvd.I
    jmax, imax = im.shape
    if method == "vpeak":
        x_vmax, I_vmax = np.array(
            [
                [get_maximum_position(pvd.xau, Iv), np.max(Iv)]
                for Iv in im.transpose(0, 1)
            ]
        ).T
        v_crit = tools.find_roots(pvd.vkms, I_vmax, f_crit * np.max(im))

        # if (np.min(I_vmax) < f_crit*cblim[1]) and (f_crit*cblim[1] < np.max(I_vmax)):
        #    print(f"Use {self.f_crit}*cblim")
        #    v_crit = mytools.find_roots(self.vkms, I_vmax, self.f_crit*cblim[1])
        # else:
        #    print(f"Use {self.f_crit}*max Ippv")
        #    v_crit = mytools.find_roots(self.vkms, I_vmax, self.f_crit*np.max(Ippv) )
        if len(v_crit) == 0:
            v_crit = [pvd.vkms[0]]

        x_crit = tools.find_roots(x_vmax, pvd.vkms, v_crit[0])
        # M_CB = (abs(x_crit[0]) * nc.au * (v_crit[0]*nc.kms)**2 )/(np.sqrt(2)*nc.G*nc.Msun)
        M_CB = (abs(x_crit[0]) * nc.au * (v_crit[0] * nc.kms) ** 2) / (
            2 * nc.G * nc.Msun
        )
        # M_CR_vpeak = (abs(x_crit[0]) * nc.au * (v_crit[0]*nc.kms)**2 )/(np.sqrt(2)*nc.G*nc.Msun)
        if pvd.Ms == 0.2 and pvd.CR == 200:
            print(x_crit, v_crit, M_CB)
        return M_CB

    elif method == "ipeak":
        jpeak, ipeak = peak_local_max(
            im[jmax // 2 :, imax // 2 :], num_peaks=1
        )[0]
        vkms_peak = pvd.vkms[jmax // 2 + jpeak]
        xau_peak = pvd.xau[imax // 2 + ipeak]
        M_CR = (abs(xau_peak) * nc.au * (vkms_peak * nc.kms) ** 2) / (
            nc.G * nc.Msun
        )
        return M_CR


#############################################################


def read_data_from_PVchi2():
    PV2_pathlist_out = [
        path for path in glob.glob(f"{out_dir}/*.out") if pat_PV2.search(path)
    ]
    return make_table(PV2_pathlist_out, get_val_method=read_value_from_PVchi2)


def read_value_from_PVchi2(filename):
    with open(filename) as f:
        line = f.readlines()[-4]
        print(line)
        val = line.split()[-1]
    return val


def make_table(filenamelist, get_val_method=read_value_from_PVchi2):
    def gen_line(fn):
        mat = pat_PV2.search(fn)
        if mat is None:
            raise Exception(fn)
        CR, M = mat.groups()
        return [float(CR), float(M), float(get_val_method(fn))]

    return np.array([gen_line(fn) for fn in filenamelist])


#############################################################

plotmode = {1: "tricontourf", 2: "scatter"}[1]


def plot_map_SSD(table):
    points = np.array(table)
    points[:, 2] = points[:, 2] / min(points[:, 2])
    points[:, :2] = np.log10(points[:, :2])
    mp = myp.Plotter("F")
    # cbdelta = 0.1
    # cbax = np.arange( max(points[:,2]), min(points[:,2])-cbdelta, -cbdelta)[::-1]
    mp.map(
        points=points,
        mode=plotmode,
        xlim=[0.8, 3.8],
        ylim=[-2.2, 1.2],
        square=True,
        save=False,
        cbl="",
        xl="log CR [au]",
        yl=r"log M [M$_{\odot}$]",
    )
    mp.ax.scatter(
        np.log10(CR0),
        np.log10(M0),
        c="r",
        s=50,
        alpha=1,
        linewidth=0,
        zorder=10,
    )
    ibest = np.argmin(points[:, 2])
    mp.ax.scatter(
        points[ibest, 0],
        points[ibest, 1],
        c="orange",
        s=30,
        alpha=1,
        linewidth=0,
        zorder=10,
    )
    zfun_rbf = interpolate.Rbf(
        points[:, 0], points[:, 1], points[:, 2], function="cubic", smooth=0
    )  # default smooth=0 for interpolation
    res = optimize.minimize(
        lambda v: zfun_rbf(v[0], v[1]), np.array([np.log10(CR0), np.log10(M0)])
    )
    mp.ax.scatter(
        res.x[0], res.x[1], c="green", s=30, alpha=1, linewidth=0, zorder=10
    )
    mp.save("map")


def plot_map_ZNCC(pvd_list):
    table = np.array([[pvd.CR, pvd.Ms, pvd.corr] for pvd in pvd_list])
    table_nonan = table[~np.isnan(table[:, -1])]
    points = np.array(table_nonan)
    points[:, 2] = points[:, 2] / max(points[:, 2])
    points[:, :2] = np.log10(points[:, :2])
    mp = myp.Plotter("F")
    cbdelta = 0.1
    cbax = np.arange(max(points[:, 2]), min(points[:, 2]) - cbdelta, -cbdelta)[
        ::-1
    ]
    # cbax = np.arange( max(points[:,2]), max(points[:,2])*0.9, -cbdelta)[::-1]
    print(points)
    mp.map(
        points=points,
        mode=plotmode,
        xlim=[1.3, 3.3],
        ylim=[-1.7, 0.3],
        square=True,
        save=False,
        cbl="",
        xl="log CR [au]",
        yl=r"log M [M$_{\odot}$]",
        cbax=cbax,
    )
    mp.ax.scatter(
        np.log10(CR0),
        np.log10(M0),
        c="r",
        s=50,
        alpha=1,
        linewidth=0,
        zorder=10,
    )
    ibest = np.argmax(points[:, 2])
    # mp.ax.scatter(points[ibest,0], points[ibest,1], c="orange", s=30 , alpha=1, linewidth=0, zorder=10)
    points_ip = points[~np.isnan(points).any(axis=1)]
    zfun_rbf = interpolate.Rbf(
        points_ip[:, 0],
        points_ip[:, 1],
        -points_ip[:, 2],
        function="cubic",
        smooth=0,
    )  # default smooth=0 for interpolation
    res = optimize.minimize(
        lambda v: zfun_rbf(v[0], v[1]), np.array([np.log10(CR0), np.log10(M0)])
    )
    print(res.x[0], res.x[1])
    mp.ax.scatter(
        res.x[0], res.x[1], c="orange", s=30, alpha=1, linewidth=0, zorder=10
    )
    points_false = points[np.isnan(points).any(axis=1)]
    mp.ax.scatter(
        points_false[:, 0],
        points_false[:, 1],
        c="red",
        s=30,
        marker="x",
        alpha=1,
        linewidth=1,
        zorder=10,
    )
    mp.save("map")


def plot_map_Mest(pvd_list, tag=""):
    table = np.array([[pvd.CR, pvd.Ms, pvd.Mest] for pvd in pvd_list])
    table_nonan = table[~np.isnan(table[:, -1])]
    points = np.array(table_nonan)

    points[:, 2] = (points[:, 2] / points[:, 1] - 1) * 100
    mp = myp.Plotter("F")
    cbdelta = 0.1
    xlim = [min(points[:, 0]), max(points[:, 0])]
    ylim = [min(points[:, 1]), max(points[:, 2])]
    cnlim = [-60, 60]
    ctdelta = 10
    ctax = np.concatenate(
        [
            np.arange(
                -ctdelta / 2,
                max(min(points[:, 2]), cnlim[0]) - ctdelta,
                -ctdelta,
            )[::-1],
            np.arange(
                ctdelta / 2,
                min(max(points[:, 2]), cnlim[1]) + ctdelta,
                ctdelta,
            ),
        ]
    )
    mp.map(
        points=points,
        mode=plotmode,
        square=True,
        save=False,
        logxy=True,
        cbl="Error in Estimated Mass [%]",
        xl="CR [au]",
        yl=r"M [M$_{\odot}$]",
        ctax=ctax,
        cmap=plt.cm.get_cmap("coolwarm"),
        cnlim=cnlim,
    )
    from matplotlib.ticker import (
        MultipleLocator,
        FormatStrFormatter,
        FuncFormatter,
    )
    import matplotlib as mpl

    mp.cbar.formatter = FuncFormatter(lambda x, pos: f"{x:> .0f}")
    mp.cbar.update_ticks()
    mp.ax.xaxis.set_minor_formatter(FormatStrFormatter(""))
    mp.ax.yaxis.set_minor_formatter(FormatStrFormatter(""))
    mp.save("map" + tag)
    return
    points_new = points[:, [0, 2, 1]]
    points_new[:, 2] = (points_new[:, 1] / points_new[:, 2] - 1) * 100
    mp.map(
        points=points_new,
        mode=plotmode,
        out="mapM",
        logxy=True,
        square=True,
        cnlim=cnlim,
        cmap=plt.cm.get_cmap("coolwarm"),
        cbl="Error in Estimated Mass [%]",
        xl="CR [au]",
        yl=r"M [M$_{\odot}$]",
    )


# main()
# main_est()
# looper()
