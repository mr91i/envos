import os
import numpy as np
from scipy import interpolate, integrate, optimize
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as mc

try:
    from skimage.feature import peak_local_max
except:
    pass

from .plot_funcs import *
from .. import nconst as nc
from .. import log
from .. import streamline
from .. import tools


def plot_mom0_map(
    cube,
    pangle_deg=None,
    poffset_au=None,
    n_lv=100,
    out="mom0map.pdf",
    normalize=False,
    xlim=(-700, 700),
    ylim=(-700, 700),
):
    def position_line(length, pangle_deg, poffset_au=0):
        line = np.linspace(-length / 2, length / 2, 10)
        pangle_rad = (pangle_deg + 90) * np.pi / 180
        pos_x = line * np.cos(pangle_rad) - poffset_au * np.sin(pangle_rad)
        pos_y = line * np.sin(pangle_rad) + poffset_au * np.sin(pangle_rad)
        return pos_x, pos_y
        # return np.stack([pos_x, pos_y], axis=-1)

    if hasattr(cube, "Ippv"):
        if len(cube.vkms) >= 2:
            dv = cube.vkms[1] - cube.vkms[0]
        else:
            dv = 1
        # Ipp = np.sum(cube.Ippv, axis=-1) * dv
        # Ipp = cube.get_mom0_map(vrange=[-0.6, 0.6])
        # Ipp = cube.get_mom0_map(vrange=[-2.5, 2.5])
        Ipp = cube.get_mom0_map()

        # exit()
    elif hasattr(cube, "Ipp"):
        Ipp = cube.Ipp
    else:
        raise Exception("Data type error")

    if normalize:
        Ipp /= np.max(Ipp)

    lvs = np.linspace(0, np.max(Ipp), 11)
    # plt.figure(figsize=(8, 6))

    img = plt.pcolormesh(
        cube.xau,
        cube.yau,
        Ipp.T,
        cmap=make_listed_cmap("magma", len(lvs) - 1, extend="neither"),
        norm=mc.BoundaryNorm(lvs, len(lvs) - 1, clip=False),
        shading="nearest",
        rasterized=True,
    )
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(r"$x$ [au]")
    plt.ylabel(r"$z$ [au]")
    cbar = plt.colorbar(img, pad=0.02)
    cbar.set_label(r"$I ~/ ~I_{\rm max}$")

    ax = plt.gca()
    draw_center_line()
    ax.tick_params("both", direction="inout")

    if pangle_deg is not None:

        for _pangle_deg in pangle_deg:
            x, y = position_line(1400, _pangle_deg)
            # plt.plot(x, y, ls="--", c="w", lw=1.5)
            ax.annotate(
                "",
                xy=[x[-1], y[-1]],
                xytext=[x[0], y[0]],
                va="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1)
                # arrowprops=dict(shrink=0, width=0.5, headwidth=16, headlength=18, connectionstyle='arc3', facecolor='gray', edgecolor="gray")
            )

        if poffset_au is not None:
            pline = position_line(cube.xau, pangle_deg, poffset_au=poffset_au)
            plt.plot(pline[0], pline[1], c="w", lw=1)

    if (cube.conv_info is not None) and cube.beam_maj_au:
        draw_beamsize(
            ax,
            "mom0",
            cube.beam_maj_au,
            cube.beam_min_au,
            cube.beam_pa_deg,
        )

    savefig(out)


def plot_image(image,
    n_lv=11,
    out="image.pdf",
    normalize=False,
    xlim=(None, None), #(-700, 700),
    ylim=(None, None), #(-700, 700),
    Iunit="Tb",
    freq0=None,
    refimage=None,
    sigma=None,
    n_clv=100,
    zmax=None,
):
    if Iunit is None:
        Ipp = image.Ipp
        clabel = r"$I [Jy/pixel]$"
    elif (Iunit == "Tb") and (freq0 is not None):
        Ipp = image.convert_Jyppix_to_brightness_temperature(image.data, freq0=freq0)
        if refimage is not None:
            Ippref = refimage.convert_Jyppix_to_brightness_temperature(refimage.data, freq0=freq0)
        clabel = r"$T_b [{\rm K}]$"
    elif Iunit == "norm":
        Ipp /= np.max(Ipp)
        clabel = r"$I ~/ ~I_{\rm max}$"
    else:
        raise Exception("Failed to input data")

    if zmax is None:
        zmax = max([x for x in [np.max(Ipp), np.max(Ippref)] if x is not None])

    if sigma is not None:
        clvs = np.linspace(0, zmax, n_clv)
        contlvs = np.arange(0, zmax, sigma)
    else:
        clvs = np.linspace(np.min(Ipp), zmax, n_clv)
        contlvs = np.linspace(np.min(Ipp), zmax, n_lv)

    if refimage is not None:
        refcontlvs = contlvs

    img = plt.pcolormesh(
        image.xau,
        image.yau,
        Ipp.T,
        cmap=make_listed_cmap("magma", len(clvs) - 1, extend="neither"),
        norm=mc.BoundaryNorm(clvs, len(clvs) - 1, clip=False),
        shading="nearest",
        rasterized=True,
        zorder=0,
    )
    plt.contour(
        image.xau,
        image.yau,
        Ipp.T,
        levels=contlvs,
        colors="w",
        linewidths=0.8,
        zorder=2,
    )


    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(r"$x$ [au]")
    plt.ylabel(r"$y$ [au]")
    cbar = plt.colorbar(img, pad=0.02)
    cbar.set_label(clabel)

    ax = plt.gca()
    draw_center_line()
    ax.tick_params("both", direction="inout")

    if (image.obreso is not None) and image.obreso.beam_maj_au:
        draw_beamsize(
            ax,
            "mom0",
            image.obreso.beam_maj_au,
            image.obreso.beam_min_au,
            image.obreso.beam_pa_deg,
        )

    if refimage is not None:
        plt.contour(
            refimage.xau,
            refimage.yau,
            Ippref.T,
            levels=refcontlvs,
            colors="skyblue",
            linewidths=0.5,
            zorder=1,
        )

    savefig(out)



def plot_lineprofile(cube, xau_range=None, yau_range=None, unit=None, freq0=None):
    if cube.Ippv.shape[0] == 1 and cube.Ippv.shape[1] == 1:
        lp = cube.Ippv[0, 0, :]
    else:
        imin = imax = jmin = jmax = None
        if xau_range is not None:
            imin = np.where(cube.xau > xau_range[0])[0][0]
            imax = np.where( cube.xau < xau_range[-1])[0][-1]
        if yau_range is not None:
            jmin = np.where(cube.yau > yau_range[0])[0][0]
            jmax = np.where( cube.yau < yau_range[-1])[0][-1]

        if unit is None:
            Ippv = cube.Ippv
        elif unit == "Tb":
            Ippv = cube.convert_Jyppix_to_brightness_temperature(cube.Ippv, freq0=freq0)

        #lp = integrate.simps(
        #    integrate.simps(Ippv[imin:imax,jmin:jmax,:], cube.xau[imin:imax], axis=0), cube.yau[jmin:jmax], axis=0
        #)
        lp = np.average(
            np.average(Ippv[imin:imax,jmin:jmax,:], axis=0), axis=0
        )

#    print(lp)

    plt.plot(cube.vkms, lp)
    plt.xlabel("Velocity [km/s]")
    plt.ylabel("Tb [K]")

    filepath = os.path.join(gpath.fig_dir, "line.pdf")
    print("saved ", filepath)
    plt.savefig(filepath)
    plt.clf()



def plot_pvdiagram(
    pv,
    n_lv=10,
    Ms_Msun=None,
    rCR_au=None,
    incl=90,
    f_crit=0.1,
    quadrant=None,
    analysis=None,  # "mass_estimates" , "positions"
    mass_estimate=False,
    mass_ip=False,
    mass_vp=False,
    mapmode="grid",
    oplot={},
    subax=False,
    figsize=None,  # (8, 6),
    xlim=(-700, 700),
    ylim=(-2.5, 2.5),
    show_beam=True,
    discrete=True,
    refpv=None,
    loglog=False,
    out="pvdiagrams.pdf",
    clim=(0, None),
):
    Ipv = pv.Ipv
    xau = pv.xau
    vkms = pv.vkms

    plt.figure(figsize=figsize)
    lvs = np.linspace(
        clim[0], clim[1] if clim[1] is not None else np.max(Ipv), n_lv + 1
    )
    nlvs = len(lvs)

    xx, yy = np.meshgrid(xau, vkms, indexing="ij")

    if discrete:
        cmap = make_listed_cmap("cividis", n_lv, extend="neither")
        norm = mc.BoundaryNorm(lvs, n_lv, clip=False)
    else:
        cmap = plt.get_cmap("cividis")
        norm = mc.Normalize(vmin=0, vmax=np.max(Ipv))

    img = plt.pcolormesh(
        xx,
        yy,
        Ipv,
        cmap=cmap,  # make_listed_cmap("cividis", n_lv, extend="neither"),
        norm=norm,  # mc.BoundaryNorm(lvs, n_lv, clip=False),
        shading="nearest",
        rasterized=True,
    )

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Position [au]")
    plt.ylabel(r"$V$ [km s$^{-1}$]")
    cbar = plt.colorbar(img, pad=0.02)
    cbar.set_label(r"$I_{V} ~/ ~I_{V,\rm max}$")

    def calc_xmean(Ip):
        Ip = np.where(Ip > 0.1, Ip, 0)
        Ip = np.where(xau > 0, Ip, 0)
        return integrate.simpson(Ip * xau, xau) / integrate.simpson(Ip, xau)

    def calc_vmean(Iv):
        Iv = np.where(Iv > 0.1, Iv, 0)
        Iv = np.where(vkms > 0, Iv, 0)
        return integrate.simpson(Iv * vkms, vkms) / integrate.simpson(Iv, vkms)

    # xmean = [calc_xmean(Ip) for Ip in Ipv.T]
    # plt.scatter(xmean, vkms, c="r", marker="o", s=4)
    # vmean = [calc_vmean(Iv) for Iv in Ipv ]
    # plt.scatter(xau, vmean, c="y", marker="o", s=4)

    if loglog:
        plt.xlim(30, 1000)
        plt.ylim(0.1, 4)
        plt.xscale("log")
        plt.yscale("log")
        x = np.geomspace(10, 1000)

        func1 = lambda x: 0.05 * (x / 1e4) ** (-1)
        plt.plot(x, func1(x), c="w", ls="--", lw=2)
        plt.text(400 * 1.1, func1(400) * 1.1, r"$x^{-1} $", c="w", size=20)

        func2 = lambda x: 0.1 * (x / 1e4) ** (-0.5)
        plt.plot(x, func2(x), c="w", ls=":", lw=2)
        plt.text(50 * 1.1, func2(50) * 1.1, r"$x^{-0.5}$", c="w", size=20)

        if 1:
            xx, vv = np.meshgrid(xau, vkms, indexing="ij")
            vpeaks = np.array(
                [get_maximum_position(vkms, Iv) for Iv in np.where(vv > 0, Ipv, 0)]
            )
            plt.scatter(xau[vpeaks > 0], vpeaks[vpeaks > 0], c="red", marker="o", s=4)

        show_beam = False

    ax = plt.gca()
    draw_center_line()
    # ax.minorticks_on()
    # ax.tick_params("both", direction="inout")
    ax.tick_params(direction="inout")

    if refpv:
        x_ipeak, v_ipeak = get_coord_ipeak(refpv.xau, refpv.vkms, refpv.Ipv, mode="max")
        # x_vmax, v_vmax = get_coord_vmax(refpv.xau, refpv.vkms, refpv.Ipv, f_crit)
        plt.contour(
            refpv.xau,
            refpv.vkms,
            refpv.Ipv.T.clip(1e-10),
            #[0.1, 0.3, 0.5, 0.7, 0.9], #np.linspace(lvs[0], lvs[-1], 6),
            [0.3, 0.5, 0.7, 0.9], #np.linspace(lvs[0], lvs[-1], 6),
            colors="w",
            linewidths=1.0,
            linestyles="-",
            alpha=0.5,
            norm=mc.Normalize(vmin=0, vmax=np.max(refpv.Ipv)),
            corner_mask=False,
        )
        plt.scatter(x_ipeak, v_ipeak, s=15, alpha=0.5, linewidth=1, c="w", ec=None)

    if mass_estimate:
        add_mass_estimate_plot(
            xau,
            vkms,
            Ipv,
            Ms_Msun=Ms_Msun,
            rCR_au=rCR_au,
            f_crit=f_crit,
            # f_crit_list=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5 ],
            incl=incl,
            mass_ip=True,
            mass_vp=True,
            quadrant=quadrant,
        )

    if (pv.conv_info is not None) and show_beam:
        draw_beamsize(
            plt.gca(),
            "pv",
            pv.beam_maj_au,
            pv.beam_min_au,
            pv.beam_pa_deg,
            pv.pangle_deg,
            pv.vreso_kms,
        )

    if subax:
        ax2 = ax.twiny()
        # ax2.minorticks_on()
        ax2.tick_params("both", direction="inout")
        ax2.set_xlabel("Angular Offset [arcsec]")
        # ax2.set_xlim([xas[0], xas[-1]])
        ax2.set_xlim(np.array(ax.get_xlim()) / pv.dpc)

    savefig(out)
