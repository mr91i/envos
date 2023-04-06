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

"""
-- Plan for new structure

Plotter
+ PlotterBase
+ ObsPlotterBase
+ PhyPlotterBase

ImagePlotter(PlotterBase, ObsPlotterBase)


----------------


"""


#def plot_mom0_map(
#    data,
#    out="mom0map.pdf",
#    xlim=(-700, 700),
#    ylim=(-700, 700),
#    clim=(None, None),
#    nlv=10,
#    norm=None,
#    pangle_deg=None,
#    poffset_au=None,
#    discrete=False,
#    save=True,
#    Iunit=None,
#):
#    def position_line(length, pangle_deg, poffset_au=0):
#        line = np.linspace(-length / 2, length / 2, 10)
#        pangle_rad = (pangle_deg + 90) * np.pi / 180
#        pos_x = line * np.cos(pangle_rad) - poffset_au * np.sin(pangle_rad)
#        pos_y = line * np.sin(pangle_rad) + poffset_au * np.sin(pangle_rad)
#        return pos_x, pos_y
#
#    if data.dtype == "Cube":
#        image = data.get_mom0_map()
#    elif data.dtype == "Image":
#        image = data.copy()
#    else:
#        raise Exception("Data type error")
#
#    if norm is not None:
#        image.norm_I(norm)
#
#    lvs = np.linspace(
#        clim[0] if clim[0] is not None else np.min(image.get_I()),
#        clim[1] if clim[1] is not None else np.max(image.get_I()),
#        nlv + 1
#    )
#    if discrete:
#        _cmap = make_listed_cmap("magma", nlv-1, extend="neither")
#        _norm = mc.BoundaryNorm(lvs, nlv-1, clip=False)
#    else:
#        _cmap = plt.get_cmap("magma")
#        _norm = mc.Normalize(vmin=lvs[0], vmax=lvs[-1])
#
#    xx, yy = np.meshgrid(data.xau, data.yau, indexing="ij")
#
#    img = plt.pcolormesh(
#        xx,
#        yy,
#        image.get_I(),
#        cmap=_cmap, #make_listed_cmap("magma", len(lvs) - 1, extend="neither"),
#        norm=_norm, #mc.BoundaryNorm(lvs, len(lvs) - 1, clip=False),
#        shading="nearest",
#        rasterized=True,
#    )
#
#    "*** Setting color bar ***"
#    cbar = plt.colorbar(img, pad=0.02)
#    Iunit = Iunit if Iunit is not None else image.Iu()
#    cbar.set_label(r"Integrated Intensity $I$" + rf" [{Iunit}]")
#
#    nticks = nlv + 1
#    cbar.set_ticks( np.linspace(lvs[0], lvs[-1], nticks) )
#    cbar.ax.minorticks_off()
#    plt.gca().tick_params("both", direction="inout")
#
#    if pangle_deg is not None:
#        for _pangle_deg in pangle_deg:
#            x, y = position_line(1400, _pangle_deg)
#            plt.gca().annotate(
#                "",
#                xy=[x[-1], y[-1]],
#                xytext=[x[0], y[0]],
#                va="center",
#                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1),
#                zorder=3
#            )
#        if poffset_au is not None:
#            pline = position_line(data.xau, pangle_deg, poffset_au=poffset_au)
#            plt.plot(pline[0], pline[1], c="w", lw=1, ls=":")
#
#    if hasattr(data, "obreso") and \
#        (data.obreso is not None) and \
#        data.obreso.beam_maj_au:
#        draw_beamsize(
#            plt.gca(),
#            "mom0",
#            data.obreso.beam_maj_au,
#            data.obreso.beam_min_au,
#            data.obreso.beam_pa_deg,
#        )
#
#    plt.xlim(xlim[0], xlim[1])
#    plt.ylim(ylim[0], ylim[1])
#    plt.xlabel(r"RA Offset [au]")
#    plt.ylabel(r"Dec Offset [au]")
#
#    draw_center_line()
#    savefig(out)

#------------------------------------------------------------------------------#
def _plot_image(
    x, y, z,
    xlim=(None, None),
    ylim=(None, None),
    clim=(None, None),
    discrete=False,
    nlv=11,
    sigma=None,
    cname="magma",
    xlabel=None, #"RA Offset [au]",
    ylabel=None, #"Dec Offset [au]",
    clabel=None,
    contour=False,
    contopt={},
    refimage=None,
):
    "*** Setting color levels and contour levels ***"
    cmin = clim[0] if clim[0] is not None else np.min(z)
    cmax = clim[1] if clim[1] is not None else np.max(z)
    if sigma is not None:
        lvs = np.arange(cmin, cmax+sigma, sigma)
    else:
        lvs = np.linspace(cmin, cmax, nlv + 1)

    cbar_ticks = np.linspace(lvs[0], lvs[-1], nlv + 1)

    "*** Setting color levels ***"
    if discrete:
        _cmap = make_listed_cmap(cname, nlv-1, extend="neither")
        _norm = mc.BoundaryNorm(lvs, nlv-1, clip=False)
    else:
        _cmap = plt.get_cmap(cname)
        _norm = mc.Normalize(vmin=lvs[0], vmax=lvs[-1])

    "*** Plotting color map using pcolormesh ***"
    xx, yy = np.meshgrid(x, y, indexing="ij")
    img = plt.pcolormesh(
        xx, yy, z, cmap=_cmap, norm=_norm,
        shading="nearest", rasterized=True,
    )

    if contour:
        _contopt = {
            "levels": lvs,
            "colors":"w",
            "linewidths":0.8,
            "zorder":2
        }
        _contopt.update(contopt)
        plt.contour(xx, yy, z, **_contopt)

    if refimage is not None:
        print(refimage)
        plt.contour(
            refimage.xau, refimage.yau, refimage.get_I().T,
            levels=_contopt["levels"],
            colors="skyblue",
            linewidths=0.5,
        #    zorder=1,
        )

    "*** Setting color bar ***"
    cbar = plt.colorbar(img, pad=0.02)
    if clabel is not None:
        cbar.set_label(clabel)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.minorticks_off()
    plt.gca().tick_params("both", direction="inout")

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_mom0_map(*args, **kwargs):
    plot_image(*args, **kwargs)

def plot_image(
    _image,
    out="image.pdf",
    xlim=(-700, 700),
    ylim=(-700, 700),
    clim=(None, None),
    xlabel="RA Offset [au]",
    ylabel="Dec Offset [au]",
    clabel=None,
    nlv=10,
    discrete=False,
    norm=None,
    brtemp=False,
    sigma=None,
    cname="magma",
    obreso=None,
    contour=False,
    contopt={},
    arrow_angle_deg=None,
    arrow_length_au=None,
    arrow_offset_au=0,
    #arrow_args=None,
    #pangles_deg=None,
    #poffset_au=None,
    refimage=None,
    save=True,
    Iunit=None,
):
    if _image.dtype != "Image":
        raise Exception("Data type error: dtype = {_image.dtype}")
    im = _image.copy()
    refim = refimage.copy() if refimage is not None else None

    if norm is not None:
        im.norm_I(norm)
        if refim:
            refim.norm_I(norm)

    if brtemp:
        im.convto_Tb()
        clabel = "Brightness Temperature [K]"
        if refim:
            refim.convto_Tb()

    if clabel is None:
        Iunit = Iunit if Iunit is not None else im.Iu()
        clabel = r"Integrated Intensity $I$" + rf" [{Iunit}]"

    pl = _plot_image(
        im.xau, im.yau, im.get_I(),
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        discrete=discrete,
        nlv=nlv,
        sigma=sigma,
        cname=cname,
        xlabel=xlabel, #"RA Offset [au]",
        ylabel=ylabel, #"Dec Offset [au]",
        clabel=clabel,
        contour=contour,
        contopt=contopt,
        refimage=refim,
    )

    if arrow_angle_deg is not None:
        if arrow_length_au is None:
           arrow_length_au = abs(xlim[1] - xlim[0])
        draw_arrows(arrow_angle_deg, arrow_length_au, arrow_offset_au)#pangles_deg, length, poffset_au)

    if (obreso is not None) and data.obreso.beam_maj_au:
        draw_beamsize(
            plt.gca(), "mom0",
            obreso.beam_maj_au,
            obreso.beam_min_au,
            obreso.beam_pa_deg,
        )

    draw_center_line()
    if save:
        savefig(out)


def draw_arrows(pangles_deg, L, offset=0):
    if np.isscalar(pangles_deg):
        pangles_deg = [pangles_deg]
    for _pangle_deg in pangles_deg:
        xy = tools.position_line(_pangle_deg, L, offset=offset)
        plt.gca().annotate(
            "",
            xy=xy[-1],
            xytext=xy[0],
            va="center",
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1, shrinkA=0, shrinkB=0),
            #arrowprops=dict(color="#aaaaaa", width=0.3, headwidth=3, headlength=3, shrink=0),
            zorder=3
        )

        if offset != 0:
            x, y = tools.position_line(_pangle_deg, L).T
            plt.plot(x, y, color="#aaaaaa", lw=1, ls=":")
#------------------------------------------------------------------------------#

#def plot_image(image,
#    n_lv=11,
#    out="image.pdf",
#    normalize=False,
#    xlim=(None, None), #(-700, 700),
#    ylim=(None, None), #(-700, 700),
#    Iunit="Tb",
#    freq0=None,
#    refimage=None,
#    sigma=None,
#    n_clv=100,
#    zmax=None,
#):
#    Ippref = None
#    if Iunit is None:
#        Ipp = image.Ipp
#        clabel = r"$I [Jy/pixel]$"
#    elif (Iunit == "Tb") and (freq0 is not None):
#        Ipp = image.convert_Jyppix_to_brightness_temperature(image.data, freq0=freq0)
#        if refimage is not None:
#            Ippref = refimage.convert_Jyppix_to_brightness_temperature(refimage.data, freq0=freq0)
#        clabel = r"$T_b [{\rm K}]$"
#    elif Iunit == "norm":
#        Ipp /= np.max(Ipp)
#        clabel = r"$I ~/ ~I_{\rm max}$"
#    else:
#        raise Exception("Failed to input data")
#
#    if zmax is None:
#        zmax = max([x for x in [np.max(Ipp), np.max(Ippref)] if x is not None])
#
#    if sigma is not None:
#        clvs = np.linspace(0, zmax, n_clv)
#        contlvs = np.arange(0, zmax, sigma)
#    else:
#        clvs = np.linspace(np.min(Ipp), zmax, n_clv)
#        contlvs = np.linspace(np.min(Ipp), zmax, n_lv)
#
#    if refimage is not None:
#        refcontlvs = contlvs
#
#    xx, yy = np.meshgrid(image.xau, image.yau, indexing="ij")
#    img = plt.pcolormesh(
#        xx, yy, Ipp,
#        cmap=make_listed_cmap("magma", len(clvs) - 1, extend="neither"),
#        norm=mc.BoundaryNorm(clvs, len(clvs) - 1, clip=False),
#        shading="nearest",
#        rasterized=True,
#        zorder=0,
#    )
#    plt.contour(
#        xx, yy, Ipp,
#        levels=contlvs,
#        colors="w",
#        linewidths=0.8,
#        zorder=2,
#    )
#
#    plt.xlim(xlim[0], xlim[1])
#    plt.ylim(ylim[0], ylim[1])
#    plt.xlabel(r"$\Delta$RA [au]")
#    plt.ylabel(r"$\Delta$Dec [au]")
#    cbar = plt.colorbar(img, pad=0.02)
#    cbar.set_label(clabel)
#
#    ax = plt.gca()
#    draw_center_line()
#    ax.tick_params("both", direction="inout")
#
#    if (image.obreso is not None) and image.obreso.beam_maj_au:
#        draw_beamsize(
#            ax,
#            "mom0",
#            image.obreso.beam_maj_au,
#            image.obreso.beam_min_au,
#            image.obreso.beam_pa_deg,
#        )
#
#    if refimage is not None:
#        plt.contour(
#            xx, yy, Ippref,
#            levels=refcontlvs,
#            colors="skyblue",
#            linewidths=0.5,
#            zorder=1,
#        )
#
#    savefig(out)
#

#------------------------------------------------------------------------------#

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


#------------------------------------------------------------------------------#

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
    figsize=None,
    xlim=(-700, 700),
    ylim=(-2.5, 2.5),
    clim=(None, None),
    show_beam=True,
    discrete=False, # True,
    contour=True,
    refpv=None,
    loglog=False,
    Iunit=None,
    norm=None,
    smooth_contour=False,
    out="pvdiagrams.pdf",
):
    print(pv)
    pv = pv.copy()
    if norm:
        pv.norm_I(norm)
    Ipv = pv.Ipv
    xau = pv.xau
    vkms = pv.vkms

    nlv_c = n_lv
    plt.figure(figsize=figsize)
    lvs = np.linspace(
        clim[0] if clim[0] is not None else np.min(Ipv),
        clim[1] if clim[1] is not None else np.max(Ipv),
        nlv_c + 1
    )
    nlvs = len(lvs)

    xx, yy = np.meshgrid(xau, vkms, indexing="ij")

    if discrete:
        _cmap = make_listed_cmap("cividis", nlv_c, extend="neither")
        _norm = mc.BoundaryNorm(lvs, nlv_c, clip=False)
    else:
        _cmap = plt.get_cmap("cividis")
        _norm = mc.Normalize(vmin=lvs[0], vmax=lvs[-1])

    print("pvdshape:" , xx.shape, yy.shape, Ipv.shape)

    img = plt.pcolormesh(
        xx,
        yy,
        Ipv,
        cmap=_cmap,
        norm=_norm,
        shading="nearest",
        rasterized=True,
        alpha=.9
    )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(r"Offset from Center $x$ [au]")
    plt.ylabel(r"Line-of-sight Velocity $V$ [km s$^{-1}$]")

    if loglog:
        plt.xlim(30, 1000)
        plt.ylim(0.2, 3)
        plt.xscale("log")
        plt.yscale("log")
        #plt.gca().yaxis.set_minor_locator(mt.LogLocator())
        #cbar.ax.minorticks_off()
        #plt.gca().yaxis.set_minor_formatter(mt.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.4) ))
        #plt.gca().yaxis.set_minor_locator(mt.LogLocator(subs=[1,2,5])) #set the ticks position
        plt.gca().yaxis.set_minor_formatter(mt.FormatStrFormatter("%g"))  #add the custom ticks

        x = np.geomspace(10, 1000)
        func1 = lambda x: 0.036 * (x / 1e4) ** (-1)
        plt.plot(x, func1(x), c="w", ls="--", lw=2, zorder=10)
        plt.text(400 * 1.1, func1(400) * 1.1, r"$x^{-1} $", c="w", size=20, zorder=10)
        func2 = lambda x: 0.1 * (x / 1e4) ** (-0.5)
        plt.plot(x, func2(x), c="w", ls=":", lw=2, zorder=10)
        plt.text(50 * 1.1, func2(50) * 1.1, r"$x^{-0.5}$", c="w", size=20, zorder=10)
        if 0:
            xx, vv = np.meshgrid(xau, vkms, indexing="ij")
            vpeaks = np.array(
                [get_maximum_position(vkms, Iv) for Iv in np.where(vv > 0, Ipv, 0)]
            )
            plt.scatter(xau[vpeaks > 0], vpeaks[vpeaks > 0], c="red", marker="o", s=4)
        show_beam = False

    cbar = plt.colorbar(img, pad=0.02)

    #cbar.set_label(r"$I_{V}$"+" ~/ ~" + "$I_{V,\rm max}$")
    print(Iunit)
    Iunit = Iunit if Iunit is not None else pv.Iu()
    print(Iunit)
    cbar.set_label(r"Intensity $I_{V}$" + rf" [{Iunit}]")
    ticks = np.linspace(lvs[0], lvs[-1], n_lv +1)
    cbar.set_ticks(ticks)
    #cbar.set_ticklabels(ticks)
    cbar.ax.minorticks_off()
    cbar.ax.yaxis.set_major_formatter(mt.FormatStrFormatter('%.2f'))
    ax = plt.gca()
    draw_center_line()
    # ax.minorticks_on()
    # ax.tick_params("both", direction="inout")
    ax.tick_params(direction="inout")

    if contour:
        print(xx.shape, yy.shape, Ipv.shape)
        if smooth_contour:
            _xx, _yy, _Ipv = smooth_image(xx, yy, Ipv)
        else:
            _xx, _yy, _Ipv = xx, yy, Ipv

        contlvs = np.array([0.3, 0.5, 0.7, 0.9])
        c = "k"#, "dimgray"
        img = plt.contour(
            _xx,
            _yy,
            _Ipv,
            contlvs*(lvs[-1] - lvs[0]) + lvs[0],
            colors=c, # make_listed_cmap("cividis", n_lv, extend="neither"),
            norm=mc.Normalize(vmin=0, vmax=np.max(Ipv)),  # mc.BoundaryNorm(lvs, n_lv, clip=False),
            corner_mask=False,
            linewidths=1.3,
            linestyles="-",
            alpha=0.6,
            zorder=4,
        )
        x_ipeak, v_ipeak = get_coord_ipeak(xau, vkms, Ipv, mode="max")
        plt.scatter(x_ipeak, v_ipeak, s=15, alpha=0.9, linewidth=1, c=c, ec=None,zorder=4)

    if refpv:
        if smooth_contour:
            _xx, _yy, _Ipv = smooth_image(refpv.xau, refpv.vkms, refpv.Ipv, axes="ax")
        else:
            _xx, _yy = np.meshgrid(refpv.xau, refpv.vkms, indexing="ij")
            _Ipv = refpv.Ipv
        contlvs = np.array([0.3, 0.5, 0.7, 0.9])
        x_ipeak, v_ipeak = get_coord_ipeak(refpv.xau, refpv.vkms, refpv.Ipv, mode="max")
        # x_vmax, v_vmax = get_coord_vmax(refpv.xau, refpv.vkms, refpv.Ipv, f_crit)
        c="#90D0FF" #"lightsalmon"#"mistyrose"
        print(_xx.shape, _yy.shape, _Ipv.shape)
        plt.contour(
            _xx,
            _yy,
            _Ipv.clip(1e-10),
            contlvs*(lvs[-1] - lvs[0]) + lvs[0],
            colors=c,
            linewidths=.8,
            linestyles="-",
            alpha=.9,
            norm=mc.Normalize(vmin=0, vmax=np.max(_Ipv)),
            corner_mask=False,
            zorder=5,
        )
        plt.scatter(x_ipeak, v_ipeak, s=12, alpha=1.0, linewidth=1, c=c, fc=c, zorder=5)

    if mass_estimate:
        add_mass_estimate_plot(
            xau,
            vkms,
            Ipv,
            Ms_Msun=Ms_Msun,
            rCR_au=rCR_au,
            f_crit=f_crit,
            incl=incl,
            mass_ip=True,
            mass_vp=True,
            quadrant=quadrant,
        )

    if hasattr(pv, "obreso") and \
        (pv.obreso is not None) and \
        pv.obreso.beam_maj_au and \
        not loglog :
        draw_beamsize(
            plt.gca(),
            "pv",
            pv.obreso.beam_maj_au,
            pv.obreso.beam_min_au,
            pv.obreso.beam_pa_deg,
            pv.pangle_deg,
            pv.obreso.vreso_kms,
        )

    if subax:
        ax2 = ax.twiny()
        # ax2.minorticks_on()
        ax2.tick_params("both", direction="inout")
        ax2.set_xlabel("Angular Offset [arcsec]")
        # ax2.set_xlim([xas[0], xas[-1]])
        ax2.set_xlim(np.array(ax.get_xlim()) / pv.dpc)

    savefig(out)

def smooth_image(xx, yy, image, axes="mg"):
    _x, _y = (xx[:,0], yy[0,:]) if axes == "mg" else (xx, yy)
    _newx = np.linspace(np.min(_x), np.max(_x), len(_x)*4)
    _newy = np.linspace(np.min(_y), np.max(_y), len(_y)*4)
    _xx, _yy =  np.meshgrid(_newx, _newy, indexing="ij")
    newgrid = np.stack([_xx, _yy], axis=-1)
    _image = interpolate.interpn((_x, _y), image, newgrid, method="splinef2d")
    print(xx.shape, yy.shape, image.shape, _xx.shape, _yy.shape, newgrid.shape, _image.shape)
    return _xx, _yy, _image

#(644, 640) (161, 160) (644, 640, 2) (644, 640)
#(2264, 640) (566, 160) (2264, 640, 2) (2264, 640)

