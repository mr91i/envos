import os
import numpy as np
from scipy import interpolate, integrate, optimize
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as mc

from . import gpath
from . import nconst as nc
from . import log
from . import streamline
from . import tools
# from myplot import mpl_setting, color

logger = log.set_logger(__name__)
matplotlib.use("Agg")
# os.makedirs(gpath.fig_dir, exist_ok=True)
color_def = ["#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#34495e",
         "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50"]

"""
Plotting functions
"""


def plot_density_map(
    model,
    rlim=700,
    streams=False, #True,
    trajectries=False, #True,
):
    lvs = np.linspace(-19, -16, 10)[2:]
    img = plt.pcolormesh(
        model.R[:, :, 0] / nc.au,
        model.z[:, :, 0] / nc.au,
        np.log10(model.rhogas[:, :, 0].clip(1e-300)),
        cmap=make_listed_cmap("viridis", len(lvs)-1),
        norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
        shading="nearest",
        rasterized=True,
    )
    plt.xlim(0, rlim)
    plt.ylim(0, rlim)
    plt.xlabel("R [au]")
    plt.ylabel("z [au]")
    cbar = plt.colorbar(img, format="%.1f", extend="both", pad=0.02)
    cbar.set_label(r"Log Gas Density [g cm$^{-3}$]")
    cbar.ax.minorticks_off()

    if trajectries:
        add_trajectries(model)

    if streams:
        add_streams(model, rlim)

    savefig("density.pdf")


def plot_midplane_numberdensity_profile(km):
    plt.plot(km.rc_ax, km.rho[:, -1, 0] / km.meanmolw)
    plt.xlim(10, 1000)
    plt.ylim(10, 1000)
    plt.xscale("log")
    plt.yscale("log")
    savefig("ndens_prof.pdf")

def plot_midplane_velocity_profile(model):
    plt.plot(model.rc_ax/nc.au, -model.vr[:, -1, 0]/1e5, label=r"$- v_r$", ls="-")
    plt.plot(model.rc_ax/nc.au,  model.vt[:, -1, 0]/1e5, label=r"$v_{\theta}$", ls=":")
    plt.plot(model.rc_ax/nc.au,  model.vp[:, -1, 0]/1e5, label=r"$v_{\phi}$", ls="--", )
    plt.xlim(0, 300)
    plt.ylim(0, 3)
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Velocity [km s$^{-1}$]")
    plt.legend()
    savefig("v_prof.pdf")

def plot_midplane_velocity_map(model, rlim=600):
    rax = model.rc_ax
    tax = model.tc_ax
    pax = model.pc_ax if len(model.pc_ax) != 1 else np.linspace(-np.pi, np.pi, 91)
    #pax = np.linspace(-np.pi, np.pi, 91)
    rr, tt, pp = np.meshgrid(rax, tax, pax, indexing='ij')
    R, z = rr * [np.sin(tt), np.cos(tt)]
    x, y = R * [np.cos(pp), np.sin(pp)]
    #vx = model.vr * np.cos(pp) - model.vp * np.sin(pp)
    #vy = model.vr * np.sin(pp) + model.vp * np.cos(pp)

    vls = x/rr * model.vp + y/rr*model.vr

    lvs = np.linspace(-1.7, 1.7, 18)
    #img  = plt.contourf(x[:,-1,:]/nc.au, y[:,-1,:]/nc.au, vls[:,-1,:]/1e5, lvs,  cmap=plt.get_cmap('seismic'))
#    img  = plt.contourf(x[:,-1,:]/nc.au, y[:,-1,:]/nc.au, vls[:,-1,:]/1e5)
    img  = plt.tricontourf(x[:,-1,:].flatten()/nc.au, y[:,-1,:].flatten()/nc.au, vls[:,-1,:].flatten()/1e5, lvs,  cmap=plt.get_cmap('RdBu_r'), extend="both", )

    #img  = plt.contourf(x[:,-1,:]/nc.au, y[:,-1,:]/nc.au, vls[:,-1,:]/1e5, lvs,  cmap=plt.get_cmap('RdBu_r'))
    #img  = plt.scatter(x[:,-1,:]/nc.au, y[:,-1,:]/nc.au, c=vls[:,-1,:]/1e5, cmap=plt.get_cmap('RdBu_r'), alpha=0.5)
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim, rlim)
    plt.xlabel("x [au]")
    plt.ylabel("Distance along Line-of-Sight [au]")
    cbar = plt.colorbar(img, ticks= np.linspace(-2, 2, 11) , pad=0.02)
    cbar.set_label(r'$V_{\rm LOS}$ [km s$^{-1}$]')
    cbar.ax.minorticks_off()
    savefig("v_los.pdf")




def plot_temperature_map(
    m,
    rlim=700,
    streams=False, #True,
    trajectries=False, #True,
):
    lvs = np.linspace(10, 100, 10)
    cmap = plt.get_cmap("inferno", len(lvs))
    norm = matplotlib.colors.BoundaryNorm(lvs, len(lvs))
    img = plt.pcolormesh(
        m.R[:, :, 0] / nc.au,
        m.z[:, :, 0] / nc.au,
        m.Tgas[:, :, 0],
        cmap=make_listed_cmap("inferno", len(lvs)-1),
        norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
        shading="nearest", rasterized=True,
    )
   # cont = plt.contour(
   #     m.R[:, :, 0] / nc.au,
   #     m.z[:, :, 0] / nc.au,
   #     m.Tgas[:, :, 0],
   #     colors="w",
   #     levels=lvs,
   #     linewidths=1.5,
   #     linestyles="dashed"
    #)
    #cont.clabel(fmt='%1.0f K', fontsize=10)

    plt.xlim(0, rlim)
    plt.ylim(0, rlim)
    plt.xlabel("R [au]")
    plt.ylabel("z [au]")
    cbar = plt.colorbar(img, extend="both", pad=0.02)
    cbar.set_label(r"Gas Temperature [K]")
    cbar.ax.minorticks_off()
    # cbar.ax.yaxis.set_major_formatter(mt.ScalarFormatter())
    # cbar.ax.yaxis.set_minor_formatter(mt.ScalarFormatter())

    if trajectries:
        add_trajectries(m)

    if streams:
        add_streams(m, rlim)

    savefig("gtemp.pdf")


def plot_opacity():
    wav, kabs, kscat = np.loadtxt("./run/radmc/dustkappa_MRN20.inp",  skiprows=3, unpack=True)
    plt.plot(wav, kabs, c="k", lw=3)
    ax = plt.gca()
    ax.minorticks_on()
    plt.xlim(0.1, 1e5)
    plt.ylim(1e-4, 1e6)
    plt.xscale("log")
    plt.yscale("log")
    ax.yaxis.set_minor_locator(mt.LogLocator(numticks=15))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())  #add the custom ticks

    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel(r"Absorption Opacity [cm$^{2}$ g$^{-1}$]")
    savefig("opac.pdf")

def plot_mom0_map(
    obsdata,
    pangle_deg=None,
    poffset_au=None,
    n_lv=100,
):
    def position_line(length, pangle_deg, poffset_au=0):
        line = np.linspace(-length/2, length/2, 10)
        pangle_rad = (pangle_deg + 90)* np.pi / 180
        pos_x = line * np.cos(pangle_rad) - poffset_au * np.sin(pangle_rad)
        pos_y = line * np.sin(pangle_rad) + poffset_au * np.sin(pangle_rad)
        return pos_x, pos_y
        #return np.stack([pos_x, pos_y], axis=-1)
    #Ipp = integrate.simps(obsdata.Ippv, obsdata.vkms, axis=2)
    Ipp = np.sum(obsdata.Ippv, axis=2) * (obsdata.vkms[1] - obsdata.vkms[0])
    Ipp /= np.max(Ipp)
    lvs = np.linspace(0, np.max(Ipp), 11)
    plt.figure(figsize=(8,6))
    img = plt.pcolormesh(
        obsdata.xau, obsdata.yau , Ipp.T,
        cmap=make_listed_cmap("magma", len(lvs)-1, extend="neither"),
        norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
        shading="nearest", rasterized=True
    )
    plt.xlim(-700, 700)
    plt.ylim(-700, 700)
    plt.xlabel("x [au]")
    plt.ylabel("y [au]")
    cbar = plt.colorbar(img, pad=0.02)
    cbar.set_label(r"Normalized Integrated Intesity")

    ax = plt.gca()
    draw_center_line()
    ax.tick_params("both", direction="inout")

    if pangle_deg is not None:

        for _pangle_deg in pangle_deg:
            x, y = position_line(1400, _pangle_deg)
            plt.plot(x, y, ls="--", c="w", lw=1.5)

        if poffset_au is not None:
            pline = position_line(
                obsdata.xau, pangle_deg, poffset_au=poffset_au
            )
            plt.plot(pline[0], pline[1], c="w", lw=1)

    print(obsdata.convolve, obsdata.beam_maj_au)
    if obsdata.convolve and obsdata.beam_maj_au:
        draw_beamsize(
            ax,
            "mom0",
            obsdata.beam_maj_au,
            obsdata.beam_min_au,
            obsdata.beam_pa_deg,
        )

    savefig("mom0map.pdf")


def plot_lineprofile(obsdata):
    lp = integrate.simps(
        integrate.simps(obsdata.Ippv, obsdata.xau, axis=2), obsdata.yau, axis=1
    )
    plt.plot(obsdata.vkms, lp)

    filepath = os.path.join(gpath.fig_dir, "line.pdf")
    print("saved ", filepath)
    plt.savefig(filepath)
    plt.clf()


def plot_pvdiagram(
    PV,
    n_lv=5,
    Ms_Msun=None,
    rCR_au=None,
    incl=90,
    f_crit=0.1,
    mass_estimate=False,
    mass_ip=False,
    mass_vp=False,
    mapmode="grid",
    oplot={},
    subax=False,
    figsize=(8,6),
    xlim=(-700, 700),
    ylim=(-2, 2)
):
    print("hi")

    Ipv = PV.Ipv
    xau = PV.xau
    # xas = xau / PV.dpc
    vkms = PV.vkms

    plt.figure(figsize=figsize)
    lvs = np.linspace(np.min(Ipv), np.max(Ipv), 11)
    img = plt.pcolormesh(xau, vkms, Ipv,
        cmap=make_listed_cmap("cividis", len(lvs)-1, extend="neither"),
        norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
        shading="nearest", rasterized=True)

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Position [au]")
    plt.ylabel(r"Line-of-Sight Velocity [km s$^{-1}$]")
    cbar = plt.colorbar(img, pad=0.02)
    cbar.set_label(r"Intensity [$I_{\rm max}$]")

    ax = plt.gca()
    draw_center_line()
    # ax.minorticks_on()
    #ax.tick_params("both", direction="inout")
    ax.tick_params(direction="inout")

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
        mass_vp=True)

    if PV.convolve:
        draw_beamsize(
            plt.gca(),
            "PV",
            PV.beam_maj_au,
            PV.beam_min_au,
            PV.beam_pa_deg,
            PV.pangle_deg,
            PV.vreso_kms,
        )

    if subax:
        ax2 = ax.twiny()
        # ax2.minorticks_on()
        ax2.tick_params("both", direction="inout")
        ax2.set_xlabel("Angular Offset [arcsec]")
        # ax2.set_xlim([xas[0], xas[-1]])
        ax2.set_xlim(np.array(ax.get_xlim()) / PV.dpc)

    savefig("pvdiagram.pdf")


"""
plotting tools
"""
def add_colorbar(label=None, fmt=None, extend="both", pad=0.02, minticks=False):
    cbar = plt.colorbar(img, format=fmt, extend="both", pad=pad)
    cbar.set_label(label)
    if minticks:
        cbar.ax.minorticks_off()


def make_listed_cmap(cmap_name, ncolors, extend="both"):
    if extend=="both":
        io = 1
        iu = 1
    elif extend=="neither":
        io = 0
        iu = 0
    cmap = plt.get_cmap(cmap_name, ncolors + io + iu)
    colors = cmap.colors
    clist = colors[1 if iu else 0: -1 if io else None]
    lcmap = mc.ListedColormap(clist)
    #lcmap.set_under(colors[0])
    #lcmap.set_over(colors[-1])
    return lcmap

def add_streams(km, rlim, r0=None):
    r0 = r0 or rlim
    rau = np.linspace(0, r0, 1000)
    xx, yy = np.meshgrid(rau, rau)
    newgrid = np.stack(
        [np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)], axis=-1
    )
    vR = interpolate.interpn(
        (km.rc_ax / nc.au, km.tc_ax),
        km.vR[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=None,
    )
    vz = interpolate.interpn(
        (km.rc_ax / nc.au, km.tc_ax),
        km.vz[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=None,
    )
    start_points = [
        (r0 * np.sin(th0), r0 * np.cos(th0))
        for th0 in np.radians(np.linspace(0, 90, 19))
    ]
    opt = {"density": 5, "linewidth": 0.25, "color": "w", "arrowsize": 0.6}
    #opt={}
    #rau = np.linspace(0, rlim, 1000)
    plt.streamplot(rau, rau, vR, vz, start_points=start_points, **opt)


def add_trajectries(km):
    r0 = km.ppar.cs * km.ppar.t  # / nc.au
    start_points = [(r0, th0) for th0 in np.radians(np.linspace(0, 90, 10))]
    sls = streamline.calc_streamlines(
        km.rc_ax,
        km.tc_ax,
        km.vr[:, :, 0],
        km.vt[:, :, 0],
        start_points,
        method="RK23",
    )
    for sl in sls:
        plt.plot(
            sl.R / nc.au, sl.z / nc.au, c="orange", lw=1.0, marker=".", ms=5
        )
    streamline.save_data(sls)


def vertical_integral(value_rt, R_rt, z_rt, R_ax, z_ax, log=False):
    points = np.stack((R_rt.flatten(), z_rt.flatten()), axis=-1)
    npoints = np.stack(np.meshgrid(R_ax, z_ax), axis=-1)
    if log:
        fltr = np.logical_and.reduce(
            (
                [np.all(np.isfinite(a)) for a in np.log10(points)],
                np.isfinite(np.log10(value_rt)),
            )
        )
        fltr = fltr.flatten()
        v = value_rt.flatten()
        ret = 10 ** interpolate.griddata(
            np.log10(points[fltr]),
            np.log10(v[fltr]),
            np.log10(npoints),
            method="linear",
        )
    else:
        ret = interpolate.griddata(
            points, value_rt.flatten(), npoints, method="linear"
        )
    s = np.array([integrate.simps(r, z_ax) for r in np.nan_to_num(ret)])
    return s


def draw_center_line():
    draw_cross_pointer(0, 0, c="k", lw=1, ls="-", s=0, alpha=0.5, zorder=1)

def draw_cross_pointer(x, y, c="k", lw=2, ls=":", s=10, alpha=1, zorder=1):
    plt.axhline(y=y, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    plt.axvline(x=x, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    plt.scatter(x, y, c=c, s=s, alpha=alpha, linewidth=0, zorder=zorder)


def draw_beamsize(
    ax,
    mode,
    beam_maj_au,
    beam_min_au,
    beam_pa_deg,
    pangle_deg=None,
    vreso_kms=None,
    with_box=False,
):
    print(beam_maj_au, beam_min_au, vreso_kms)

    if mode == "PV":
        cross_angle = (beam_pa_deg - pangle_deg) * np.pi / 180
        beam_crosslength_au = 1 / np.sqrt(
            (np.cos(cross_angle) / beam_maj_au) ** 2
            + (np.sin(cross_angle) / beam_min_au) ** 2
        )
        beamx = beam_crosslength_au
        beamy = vreso_kms
    elif mode == "mom0":
        beamx = beam_maj_au
        beamy = beam_min_au

    if with_box:
        e1 = matplotlib.patches.Ellipse(
            xy=(0, 0),
            width=beamx,
            height=beamy,
            lw=1,
            fill=True,
            ec="k",
            fc="0.6",
        )
    else:
        e1 = matplotlib.patches.Ellipse(
            xy=(0, 0),
            width=beamx,
            height=beamy,
            lw=0,
            fill=True,
            ec="k",
            fc="0.7",
            alpha=0.6,
        )

    box = AnchoredAuxTransformBox(
        ax.transData,
        loc="lower left",
        frameon=with_box,
        pad=0.0,
        borderpad=0.4,
    )
    box.drawing_area.add_artist(e1)
    box.patch.set_linewidth(1)
    ax.add_artist(box)

def add_mass_estimate_plot(
    xau, vkms, Ipv,
    Ms_Msun,
    rCR_au,
    incl=90,
    f_crit=None,
    mass_ip=False,
    mass_vp=False,
):
    from skimage.feature import peak_local_max

    overplot = {
        "KeplerRotation": False,
        "ire_fit": False,
        "LocalPeak_Pax": False,
        "LocalPeak_Vax": False,
        "LocalPeak_2D": False,
    }

    def find_local_peak_position(x, y, i):
        if 2 <= i <= len(x) - 3:
            grad_y = interpolate.InterpolatedUnivariateSpline(
                x[i - 2 : i + 3], y[i - 2 : i + 3]
            ).derivative(1)
            return optimize.root(grad_y, x[i]).x[0]
        else:
            return np.nan

    def get_localpeak_positions(x, y, min_distance=3, threshold_abs=None):
        maxis = [
            find_local_peak_position(x, y, mi)
            for mi in peak_local_max(
                y, min_distance=min_distance, threshold_abs=threshold_abs
            )[:, 0]
        ]
        return np.array(maxis)

    def get_maximum_position(x, y):
        return find_local_peak_position(x, y, np.argmax(y))


    overplot = {
        "LocalPeak_Pax": False,
        "LocalPeak_Vax": False,
        "LocalPeak_2D": False,
    }

    if overplot["LocalPeak_Pax"]:
        for Iv, v_ in zip(Ipv.transpose(0, 1), vkms):
            for xM in get_localpeak_positions(
                xau, Iv, threshold_abs=np.max(Ipv) * 1e-10
            ):
                plt.plot(xM, v_, c="red", markersize=1, marker="o")

    if overplot["LocalPeak_Vax"]:
        for Ip, x_ in zip(Ipv.transpose(1, 0), xau):
            for vM in get_localpeak_positions(
                vkms, Ip, threshold_abs=np.max(Ipv) * 1e-10
            ):
                plt.plot(x_, vM, c="blue", markersize=1, marker="o")

    if overplot["LocalPeak_2D"]:
        for jM, iM in peak_local_max(
            Ipv, num_peaks=4, min_distance=10
        ):  #  min_distance=None):
            plt.scatter(xau[iM], vkms[jM], c="k", s=20, zorder=10)
            print(
                f"Local Max:   {xau[iM]:.1f} au  , {vkms[jM]:.1f}    km/s  ({jM}, {iM})"
            )

        del jM, iM

    def calc_M(xau, vkms, fac=1):
        # calc xau*nc.au * (vkms*nc.kms)**2 / (nc.G*nc.Msun)
        return 0.001127 * xau * vkms ** 2 * fac

    if mass_ip:
        ## M_ipeak
        jmax, imax = Ipv.shape
        i0 = imax // 2
        j0 = jmax // 2
        jpeak, ipeak = peak_local_max(Ipv[j0:, i0:], num_peaks=1)[0]

        xau_peak = xau[i0+ipeak]
        vkms_peak = vkms[j0+jpeak]


        draw_cross_pointer(xau_peak, vkms_peak, color_def[1], lw=1.5, s=18, ls=":")
        M_CR = calc_M(abs(xau_peak), vkms_peak/np.sin(np.deg2rad(incl)), fac=1)
        txt_Mip = rf"$M_{{\rm ipeak}}$={M_CR:.3f}"

        logger.info("Mass estimation with intensity peak:")
        logger.info(f" x_ipeak = {xau_peak} au")
        logger.info(f" V_ipeak = {vkms_peak} km/s")
        logger.info(f" V_ipeak/sin i = {vkms_peak/np.sin(np.deg2rad(incl))} km/s")
        logger.info(f" M_ipeak = {M_CR} Msun")

    if mass_vp:
        ## M_vpeak
        x_vmax, I_vmax = np.array([
                [get_maximum_position(xau, Iv), np.max(Iv)]
                for Iv in Ipv.transpose(0, 1)
            ]).T
        v_crit = tools.find_roots(vkms, I_vmax, f_crit * np.max(Ipv))
        if len(v_crit) == 0:
            v_crit = vkms[0]
        else:
            v_crit = v_crit[-1]

        x_crit = tools.find_roots(x_vmax, vkms, v_crit)[-1]
        M_CB = calc_M(abs(x_crit), v_crit/np.sin(np.deg2rad(incl)) , fac=1 / 2)
        draw_cross_pointer(x_crit, v_crit, color_def[0], lw=1.5, s=18, ls=":")
        txt_Mvp = rf"$M_{{\rm vmax,\,{f_crit*100:.0f}\%}}$={M_CB:.3f}"

        logger.info("Mass estimation with maximum velocity:")
        logger.info(f" x_vmax = {x_crit} au")
        logger.info(f" V_vmax = {v_crit} km/s")
        logger.info(f" V_vmax/sin(i) = {v_crit/np.sin(np.deg2rad(incl))} km/s")
        logger.info(f" M_vmax = {M_CB} Msun")

    if mass_ip or mass_vp:
        plt.text(
            0.95,
            0.05,
            txt_Mip + "\n" + txt_Mvp,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(fc="white", ec="black", pad=5),
        )


def savefig(filename):
    gpath.make_dirs(fig=gpath.fig_dir)
    filepath = os.path.join(gpath.fig_dir, filename)
    print("saved ", filepath)
    plt.savefig(filepath)
    plt.clf()


