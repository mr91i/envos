import os
import numpy as np
import dataclasses
from scipy import integrate, optimize

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as mc
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox

from .plot_funcs import *
from .. import nconst as nc
from .. import log
from .. import streamline
from .. import tools

figext = "pdf"

from cycler import cycler
cyclestyle = cycler(ls=["-", "--", ":", "-."])
logger = log.logger

## Not use easyplot
## - it make the code complicated, though useful
def set_plot_format(
    xlim=None, 
    ylim=None, 
    xlb=None, 
    ylb=None, 
    legend=False, 
    xlog=False, 
    ylog=False, 
    loglog=False
):
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlb is not None:
        plt.xlabel(xlb)
    if ylb is not None:
        plt.ylabel(ylb)
    if legend:
        plt.legend()
    if xlog or loglog:
        plt.xscale("log")
    if ylog or loglog:
        plt.yscale("log")

## New generation plotting format
## - make a plotting format independent of variables
## 
def plot_variable_meridional_map(
    model,
    variable_name,
    save_name=None,
    xlim=500,
    rinlim=0,
    clabel="",
    clog=False,
    save=True,
    streams=False,
    streams_option={},
    trajectories=False,
    trajectories_option={},
    dlv=None,
    aspect="equal",
    extend="min",
    **kwargs,
):
    if xlim > np.max(model.rr[...,0]) / nc.au:
        xlim = np.max(model.rr[...,0]) / nc.au

    var_toridal_average = np.average(getattr(model, variable_name), axis=2)

    plot_colormap(
        model.R[..., 0] / nc.au,
        model.z[..., 0] / nc.au,
        var_toridal_average, 
        #zregion=(modepl.R[...,0] < xlim * nc.au) & (model.z[...,0] < xlim * nc.au),
        zregion= (model.R[...,0] < xlim * nc.au) & (model.z[...,0] < xlim * nc.au) & (model.rr[...,0] > rinlim * nc.au),
        clabel=clabel, 
        clog=clog,
        dlv=dlv,
        aspect=aspect,
        extend=extend,
        **kwargs,
    )

    if trajectories:
        _trj_option = {"r0_au": xlim, "theta0_deg": 30}  
        _trj_option.update(trajectories_option)
        add_trajectories(model, **_trj_option)

    if streams:
        add_streams(
            model, xlim, r0=np.sqrt(2)*xlim, 
            use_mu0=hasattr(model, "mu0"),
            cavity_region=(model.rhogas[...,0]==0)
        )

    set_plot_format(
        xlim=(0, xlim), 
        ylim=(0, xlim), 
        xlb=r"$R$ [au]", 
        ylb=r"$z$ [au]"
    )
 
    if save:
        name = variable_name if save_name is None else save_name
        savefig(name + "." + figext)
    return

# example of gas density
def plot_rhogas_map(
    model,
    **kwargs
):
    _kwargs = {
        "clabel": rf"Gas Density [g cm$^{{-3}}$]",
        "clog": True,
        "cname": "viridis",
        "dlv": 0.2,
    }
    _kwargs.update(kwargs)
    plot_variable_meridional_map(model,"rhogas",**_kwargs)

# example of gas temperature
def plot_Tgas_map(
    model,
    **kwargs
):
    _kwargs = {
        "clabel": r"Gas Temperature [K]",
        "rinlim": 20,
        "cname": "inferno",
        "dlv": 10,
        "extend": "neither",
    }
    _kwargs.update(kwargs)
    plot_variable_meridional_map(model,"Tgas",**_kwargs)




##################################################################################################
#                                                                                                #
#                                          L E G A C Y                                           #
#                                                                                                #
##################################################################################################

@dataclasses.dataclass
class PlotInfo:
    name: str
    xlb: str = None
    ylb: str = None
    xlim: tuple = None
    ylim: tuple = None
    xlog: bool = False
    ylog: bool = False
    loglog: bool = False
    legend: bool = False

def easyplot(func):
    def wrapper(*args, **kwargs):
        save = True
        if ("nosave" in kwargs) and (kwargs["nosave"]):
            save = False
            del kwargs["nosave"]
        pi = func(*args, **kwargs)
        if pi.xlim is not None:
            plt.xlim(*pi.xlim)
        if pi.ylim is not None:
            plt.ylim(*pi.ylim)
        plt.xlabel(pi.xlb)
        plt.ylabel(pi.ylb)
        if pi.legend:
            plt.legend()
        if pi.xlog or pi.loglog:
            plt.xscale("log")
        if pi.ylog or pi.loglog:
            plt.yscale("log")

        plt.draw()
        if save:
            savefig(pi.name + "." + figext)

    return wrapper

########################

@easyplot
def plot_density_map(
    m, # model
    xlim=500,
    r0=None,
    streams=False,
    trajectories=False,
    trajectories_option={},
    dloglv=0.25,
):
    if xlim > np.max(m.rr[...,0]) / nc.au:
        xlim = np.max(m.rr[...,0]) / nc.au

    rho = np.average(m.rhogas, axis=2)
    #ex = np.floor( np.log10( np.abs( np.max(rho) ) ) )
    logger.debug(f"Maximum density is {np.max(rho)}, minimum density is {np.min(rho)}" )
    plot_colormap(
        m.R[..., 0] / nc.au,
        m.z[..., 0] / nc.au,
        rho, # / 10**ex,
        zregion= (m.R[...,0] < xlim * nc.au) & (m.z[...,0] < xlim * nc.au),
        #clabel=rf"Gas Density [10$^{{{ex:.0f}}}$ g cm$^{{-3}}$]",
        clabel=rf"Gas Density [g cm$^{{-3}}$]",
        clog=True,
        dlv=0.2,
        aspect="equal",
        extend="min",
        # cformat="%.3g"
    )
    if trajectories:
        trj_option = {"r0_au": xlim, "theta0_deg": 30}  
        trj_option.update(trajectories_option)
        add_trajectories(m, **trj_option)

    if streams:
        add_streams(m, xlim, r0=r0, use_mu0=hasattr(m, "mu0"),
        cavity_region=(m.rhogas[...,0]==0))

    return PlotInfo("density", r"$R$ [au]", r"$z$ [au]", (0, xlim), (0, xlim))

@easyplot
def plot_temperature_map(
    m, # model
    xlim=500,
    r0=None,
    streams=False,  # True,
    trajectories=False,  # True,
    trajectories_option={},
    rinlim=20,
):
    if xlim > np.max(m.rr[...,0]) / nc.au:
        xlim = np.max(m.rr[...,0]) / nc.au
    #lvs = np.arange(
    #    10 * np.floor(1 / 10 * np.min(m.Tgas)),
    #    np.max(m.Tgas[m.rr > rinlim * nc.au])/10 + 1e-10,
    #    10
    #)
    #logger.debug("Level is", lvs, np.average(m.Tgas, axis=2) )
    plot_colormap(
        m.R[..., 0] / nc.au,
        m.z[..., 0] / nc.au,
        np.average(m.Tgas, axis=2),
        zregion= (m.R[...,0] < xlim * nc.au) & (m.z[...,0] < xlim * nc.au) & (m.rr[...,0] > rinlim * nc.au),
        #zregion= (m.rr[...,0] < xlim * nc.au),
        clabel=r"Gas Temperature [K]",
        clog=False,
        aspect="equal",
        cname="inferno",
        dlv=10,
        #lvs = lvs,
        extend="neither"
    )
    if trajectories:
        add_trajectories(m, **trajectories_option)
    if streams:
        add_streams(m, xlim, r0=r0, use_mu0=hasattr(m, "mu0"),cavity_region=(m.rhogas[...,0]==0))

    return PlotInfo("gtemp", r"$R$ [au]", r"$z$ [au]", (0, xlim), (0, xlim))

#
#   Midplane plofiles
#

@easyplot
def plot_midplane_density_profile(model, mode="average", **kwargs):
    if mode == "average":
        rhomid = model.get_midplane_profile("rhogas")
    elif mode =="slice":
        index_mid = np.argmin(np.abs(model.tc_ax - np.pi / 2))
        rhomid = model.rhogas[:,index_mid,:]
    plt.plot(model.rc_ax/nc.au, rhomid, **kwargs)
    return PlotInfo(
        "dens_prof",
        "Distance from Star [au]",
        r"Gas Density [g cm$^{-3}$]",
        loglog=True,
    )

@easyplot
def plot_midplane_temperature_profile(model, mode="average", refs_Tmid=[], refs_args=[], legend=False, **kwargs):
    if mode == "average":
        Tmid = model.get_midplane_profile("Tgas")
    elif mode =="slice":
        index_mid = np.argmin(np.abs(model.tc_ax - np.pi / 2))
        Tmid = model.Tgas[:,index_mid,:]
    plt.plot(model.rc_ax / nc.au, Tmid, zorder=2, **kwargs)

    for Tm, args in zip(refs_Tmid, refs_args):
        plt.plot(model.rc_ax / nc.au, Tm, zorder=1, **args)

    return PlotInfo(
        "T_prof",
        "Distance from Star [au]",
        r"Tempearture [K]",
        loglog=True,
        legend=legend,
    )

@easyplot
def plot_midplane_velocity_profile(model, rlim=400, ylim=(-0.5, 4), mode="average", icycle=0):
    index_mid = np.argmin(np.abs(model.tc_ax - np.pi / 2))
    cav = np.where(model.rhogas != 0, 1, 0)[:, index_mid, 0]
    rau = model.rc_ax / nc.au
    if mode == "average":
        vrmid = model.get_midplane_profile("vr")
        vtmid = model.get_midplane_profile("vt", vabs=True)
        vpmid = model.get_midplane_profile("vp")
    elif mode =="slice":
        vrmid = model.vr[:,index_mid,:]
        vtmid = model.vt[:,index_mid,:]
        vpmid = model.vp[:,index_mid,:]
    _lss = cyclestyle.by_key()['ls']
    plt.plot(
        rau,
        -vrmid * cav / 1e5,
        label=r"$- v_r$",
        ls=_lss[icycle],
    )
    plt.plot(
        rau,
        np.abs(vtmid) * cav / 1e5,
        label=r"$v_{\theta}$",
        ls=_lss[icycle+1],
    )
    plt.plot(
        rau,
        np.abs(vpmid) * cav / 1e5,
        label=r"$v_{\phi}$",
        ls=_lss[icycle+2],
    )

    plt.legend()
    return PlotInfo(
        "v_prof",
        "Distance from Star [au]",
        "Velocity [km s$^{-1}$]",
        (0, rlim),
        ylim
    )



def plot_midplane_angular_velocity_profile(model, rlim=400, ylim=(-0.5, 4)):
    index_mid = np.argmin(np.abs(model.tc_ax - np.pi / 2))
    cav = np.where(model.rhogas != 0, 1, 0)[:, index_mid, 0]
    plt.plot(
        model.rc_ax / nc.au,
        model.vp[:, index_mid, 0] * cav / model.R[:, index_mid, 0],
        label=r"$\Omega$",
        ls="--",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Angular Velocity [s$^{-1}$]")
    plt.legend()
    savefig("Omega_prof.pdf")



def plot_midplane_density_velocity_profile(
    model, xlim=(50, 1000), ylim_rho=(1e-18, 1e-15), ylim_vel=(-0.01, 2.5), mode="average"
):
    plot_midplane_density_profile(model, mode=mode, nosave=True, c="dimgray")
    plt.xlim(*xlim)
    plt.ylim(ylim_rho)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Gas Density [g cm$^{-3}$]")
    # savefig("v_dens_prof.pdf")

    ax2 = plt.twinx()
    plt.plot(np.nan, ls="-", c="dimgray", label=r"$\rho$")
    plot_midplane_velocity_profile(model, rlim=400, ylim=(-0.5, 4), mode=mode, nosave=True, icycle=1)
    plt.xscale("log")
    plt.xlim(*xlim)
    plt.ylim(ylim_vel)
    plt.ylabel("Velocity [km s$^{-1}$]")
    plt.legend(handlelength=2, fontsize=16)
    plt.minorticks_on()
    savefig("v_dens_prof.pdf")

def plot_midplane_velocity_map(model, rlim=400, mode="average"):
    rax = model.rc_ax
    tax = model.tc_ax
    axsym = True if len(model.pc_ax) == 1 else False
    pax = model.pc_ax if not axsym else np.linspace(-np.pi, np.pi, 91)

    rr, tt, pp = np.meshgrid(rax, tax, pax, indexing="ij")
    R, z = rr * [np.sin(tt), np.cos(tt)]
    x, y = R * np.sin(tt) * [np.cos(pp), np.sin(pp)]
    cav = np.where(model.rhogas != 0, 1, 0)
    vls = (x / R * model.vp + y / R * model.vR)*cav

    xau_mid = tools.take_midplane_average(model, x)/nc.au
    yau_mid = tools.take_midplane_average(model, y)/nc.au
    Vkms_mid = tools.take_midplane_average(model, vls)/1e5

    Vmax = np.max(Vkms_mid)
    lvs_half = np.arange(0.1, Vmax + 0.2, 0.2)
    lvs = np.concatenate([-lvs_half[::-1], lvs_half[0:]])

    img = plt.tricontourf(
        xau_mid.ravel(),
        yau_mid.ravel(),
        Vkms_mid.ravel(),
        lvs,
        cmap=plt.get_cmap("RdBu_r"),
        extend="both",
    )

    rhomid = tools.take_midplane_average(model, model.rhogas)
    Tmid = tools.take_midplane_average(model, model.Tgas)
    emis = rhomid*Tmid if not axsym else np.tile(rhomid * Tmid, 91)
    emis_av = np.array([ np.average(emis, axis=1) ] * x.shape[2]).T

    z = (100 * emis_av / np.max(emis_av)).ravel()
    cond = np.where(z > 0.1 , True, False)
    lvs = np.array([2**(-6), 2**(-5),  2**(-4), 2**(-3), 2**(-2), 2**(-1), 0.99999])* 100
    lss = ["solid"]*(len(lvs) -1) + ["dashed"]
    cont = plt.tricontour(
        xau_mid.ravel()[cond],
        yau_mid.ravel()[cond],
        z[cond],
        levels=lvs,
        cmap=plt.get_cmap("Greys"),
        extend="both", # "neither",
        linewidths=[0.7]*(len(lvs) -1) + [1.1],
        alpha=0.9,
        linestyles=lss,
        norm=mc.LogNorm(vmin=np.min(lvs))
    )

    try:
        del cont.collections[-1]._paths[1]
    except:
        pass
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Offset along Major Axis $x$ [au]", fontsize=14)
    plt.ylabel("Offset along Line-of-Sight $y$ [au]", fontsize=14)
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim, rlim)

    ticks_half = np.arange(0, Vmax + 0.6, 0.6)
    ticks = np.concatenate([-ticks_half[::-1], ticks_half[1:]])
    cbar = plt.colorbar(img, ticks=ticks, pad=0.02)

    cbar.set_label(r"Line-of-sight Velocity $V$ [km s$^{-1}$]")
    cbar.ax.minorticks_off()
    cont.clabel(fmt=f"%1.0f%%", fontsize=7, inline_spacing=20)
    plt.draw()
    savefig("v_los.pdf")




def plot_opacity():
    wav, kabs, kscat = np.loadtxt(
        "./run/radmc/dustkappa_MRN20.inp", skiprows=3, unpack=True
    )
    plt.plot(wav, kabs, c="k", lw=3)
    ax = plt.gca()
    ax.minorticks_on()
    plt.xlim(0.1, 1e5)
    plt.ylim(1e-4, 1e6)
    plt.xscale("log")
    plt.yscale("log")
    ax.yaxis.set_minor_locator(mt.LogLocator(numticks=15))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())  # add the custom ticks

    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel(r"Absorption Opacity [cm$^{2}$ g$^{-1}$]")
    savefig("opac.pdf")
