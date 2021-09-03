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
eps=1e-3
"""
Plotting functions
"""


def plot_density_map(
    model,
    rlim=500,
    r0=None,
    streams=False, #True,
    trajectries=False, #True,
    nlv = 10,
):
    rho = model.rhogas
    lvs = make_levels(rho, 0.2, log=True)
    img = plt.pcolormesh(
        model.R[...,0]/nc.au,
        model.z[...,0]/nc.au,
        #np.log10(rho[...,0].clip(1e-300)),
        rho[...,0] ,
        shading="nearest",
        rasterized=True,
    )
    plt.xlim(0, rlim)
    plt.ylim(0, rlim)
    img.set_cmap(make_listed_cmap("viridis", len(lvs)-1, extend="both"))
    img.set_norm(mc.BoundaryNorm(lvs, len(lvs)-1, clip=True))
    plt.xlabel("R [au]")
    plt.ylabel("z [au]")
    fmt = "%.1e" #mt.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(10, 0.5))
    cbar = plt.colorbar(img, format=fmt, extend="both", pad=0.02)
    cbar.set_label(r"Gas Density [g cm$^{-3}$]")
    #cbar.ax.minorticks_on()
    plt.gca().set_aspect('equal', adjustable='box')

    if trajectries:
        add_trajectries(model)

    if streams:
        add_streams(model, rlim, r0=r0, use_mu0=hasattr(model, "mu0"))

    savefig("density.pdf")


def plot_midplane_density_profile(model):
    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    plt.plot(model.rc_ax/nc.au, model.rhogas[:, index_mid, 0])
    #plt.xlim(np.min(model.rc_ax/nc.au), np.max(model.rc_ax/nc.au))
    #plt.ylim(np.min(model.rhogas[:, index_mid, 0])*0.9, np.max(model.rhogas[:, index_mid, 0])*1.1)
    #plt.ylim(1e-19, 1e-15)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Gas Density [g cm$^{-3}$]")
    savefig("dens_prof.pdf")

def plot_midplane_temperature_profile(model):
    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    plt.plot(model.rc_ax/nc.au, model.Tgas[:, index_mid, 0])
    #plt.xlim(10, 1000)
    #plt.ylim(1, 1000)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Temperature [K]")
    savefig("T_prof.pdf")

def plot_midplane_velocity_profile(model, rlim=400, ylim=(-0.5,4)):
    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    cav = np.where(model.rhogas != 0, 1, 0)[:,index_mid, 0]
    plt.plot(model.rc_ax/nc.au, -model.vr[:, index_mid, 0]*cav/1e5, label=r"$- v_r$", ls="-")
    plt.plot(model.rc_ax/nc.au,  model.vt[:, index_mid, 0]*cav/1e5, label=r"$v_{\theta}$", ls=":")
    plt.plot(model.rc_ax/nc.au,  model.vp[:, index_mid, 0]*cav/1e5, label=r"$v_{\phi}$", ls="--", )
    #vmax = np.max(np.array([-model.vr, model.vt, model.vp]) * cav)
    #vlev_max = np.round(vmax)
    plt.xlim(0, rlim)
    plt.ylim(ylim)
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Velocity [km s$^{-1}$]")
    plt.legend()
    savefig("v_prof.pdf")

def plot_midplane_angular_velocity_profile(model, rlim=400, ylim=(-0.5,4)):
    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    cav = np.where(model.rhogas != 0, 1, 0)[:,index_mid, 0]
    plt.plot(model.rc_ax/nc.au,  model.vp[:, index_mid, 0]*cav/model.R[:, index_mid, 0] , label=r"$\Omega$", ls="--")

    #vmax = np.max(np.array([-model.vr, model.vt, model.vp]) * cav)
    #vlev_max = np.round(vmax)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Angular Velocity [s$^{-1}$]")
    plt.legend()
    savefig("Omega_prof.pdf")

def plot_midplane_density_velocity_profile(model, rlim=1000, ylim_rho=(1e-19,1e-15), ylim_vel=(-0.5, 4)):
    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    plt.plot(model.rc_ax/nc.au, model.rhogas[:, index_mid, 0], ls="-", c="dimgray", label=r"$\rho$")
    plt.xlim(10, rlim)
    plt.ylim(ylim_rho)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Distance from Star [au]")
    plt.ylabel("Gas Density [g cm$^{-3}$]")

    ax2 = plt.twinx()
    cav = np.where(model.rhogas != 0, 1, 0)[:,index_mid, 0]
    plt.plot(np.nan, ls="-", c="dimgray", label=r"$\rho$")
    plt.plot(model.rc_ax/nc.au, -model.vr[:, index_mid, 0]*cav/1e5, label=r"$- v_r$", ls="--")
    plt.plot(model.rc_ax/nc.au,  model.vt[:, index_mid, 0]*cav/1e5, label=r"$v_{\theta}$", ls=":")
    plt.plot(model.rc_ax/nc.au,  model.vp[:, index_mid, 0]*cav/1e5, label=r"$v_{\phi}$", ls="-.", )
    plt.ylim(ylim_vel)
    plt.ylabel("Velocity [km s$^{-1}$]")
    plt.legend(handlelength=3)
    plt.minorticks_on()
    plt.axhline(0, ls="-", c="k", lw=2, zorder=-10)

    savefig("v_dens_prof.pdf")

def plot_midplane_velocity_map(model, rlim=300):
    rax = model.rc_ax
    tax = model.tc_ax
    pax = model.pc_ax if len(model.pc_ax) != 1 else np.linspace(-np.pi, np.pi, 91)
    rr, tt, pp = np.meshgrid(rax, tax, pax, indexing='ij')
    R, z = rr * [np.sin(tt), np.cos(tt)]
    x, y = R * np.sin(tt) * [np.cos(pp), np.sin(pp)]
    vR = model.vr * np.sin(tt)
    vls = x/R * model.vp + y/R * model.vR
    cav = np.where(model.rhogas != 0, 1, 0)
    vls *= cav

    index_mid = np.argmin(np.abs( model.tc_ax - np.pi/2))
    xau_mid = x[:,index_mid,:].ravel()/nc.au
    yau_mid = y[:,index_mid,:].ravel()/nc.au
    Vkms_mid = vls[:,index_mid,:].ravel()/1e5

    Vmax = np.max(Vkms_mid)
    lvs_half = np.arange(0.1, Vmax+0.2, 0.2)
    lvs = np.concatenate([-lvs_half[::-1], lvs_half[0:]])

    img  = plt.tricontourf(xau_mid, yau_mid, Vkms_mid,
                           lvs, cmap=plt.get_cmap('RdBu_r'), extend="both", )
    emis =  model.rhogas[:,index_mid,:] * model.Tgas[:,index_mid,:] \
              if len(model.pc_ax) != 1 \
              else np.hstack([ model.rhogas[:,index_mid,:] * model.Tgas[:,index_mid,:] ]*91)

    emis_av = np.array([np.average(emis, axis=1)] * x.shape[2] ).T

    xau_ax = x[:,index_mid,0]/nc.au
    def func(z0, a):
        z = emis_av/np.max(emis_av)
        z_lim = np.where(z >= z0, z, 0)
        vol = integrate.simpson(2 * np.pi * xau_ax * np.average(z_lim, axis=1), xau_ax )
        vol_tot = integrate.simpson(2 * np.pi * xau_ax * np.average(z, axis=1), xau_ax )
        #print(vol/vol_tot)
        return vol/vol_tot - a

    def level_func(lev):
        sol = optimize.root_scalar(func , bracket=(0,1), args=(lev,), method='brentq' ).root
        print("level:",lev, "sol:", sol)
        return sol

    cont = plt.tricontour(xau_mid,
                          yau_mid,
                          (100*emis_av/np.max(emis_av)).ravel(),
                          #[level_func(0.75)*100, level_func(0.5)*100, level_func(0.25)*100, 101],
                          [10, 40, 70, 101],
                          cmap=plt.get_cmap('Greys'),
                          extend="both", linewidths=1)

    cond = np.where(emis_av == np.max(emis_av) ,True, False).ravel()
    plt.plot(xau_mid[cond], yau_mid[cond], ls="--", c="black", lw=1.5 )

    cont.clabel(fmt=f'%1.0f%%', fontsize=8)
    plt.xlim(-rlim, rlim)
    plt.ylim(-rlim, rlim)
    plt.xlabel("x [au]")
    #plt.ylabel("Distance along Line-of-Sight [au]")
    plt.ylabel("y [au]")

    ticks_half = np.arange(0, Vmax+0.6, 0.6)
    ticks = np.concatenate([-ticks_half[::-1], ticks_half[1:]])

    plt.gca().set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(img, ticks=ticks, pad=0.02)
    #cbar.set_label(r'$V_{\rm LOS}$ [km s$^{-1}$]')
    cbar.set_label(r'Line-of-sight Velocity $V$ [km s$^{-1}$]')
    cbar.ax.minorticks_off()
    savefig("v_los.pdf")

def plot_temperature_map(
    m,
    rlim=500,
    r0=None,
    streams=False, #True,
    trajectries=False, #True,
):
    lvs = np.linspace(10, 100, 10)
    T = m.Tgas
    lvs = np.arange(10*np.floor( 1/10*np.min(T)), np.max(T[m.rr > 50 * nc.au]) + 10 + 1e-10, 10 )


    img = plt.pcolormesh(
        m.R[:, :, 0] / nc.au,
        m.z[:, :, 0] / nc.au,
        T[:, :, 0],
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
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.axis('square')
    # cbar.ax.yaxis.set_major_formatter(mt.ScalarFormatter())
    # cbar.ax.yaxis.set_minor_formatter(mt.ScalarFormatter())

    if trajectries:
        add_trajectries(m)

    if streams:
        #mu0 = m.mu0 if hasattr(m, mu0) else None
        #add_streams(m, rlim, mu0)
        add_streams(m, rlim, r0=r0, use_mu0=hasattr(m, "mu0"))

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
        obsdata.xau, obsdata.yau, Ipp.T,
        cmap=make_listed_cmap("magma", len(lvs)-1, extend="neither"),
        norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
        shading="nearest", rasterized=True
    )
    plt.xlim(-700, 700)
    plt.ylim(-700, 700)
    plt.xlabel("x [au]")
    #plt.ylabel("y [au]")
    plt.ylabel("z [au]")
    cbar = plt.colorbar(img, pad=0.02)
    #cbar.set_label(r"Normalized Integrated Intesity")
    #cbar.set_label(r"Integrated Intesity Normalized by Maximum")
    cbar.set_label(r"$I/I_{\rm max}$")

    ax = plt.gca()
    draw_center_line()
    ax.tick_params("both", direction="inout")

    if pangle_deg is not None:

        for _pangle_deg in pangle_deg:
            x, y = position_line(1400, _pangle_deg)
            #plt.plot(x, y, ls="--", c="w", lw=1.5)
            ax.annotate('', xy=[x[-1],y[-1]], xytext=[x[0],y[0]], va='center',
                arrowprops=dict(arrowstyle = "->", color = "gray", lw=1)
                #arrowprops=dict(shrink=0, width=0.5, headwidth=16, headlength=18, connectionstyle='arc3', facecolor='gray', edgecolor="gray")
               )


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
    ylim=(-2, 2),
    show_beam=True,
    discrete=True,
    refpv=None,
    out="pvdiagrams.pdf"
):
    Ipv = PV.Ipv
    xau = PV.xau
    # xas = xau / PV.dpc
    vkms = PV.vkms

    plt.figure(figsize=figsize)
    #lvs = np.linspace(np.min(Ipv), np.max(Ipv), 11)
    lvs = np.linspace(0, np.max(Ipv), 11)
    if discrete:
        img = plt.pcolormesh(xau, vkms, Ipv,
            cmap=make_listed_cmap("cividis", len(lvs)-1, extend="neither"),
            norm=mc.BoundaryNorm(lvs, len(lvs)-1, clip=False),
            shading="nearest", rasterized=True)
    else:
        img = plt.pcolormesh(xau, vkms, Ipv,
            cmap=plt.get_cmap("cividis"),
            norm=mc.Normalize(vmin=0, vmax=np.max(Ipv)),
            shading="nearest", rasterized=True)

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("Position [au]")
    plt.ylabel(r"V [km s$^{-1}$]")
    cbar = plt.colorbar(img, pad=0.02)
    #cbar.set_label(r"Intensity [$I_{\rm max}$]")
    #cbar.set_label(r"Intensity Normalized Maximum")
    cbar.set_label(r"$I_{V}/I_{V,\rm max}$")

    ax = plt.gca()
    draw_center_line()
    # ax.minorticks_on()
    #ax.tick_params("both", direction="inout")
    ax.tick_params(direction="inout")

    if refpv:
        x_ipeak, v_ipeak = get_coord_ipeak(refpv.xau, refpv.vkms, refpv.Ipv)
        x_vmax, v_vmax = get_coord_vmax(refpv.xau, refpv.vkms, refpv.Ipv, f_crit)

        plt.contour(xau, vkms, refpv.Ipv.clip(1e-10), [0.1, 0.4, 0.8],
            colors="w",linewidths=1.5,linestyles="-",alpha=0.2,
            norm=mc.Normalize(vmin=0, vmax=np.max(refpv.Ipv)), corner_mask=False)
        plt.scatter(x_ipeak, v_ipeak, s=15, alpha=0.5, linewidth=1, c="w", ec=None)
        plt.scatter(x_vmax, v_vmax, s=15, alpha=0.5, linewidth=1, facecolors='none', ec="w")
    if mass_estimate:
        add_mass_estimate_plot(
        xau,
        vkms,
        Ipv,
        Ms_Msun=Ms_Msun,
        rCR_au=rCR_au,
        f_crit=f_crit,
        #f_crit_list=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5 ],
        incl=incl,
        mass_ip=True,
        mass_vp=True)

    if PV.convolve and show_beam:
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

    savefig(out)


"""
plotting tools
"""
def make_levels(x, dlv, log=False):
    _x = np.log10(x[ x>0 ]) if log else x
    _x = _x[np.isfinite(_x)]
    if len(_x) == 0:
        return None
    maxlv = np.ceil( np.max(_x)/dlv )*dlv
    minlv = np.floor( np.min(_x)/dlv )*dlv
    nlv = int( (maxlv - minlv)/dlv )
    b = np.array([*range(nlv+1)])*0.2 + minlv
    return 10**b if log else b

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
    if extend=="both":
        lcmap.set_under(colors[0])
        lcmap.set_over(colors[-1])
    return lcmap

def add_streams(model, rlim, r0=None, use_mu0=False, equal_theta0=False, equal_mu0=True):
    r0 = r0 or rlim
    rau = np.linspace(0, r0, 1000)
    xx, yy = np.meshgrid(rau, rau)
    newgrid = np.stack(
        [np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)], axis=-1
    )
    vR = interpolate.interpn(
        (model.rc_ax / nc.au, model.tc_ax),
        model.vR[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=None,
    )
    vz = interpolate.interpn(
        (model.rc_ax / nc.au, model.tc_ax),
        model.vz[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=None,
    )
    if use_mu0:
        r0arg = np.argmin( np.abs(r0 * nc.au - model.rc_ax) )
        mu0_arr = model.mu0[r0arg, :, 0]
        if equal_theta0:
            theta_func = interpolate.interp1d(np.arccos(mu0_arr), model.tc_ax, fill_value="extrapolate", kind='cubic')
            theta0 = theta_func(np.radians(np.linspace(0, 90, 19)[1: -1]))
        elif equal_mu0:
            mu0_to_theta_func = interpolate.interp1d(mu0_arr, model.tc_ax, fill_value="extrapolate", kind='linear')
            theta0 = mu0_to_theta_func( np.linspace(0, 1, 11)[1:-1] )
    else:
        theta0 = np.radians(np.linspace(0, 90, 19))

    start_points = r0 * np.array([np.sin(theta0), np.cos(theta0)]).T
    opt = {"density": 5, "linewidth": 0.25, "color": "w", "arrowsize": 0.6}
    plt.streamplot(rau, rau, vR, vz, start_points=start_points, **opt)


def add_trajectries(km):
    r0 = km.ppar.cs * km.ppar.t  # / nc.au
    #start_points = [(r0, th0) for th0 in np.radians(np.linspace(0, 90, 10))]
    start_points = [(r0, th0) for th0 in np.radians([89.9, 85, 80, 75, 70, 65, 60, 55, 50, 44.9])]
    t_eval = np.arange(1e3, 1e6, 100) *  nc.year
    sls = streamline.calc_streamlines(
        km.rc_ax,
        km.tc_ax,
        km.vr[:, :, 0],
        km.vt[:, :, 0],
        start_points,
        method="RK23",
        t_eval=t_eval,
    )
    for sl in sls:
        plt.plot(
            sl.R / nc.au, sl.z / nc.au, c="orange", lw=0.7, marker=".", ms=1.5
        )
    #streamline.save_data(sls)


def vertical_integral(value_rt, R_rt, z_rt, R_ax, z_ax, log=False):
    points = np.stack((R_rt.ravel(), z_rt.ravel()), axis=-1)
    npoints = np.stack(np.meshgrid(R_ax, z_ax), axis=-1)
    if log:
        fltr = np.logical_and.reduce(
            (
                [np.all(np.isfinite(a)) for a in np.log10(points)],
                np.isfinite(np.log10(value_rt)),
            )
        )
        fltr = fltr.ravel()
        v = value_rt.ravel()
        ret = 10 ** interpolate.griddata(
            np.log10(points[fltr]),
            np.log10(v[fltr]),
            np.log10(npoints),
            method="linear",
        )
    else:
        ret = interpolate.griddata(
            points, value_rt.ravel(), npoints, method="linear"
        )
    s = np.array([integrate.simps(r, z_ax) for r in np.nan_to_num(ret)])
    return s


def draw_center_line():
    draw_cross_pointer(0, 0, c="k", lw=1, ls="-", s=0, alpha=0.5, zorder=1)

def draw_cross_pointer(x, y, c="k", lw=2, ls=":", s=10, alpha=1, zorder=1, marker_lw=1, fill=True):
    plt.axhline(y=y, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    plt.axvline(x=x, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    if fill:
        plt.scatter(x, y, c=c, s=s, alpha=alpha, linewidth=0, zorder=zorder, ec=None)
    else:
        plt.scatter(x, y, s=s, alpha=alpha, zorder=zorder, facecolors="none", ec=c, linewidth=marker_lw)

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

    if mode == "PV":
        if pangle_deg is None:
            beamx = 0.5 * (beam_maj_au + beam_min_au)
        else:
            cross_angle = (beam_pa_deg - pangle_deg) * np.pi / 180
            beam_crosslength_au =  (
                (np.sin(cross_angle)/beam_min_au) ** 2
                + (np.cos(cross_angle)/beam_maj_au) ** 2
            )**(-0.5)
            beamx = beam_crosslength_au
        beamy = vreso_kms
        angle = 0
    elif mode == "mom0":
        beamx = beam_min_au
        beamy = beam_maj_au
        angle = beam_pa_deg

    if with_box:
        e1 = matplotlib.patches.Ellipse(
            xy=(0, 0),
            width=beamx,
            height=beamy,
            lw=1,
            fill=True,
            ec="k",
            fc="0.6",
            angle=angle,
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
            angle=angle,
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


def add_peaks(
        LocalPeak_Pax= False,
        LocalPeak_Vax= False,
        LocalPeak_2D=False,
    ):
    if LocalPeak_Pax:
        for Iv, v_ in zip(Ipv.transpose(0, 1), vkms):
            for xM in get_localpeak_positions(
                xau, Iv, threshold_abs=np.max(Ipv) * 1e-10
            ):
                plt.plot(xM, v_, c="red", markersize=1, marker="o")

    if LocalPeak_Vax:
        for Ip, x_ in zip(Ipv.transpose(1, 0), xau):
            for vM in get_localpeak_positions(
                vkms, Ip, threshold_abs=np.max(Ipv) * 1e-10
            ):
                plt.plot(x_, vM, c="blue", markersize=1, marker="o")

    if LocalPeak_2D:
        for jM, iM in peak_local_max(
            Ipv, num_peaks=4, min_distance=10
        ):  #  min_distance=None):
            plt.scatter(xau[iM], vkms[jM], c="k", s=20, zorder=10)
            print(
                f"Local Max:   {xau[iM]:.1f} au  , {vkms[jM]:.1f}    km/s  ({jM}, {iM})"
            )

def add_mass_estimate_plot(
    xau, vkms, Ipv,
    Ms_Msun,
    rCR_au,
    incl=90,
    f_crit=None,
    f_crit_list=None,
    mass_ip=False,
    mass_vp=False,
):

    def calc_M(xau, vkms, fac=1):
        # calc xau*nc.au * (vkms*nc.kms)**2 / (nc.G*nc.Msun)
        return 0.001127 * xau * vkms ** 2 * fac

    if mass_ip:
        ## M_ipeak
        xau_peak, vkms_peak = get_coord_ipeak(xau, vkms, Ipv )
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
        f_crit_list = [f_crit] if f_crit is not None else f_crit_list

        txt_Mvp_list = []
        for f_crit in f_crit_list:
            x_vmax, v_vmax = get_coord_vmax(xau, vkms, Ipv, f_crit)
            M_CB = calc_M(abs(x_vmax), v_vmax/np.sin(np.deg2rad(incl)) , fac=1 / 2)
            draw_cross_pointer(x_vmax, v_vmax, color_def[0], lw=1.5, s=18, ls=":", fill=False)
            txt_Mvp_list.append( rf"$M_{{\rm vmax,\,{f_crit*100:.0f}\%}}$={M_CB:.3f}" )
            logger.info(f"Mass estimation with maximum velocity, f_crit={f_crit} :")
            logger.info(f" x_vmax = {x_vmax} au")
            logger.info(f" V_vmax = {v_vmax} km/s")
            logger.info(f" V_vmax/sin(i) = {v_vmax/np.sin(np.deg2rad(incl))} km/s")
            logger.info(f" M_vmax = {M_CB} Msun")

        txt_Mvp = "\n".join(txt_Mvp_list)

    if mass_ip or mass_vp:
        txt = plt.text(
            0.95,
            0.05,
            txt_Mip + "\n" + txt_Mvp,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(fc="white", ec="black", pad=5),
        )
        return txt


def get_coord_ipeak(xau, vkms, Ipv):
    jmax, imax = Ipv.shape
    i0 = imax // 2
    j0 = jmax // 2
    jpeak, ipeak = peak_local_max(Ipv[j0:, i0:], num_peaks=1)[0]
    xau_peak = xau[i0+ipeak]
    vkms_peak = vkms[j0+jpeak]

    fun = interpolate.RectBivariateSpline(vkms, xau, Ipv)
    fun_wrap = lambda x: 1/fun(x[0], x[1])[0,0]
    res = optimize.minimize(fun_wrap, [vkms_peak, xau_peak], bounds=[(vkms_peak*0.5, vkms_peak*1.5), (xau_peak*0.5, xau_peak*1.5)] )
    return res.x[1], res.x[0]

def get_coord_vmax(xau, vkms, Ipv, f_crit):
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
    return x_crit, v_crit

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

def get_localpeak_positions(x, y, min_distance=3, threshold_abs=None):
    maxis = [
        find_local_peak_position(x, y, mi)
        for mi in peak_local_max(
            y, min_distance=min_distance, threshold_abs=threshold_abs
        )[:, 0]
    ]
    return np.array(maxis)

def savefig(filename):
    gpath.make_dirs(fig=gpath.fig_dir)
    filepath = os.path.join(gpath.fig_dir, filename)
    print("saved ", filepath)
    plt.savefig(filepath)
    plt.clf()


