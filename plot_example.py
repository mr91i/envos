import os
import numpy as np
from scipy import interpolate, integrate
from skimage.feature import peak_local_max
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mt
import matplotlib.colors as mc
import time

# import myplot as mp
from envos import global_paths as gpath
from envos import tools
from envos import nconst as nc
from envos import log
from envos import streamline

logger = log.set_logger(__name__)

matplotlib.use("Agg")
# matplotlib.use('tkagg')
# matplotlib.use('pdf')

dpath_fig = gpath.fig_dir
####################################################################################################
from myplot import mpl_setting


def add_streams(km, rlim):
    r0 = rlim
    rau = np.linspace(0, rlim, 1000)
    xx, yy = np.meshgrid(rau, rau)
    newgrid = np.stack(
        [np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)], axis=-1
    )
    vR = interpolate.interpn(
        (km.rc_ax / nc.au, km.tc_ax),
        km.vR[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=0,
    )
    vz = interpolate.interpn(
        (km.rc_ax / nc.au, km.tc_ax),
        km.vz[:, :, 0],
        newgrid,
        bounds_error=False,
        fill_value=0,
    )
    start_points = [
        (r0 * np.sin(th0), r0 * np.cos(th0))
        for th0 in np.radians(np.linspace(0, 90, 19))
    ]
    opt = {"density": 5, "linewidth": 0.5, "color": "w", "arrowsize": 0.75}
    ret = plt.streamplot(rau, rau, vR, vz, start_points=start_points, **opt)


def add_trajectries(km):
    r0 = km.ppar.cs * km.ppar.t  # / nc.au
    start_points = [(r0, th0) for th0 in np.radians(np.linspace(0, 90, 10))]
    # trs = _trace_particles_2d_meridional(km.rc_ax/nc.au, km.tc_ax, km.vr[:,:,0], km.vt[:,:,0], start_points)
    sls = streamline.calc_streamlines(
        km.rc_ax, km.tc_ax, km.vr[:, :, 0], km.vt[:, :, 0], start_points
    )
    for sl in sls:
        plt.plot(
            sl.R / nc.au, sl.z / nc.au, c="orange", lw=1.0, marker=".", ms=5
        )
    streamline.save_data(sls)


def plot_density_map(
    km,
    rlim=500,
    fname="density.pdf",
    dpath_fig=gpath.fig_dir,
    streams=True,
    trajectries=True,
    filepath=None,
):
    lvs = np.logspace(-18, -15, 10)
    img = plt.contourf(
        km.R[:, :, 0] / nc.au,
        km.z[:, :, 0] / nc.au,
        km.rhogas[:, :, 0],
        lvs,
        norm=mc.LogNorm(),
    )
    plt.xlim(0, rlim)
    plt.ylim(0, rlim)
    plt.xlabel("R [au]")
    plt.ylabel("z [au]")
    cbar = plt.colorbar(img, ticks=mt.LogLocator())
    cbar.set_label(r"Gas Mass Density [g cm$^{-3}$]")

    if trajectries:
        add_trajectries(km)

    if streams:
        add_streams(km, rlim)
    #
    if filepath is None:
        filepath = os.path.join(gpath.run_dir, fname)
    plt.savefig(filepath)
    plt.clf()

    # matplotlib.use('tkagg')
    # plt.show()
    # plt.draw()
    # matplotlib.use('Agg')
    # exit()


def plot_midplane_numberdensity_profile(km, fname="ndens.pdf"):
    plt.plot(km.rc_ax, km.rho[:, -1, 0] / km.meanmolw)
    plt.xlim(10, 1000)
    plt.ylim(10, 1000)
    plt.xscale("log")
    plt.yscale("log")
    filepath = os.path.join(gpath.run_dir, fname)
    plt.savefig(filepath)
    # plt.show()
    plt.clf()


def plot_temperature_map(
    m,
    rlim=500,
    fname="temperature.pdf",
    dpath_fig=gpath.fig_dir,
    streams=True,
    trajectries=True,
    filepath=None,
):
    t = time.time()
    lvs = np.linspace(10, 100, 10)
    # lvs = np.logspace(1, 2, 11)
    img = plt.contourf(
        m.R[:, :, 0] / nc.au,
        m.z[:, :, 0] / nc.au,
        m.Tgas[:, :, 0],
        lvs,
        cmap=plt.get_cmap("inferno"),
    )  # norm=mc.LogNorm())
    plt.xlim(0, rlim)
    plt.ylim(0, rlim)
    plt.xlabel("R [au]")
    plt.ylabel("z [au]")
    # cbar = plt.colorbar(img, ticks=mt.LogLocator(), format=mt.ScalarFormatter())
    cbar = plt.colorbar(img)
    cbar.set_label(r"Gas Temperature [K]")
    cbar.ax.minorticks_off()
    # cbar.ax.yaxis.set_major_formatter(mt.ScalarFormatter())
    # cbar.ax.yaxis.set_minor_formatter(mt.ScalarFormatter())

    if trajectries:
        add_trajectries(m)

    if streams:
        add_streams(m, rlim)

    if filepath is None:
        filepath = os.path.join(gpath.run_dir, fname)
    plt.savefig(filepath)
    plt.clf()
    print("showing")
    print(time.time() - t)
    # plt.show()


####################################################################################################


def plot_radmc_data(rmc_data):
    xlim = [rmc_data.xauc[0], rmc_data.xauc[-1]]

    pl1d = mp.Plotter(
        rmc_data.dpath_fig,
        x=rmc_data.xauc,
        logx=True,
        leg=False,
        xl="Radius [au]",
        xlim=xlim,
        fn_wrapper=lambda s: "rmc_%s_1d" % s,
    )

    pl2d = mp.Plotter(
        rmc_data.dpath_fig,
        x=rmc_data.RR,
        y=rmc_data.zz,
        logx=False,
        logy=False,
        logcb=True,
        leg=False,
        xl="Radius [au]",
        yl="Height [au]",
        xlim=[0, 500],
        ylim=[0, 500],
        fn_wrapper=lambda s: "rmc_%s_2d" % s,
        square=True,
    )

    def plot_plofs(d, fname, lim=None, log=True, lb=None):
        pl1d.plot(d[:, -1], fname, ylim=lim, logy=log, yl=lb)
        pl2d.map(d, fname, ctlim=lim, logcb=log, cbl=lb)

    if rmc_data.use_gdens:
        nmin = rmc_data.ndens_mol.min()
        nmax = rmc_data.ndens_mol.max()
        maxlim = 10 ** (0.5 + round(np.log10(nmax)))
        plot_plofs(
            rmc_data.ndens_mol,
            "nden",
            lim=[maxlim * 1e-3, maxlim],
            lb=r"Number density [cm$^{-3}$]",
        )

    if rmc_data.use_Tgas:
        pl1d.plot(
            rmc_data.Tgas[:, -1],
            "temp",
            ylim=[1, 1000],
            logy=True,
            yl="Temperature [K]",
        )
        pl1d.plot(
            rmc_data.Tgas[:, 0],
            "temp_pol",
            ylim=[1, 1000],
            logy=True,
            yl="Temperature [K]",
        )
        pl2d.map(
            rmc_data.Tgas,
            "temp_in",
            ctlim=[0, 200],
            xlim=[0, 100],
            ylim=[0, 100],
            logcb=False,
            cbl="Temperature [K]",
        )
        pl2d.map(
            rmc_data.Tgas,
            "temp_out",
            ctlim=[0, 100],
            logcb=False,
            cbl="Temperature [K]",
        )
        pl2d.map(
            rmc_data.Tgas,
            "temp_L",
            ctlim=[0, 40],
            xlim=[0, 7000],
            ylim=[0, 7000],
            logcb=False,
            cbl="Temperature [K]",
        )
        pl2d_log = mp.Plotter(
            rmc_data.dpath_fig,
            x=rmc_data.RR,
            y=rmc_data.zz,
            logx=True,
            logy=True,
            logcb=True,
            leg=False,
            xl="log Radius [au]",
            yl="log Height [au]",
            xlim=[1, 1000],
            ylim=[1, 1000],
            fn_wrapper=lambda s: "rmc_%s_2d" % s,
            square=True,
        )

        pl2d_log.map(
            rmc_data.Tgas,
            "temp_log",
            ctlim=[10 ** 0.5, 10 ** 2.5],
            cbl="log Temperature [K]",
        )

    if rmc_data.use_gvel:
        lb_v = r"Velocity [km s$^{-1}$]"
        plot_plofs(-rmc_data.vr / 1e5, "gvelr", lim=[0, 5], log=False, lb=lb_v)
        plot_plofs(rmc_data.vt / 1e5, "gvelt", lim=[-5, 5], log=False, lb=lb_v)
        plot_plofs(
            np.abs(rmc_data.vp) / 1e5, "gvelp", lim=[0, 5], log=False, lb=lb_v
        )

    if rmc_data.use_gdens and rmc_data.use_Tgas:
        plot_plofs(
            rmc_data.t_dest / rmc_data.t_dyn,
            "tche",
            lim=[1e-3, 1e3],
            lb="CCH Lifetime/Dynamical Timescale",
        )

    if rmc_data.opac != "":
        with open(f"{radmc_dir}/dustkappa_{rmc_data.opac}.inp", mode="r") as f:
            read_data = f.readlines()
            mode = int(read_data[0])
        if mode == 2:
            lam, kappa_abs, kappa_sca = np.loadtxt(
                f"{radmc_dir}/dustkappa_{rmc_data.opac}.inp", skiprows=2
            ).T
        elif mode == 3:
            lam, kappa_abs, kappa_sca, _ = np.loadtxt(
                f"{radmc_dir}/dustkappa_{rmc_data.opac}.inp", skiprows=3
            ).T

        mp.Plotter(rmc_data.dpath_fig).plot(
            [
                ["ext", kappa_abs + kappa_sca],
                ["abs", kappa_abs],
                ["sca", kappa_sca],
            ],
            "dustopac",
            x=lam,
            xlim=[0.03, 3e4],
            ylim=[1e-4, 1e6],
            logx=True,
            logy=True,
            xl=r"Wavelength [$\mu$m]",
            yl=r"Dust Extinction Opacity [cm$^2$ g$^{-1}$]",
            ls=["--"],
            c=["k"],
            lw=[3, 2, 2],
        )

    #   if 1:
    #       pl2d.map(D.rho, 'rho_L', ctlim=[1e-20, 1e-16], xlim=[0, 5000], ylim=[0, 5000], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec, save=False)
    #       pl2d.ax.plot(rmc_data.pos_list[0].R/nc.au, rmc_data.pos_list[0].z/nc.au, c="orangered", lw=1.5, marker="o")
    #       pl2d.save("rho_L_pt")

    if rmc_data.plot_tau:
        pl1d = mp.Plotter(
            rmc_data.dpath_fig,
            x=rmc_data.imx / nc.au,
            logx=True,
            leg=False,
            xl="Radius [au]",
            xlim=xlim,
            fn_wrapper=lambda s: "rmc_%s_1d" % s,
        )

        pl2d = mp.Plotter(
            rmc_data.dpath_fig,
            x=rmc_data.imx / nc.au,
            y=rmc_data.imy / nc.au,
            logx=False,
            logy=False,
            logcb=True,
            leg=False,
            xl="Radius [au]",
            yl="Height [au]",
            xlim=[-500 / 2, 500 / 2],
            ylim=[-500 / 2, 500 / 2],
            fn_wrapper=lambda s: "rmc_%s_2d" % s,
            square=True,
        )
        plot_plofs(rmc_data.tau / nc.au, "tau", lim=[1e-2, 1000], lb=r"tau")


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


def _trace_particles_2d_meridional(
    rc_ax, tc_ax, vr, vth, start_points, t_span=None, nt=600
):
    """
    There are some choice for coordination, but i gave up genelarixation for the code simplicity.
    input axis : rth
    velocity   : rth
    return pos : rth
    """
    vr_field = interpolate.RegularGridInterpolator(
        (rc_ax, tc_ax), vr, bounds_error=False, fill_value=None
    )
    vth_field = interpolate.RegularGridInterpolator(
        (rc_ax, tc_ax), vth, bounds_error=False, fill_value=None
    )

    def func(t, pos, hit_flag=0):
        # if pos[0] > rc_ax[-1]:
        #    raise Exception(f"Too large position. r must be less than {rc_ax[-1]/nc.au} au.")
        if hit_midplane(t, pos) < 0:
            hit_flag = 1
        r, th = pos[0], pos[1]
        vr = vr_field((r, th))
        vth = vth_field((r, th))
        return np.array([vr, vth / r])

    def hit_midplane(t, pos):
        return np.pi / 2 - pos[1]

    hit_midplane.terminal = True

    if t_span is None:
        t_span = (0, 1e10)
        t_trace = None
    else:
        t_trace = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), nt)

    # if pos0[0] > rc_ax[-1]:
    #    print(f"Too large position:r0 = {pos0[0]/nc.au} au. r0 must be less than {rc_ax[-1]/nc.au} au. I use r0 = {rc_ax[-1]/nc.au} au instead of r0 = {pos0[0]/nc.au} au")
    #    pos0 = [rc_ax[-1], pos0[1]]
    start_points = [
        ((rc_ax[-1], p0[1]) if p0[0] > rc_ax[-1] else p0)
        for p0 in start_points
    ]

    # trajectries = [integrate.solve_ivp(func, t_span, p0, method='RK23', events=hit_midplane, rtol=1e-2) for p0 in start_points]
    trajectries = [
        integrate.solve_ivp(
            func,
            t_span,
            p0,
            method="RK23",
            t_eval=t_trace,
            events=hit_midplane,
            rtol=1e-3,
        )
        for p0 in start_points
    ]
    return [
        (t.y[0] * np.sin(t.y[1]), t.y[0] * np.cos(t.y[1])) for t in trajectries
    ]


####################################################################################################


def draw_center_line(ax):
    draw_cross_pointer(ax, 0, 0, c="k", lw=2, s=10, alpha=1, zorder=1)


def draw_cross_pointer(ax, x, y, c="k", lw=2, s=10, alpha=1, zorder=1):
    ax.axhline(y=y, lw=lw, ls=":", c=c, alpha=alpha, zorder=zorder)
    ax.axvline(x=x, lw=lw, ls=":", c=c, alpha=alpha, zorder=zorder)
    ax.scatter(x, y, c=c, s=s, alpha=alpha, linewidth=0, zorder=zorder)


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


def plot_mom0_map(
    obsdata, pangle_deg=None, poffset_au=None, n_lv=100,
    fname="mom0.pdf",
    dpath_fig=gpath.fig_dir,
    filepath=None,
):
    def position_line(xau, pangle_deg, poffset_au=0):
        pangle_rad = pangle_deg * np.pi / 180
        pos_x = xau * np.cos(pangle_rad) - poffset_au * np.sin(pangle_rad)
        pos_y = xau * np.sin(pangle_rad) + poffset_au * np.sin(pangle_rad)
        return np.stack([pos_x, pos_y], axis=-1)

    img = plt.contourf(
        obsdata.xau, obsdata.yau, obsdata.Ippv[:,:,0], n_lv
    )
    #plt.xlim(0, rlim)
    #plt.ylim(0, rlim)
    plt.xlabel("x [au]")
    plt.ylabel("y [au]")
    #cbar = plt.colorbar(img, ticks=mt.LogLocator())
    #cbar.set_label(r"Gas Mass Density [g cm$^{-3}$]")

    draw_center_line(plt.gca())

    if pangle_deg is not None:
        pline0 = position_line(obsdata.xau, pangle_deg)
        plt.plot(pline0[0], pline0[1], ls="--", c="w", lw=1)
        if poffset_au is not None:
            pline = position_line(
                obsdata.xau, pangle_deg, poffset_au=poffset_au
            )
            plt.plot(pline[0], pline[1], c="w", lw=1)

    if obsdata.convolve and obsdata.beam_maj_au :
        # draw_beamsize(pltr, obsdata.conv_info, mode="mom0")
        draw_beamsize(
            plt.gca(),
            "mom0",
            obsdata.beam_maj_au,
            obsdata.beam_min_au,
            obsdata.beam_pa_deg,
        )

    if filepath is None:
        filepath = os.path.join(gpath.run_dir, fname)
    print("saved ", filepath)
    plt.savefig(filepath)
    plt.clf()


    exit()








    pltr = mp.Plotter(dpath_fig, x=obsdata.xau, y=obsdata.yau)
    pltr.map(
        obsdata.Imom0,
        out="mom0map",
        xl="Position [au]",
        yl="Position [au]",
        cbl=r"Intensity [Jy pixel$^{-1}$ ]",
        div=n_lv,
        mode="grid",
        ctlim=[0, obsdata.Imom0.max()],
        square=True,
        save=False,
    )


    pltr.save("mom0map")


def plot_chmap(obsdata, dpath_fig=dpath_fig, n_lv=20):
    cbmax = obsdata.Ippv_max
    pltr = mp.Plotter(
        dpath_fig,
        x=obsdata.xau,
        y=obsdata.yau,
        xl="Position [au]",
        yl="Position [au]",
        cbl=r"Intensity [Jy pixel$^{-1}$ ]",
    )

    for i in range(obsdata.Nz):
        pltr.map(
            Ippv[i],
            out="chmap_{:0=4d}".format(i),
            n_lv=n_lv,
            ctlim=[0, cbmax],
            mode="grid",
            title="v = {:.3f} km/s".format(obsdata.vkms[i]),
            square=True,
        )


def plot_lineprofile(obsdata, dpath_fig=dpath_fig):
    lp = integrate.simps(
        integrate.simps(obsdata.Ippv, obsdata.xau, axis=2), obsdata.yau, axis=1
    )
    plt.plot(obsdata.vkms, lp)
    plt.savefig("test.pdf")


def plot_pvdiagram(
    PV,
    dpath_fig=dpath_fig,
    out="pvd",
    n_lv=5,
    Ms_Msun=None,
    rCR_au=None,
    f_crit=None,
    mass_ip=False,
    mass_vp=False,
    mapmode="grid",
    oplot={},
):
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

    #  if normalize == "peak":
    #      if Imax > 0:
    #          Ipv /= Imax
    #          ctlim = [1/n_lv, 1]
    #          n_lv -= 1
    #      else:
    #          ctlim = [0, 0.1]
    #      unit = r'[$I_{\rm max}$]'
    #  else:
    #      ctlim = [0, Imax] if not self.Imax else [0, self.Imax]
    Ipv = PV.Ipv
    xau = PV.xau
    xas = xau / PV.dpc
    vkms = PV.vkms
    dpc = PV.dpc
    ctlim = [0, np.max(Ipv)]

    print(xau)
    pltr = mp.Plotter(
        dpath_fig, x=PV.xau, y=PV.vkms, xlim=[-700, 700], ylim=[-4, 4]
    )

    pltr.map(
        z=Ipv,
        mode=mapmode,
        ctlim=ctlim,
        xl="Angular Offset [au]",
        yl=r"Velocity [km s$^{-1}$]",
        cbl=f"Intensity {PV.unit_I}",
        lw=1.5,
        div=n_lv,
        save=False,
        clabel=True,
    )

    draw_center_line(pltr.ax)
    pltr.ax.minorticks_on()
    pltr.ax.tick_params("both", direction="inout")

    pltr.ax2 = pltr.ax.twiny()
    pltr.ax2.minorticks_on()
    pltr.ax2.tick_params("both", direction="inout")
    pltr.ax2.set_xlabel("Angular Offset [arcsec]")
    pltr.ax2.set_xlim(np.array(pltr.ax.get_xlim()) / dpc)

    l = np.sqrt(nc.G * Ms_Msun * nc.Msun * rCR_au * nc.au)
    a = 2 * xau / rCR_au

    overplot = {
        "KeplerRotation": False,
        "ire_fit": False,
        "LocalPeak_Pax": False,
        "LocalPeak_Vax": False,
        "LocalPeak_2D": False,
    }
    overplot.update(oplot)
    if overplot["ire_fit"]:
        pltr.ax.plot(
            xau,
            2
            * l
            / rCR_au
            * (2 * xau.clip(0) / rCR_au) ** (-2 / 3)
            * 1
            / nc.kms,
            c="hotpink",
            ls=":",
            lw=1,
        )

    if overplot["KeplerRotation"]:
        vKep = np.sqrt(nc.G * Ms_Msun * nc.Msun / (xau * nc.au)) * 1 / nc.kms
        pltr.ax.plot(xau, vKep, c="cyan", ls=":", lw=1)

    if overplot["LocalPeak_Pax"]:
        for Iv, v_ in zip(Ipv.transpose(0, 1), vkms):
            for xM in get_localpeak_positions(
                xau, Iv, threshold_abs=np.max(Ipv) * 1e-10
            ):
                pltr.ax.plot(xM, v_, c="red", markersize=1, marker="o")

    if overplot["LocalPeak_Vax"]:
        for Ip, x_ in zip(Ipv.transpose(1, 0), xau):
            for vM in get_localpeak_positions(
                vkms, Ip, threshold_abs=np.max(Ipv) * 1e-10
            ):
                pltr.ax.plot(x_, vM, c="blue", markersize=1, marker="o")

    if overplot["LocalPeak_2D"]:
        for jM, iM in peak_local_max(
            Ipv, num_peaks=4, min_distance=10
        ):  #  min_distance=None):
            pltr.ax.scatter(xau[iM], vkms[jM], c="k", s=20, zorder=10)
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
        xau_peak = xau[i0 + ipeak]
        vkms_peak = vkms[j0 + jpeak]

        vp_peaks = [
            (vkms[jM], xau[iM])
            for jM, iM in peak_local_max(Ipv[i0:, j0:], num_peaks=1)
        ]

        print(peak_local_max(Ipv[i0:, j0:], num_peaks=1))
        i = np.argmax(
            np.sign(vp_peaks[:, 0]) * vp_peaks[:, 0] * vp_peaks[:, 1]
        )
        vkms_peak, xau_peak = vp_peaks[i]

        M_CR = calc_M(abs(xau_peak), vkms_peak, fac=1)
        M_CR_vpeak = M_CR / np.sqrt(2)
        draw_cross_pointer(xau_peak / dpc, vkms_peak, mp.c_def[1])
        txt_Mip = rf"$M_{{\rm ip}}$={M_CR:.3f}"

    if mass_vp:
        ## M_vpeak
        x_vmax, I_vmax = np.array(
            [
                [get_maximum_position(obsdata.xau, Iv), np.max(Iv)]
                for Iv in Ipv.transpose(0, 1)
            ]
        ).T

        if (np.min(I_vmax) < f_crit * ctlim[1]) and (
            f_crit * ctlim[1] < np.max(I_vmax)
        ):
            v_crit = tools.find_roots(vkms, I_vmax, f_crit * ctlim[1])
        else:
            v_crit = tools.find_roots(vkms, I_vmax, f_crit * np.max(Ippv))

        if len(v_crit) == 0:
            v_crit = [vkms[0]]

        x_crit = tools.find_roots(x_vmax, vkms, v_crit[0])
        M_CB = calc_M(abs(x_crit[0]), v_crit[0], fac=1 / 2)
        draw_cross_pointer(x_crit[0] / dpc, v_crit[0], mp.c_def[0])
        txt_Mvp = rf"$M_{{\rm vp, {f_crit*100}\%}}$={M_CB:.3f}"

    if mass_ip or mass_vp:
        plt.text(
            0.95,
            0.05,
            txt_Mip + "\n" + txt_Mvp,
            transform=pltr.ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(fc="white", ec="black", pad=5),
        )

    try:
        draw_beamsize(
            pltr.ax,
            "PV",
            PV.beam_maj_au,
            PV.beam_min_au,
            PV.beam_pa_deg,
            PV.pangle_deg,
            PV.vreso_kms,
        )
    except:
        pass

    pltr.save(out)
