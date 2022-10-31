import os
import numpy as np
from scipy import interpolate, integrate, optimize
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import matplotlib.colors as mc
import copy

try:
    from skimage.feature import peak_local_max
except:
    pass

from .. import gpath
from .. import nconst as nc
from .. import log
from .. import streamline
from .. import tools

# from myplot import mpl_setting, color

logger = log.logger # set_logger(__name__)
matplotlib.use("Agg")
# os.makedirs(gpath.fig_dir, exist_ok=True)
color_def = [
    "#3498db",
    "#e74c3c",
    "#1abc9c",
    "#9b59b6",
    "#f1c40f",
    "#34495e",
    "#446cb3",
    "#d24d57",
    "#27ae60",
    "#663399",
    "#f7ca18",
    "#bdc3c7",
    "#2c3e50",
]
eps = 1e-3

"""
plotting tools
"""

def interp_grid(X):
    dX = np.diff(X, axis=1)/2.
    X = np.hstack((X[:, [0]] - dX[:, [0]],
                   X[:, :-1] + dX,
                   X[:, [-1]] + dX[:, [-1]]))
    return X

def plot_colormap(
    xx, yy, z, *,
    zregion=None,
    clabel=None,
    clog=False,
    lvs=None,
    dlv=0.25,
    aspect="equal",
    cname="viridis",
    extend="both",
    cformat=None):

    _z = z[zregion] if zregion is not None else z
    if lvs is None:
        lvs = make_levels(_z, dlv, log=clog)

    img = plt.pcolormesh(
        xx,
        yy,
        z,
        shading="nearest",
        rasterized=True,
    )
    if lvs is not None:
        img.set_cmap(make_listed_cmap(cname, len(lvs), extend=extend))
        img.set_norm(mc.BoundaryNorm(lvs, len(lvs), clip=0))
    fmt = cformat if cformat is not None else ( "%.1e" if clog else None )
     # mt.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(10, 0.5))
    cbar = plt.colorbar(img, format=fmt, extend=extend, pad=0.02)
    if clabel is not None:
        cbar.set_label(clabel)
    cbar.minorticks_off()
    if aspect == "equal":
        plt.gca().set_aspect("equal", adjustable="box")
    return img

def make_levels(x, dlv, log=False, minfrac=1e-8):
    _x = np.log10(x[x > np.max(x) * minfrac]) if log else x
    _x = _x[np.isfinite(_x)]
    if len(_x) == 0 or len(_x) == 1:
        return None
    maxlv = np.ceil(np.max(_x) / dlv) * dlv
    minlv = np.floor(np.min(_x) / dlv) * dlv
    nlv = int(round( (maxlv - minlv) / dlv ))
    logger.debug("[make_levels] max is ", np.max(_x), " min is ", np.min(_x), ", dlv is ", dlv, ", so nlv is ", nlv)
    b = np.array([*range(nlv + 1)]) * dlv + minlv
    if len(b) > 1:
        return 10 ** b if log else b
    else:
        return None


def add_colorbar(label=None, fmt=None, extend="both", pad=0.02, minticks=False):
    cbar = plt.colorbar(img, format=fmt, extend="both", pad=pad)
    cbar.set_label(label)
    if minticks:
        cbar.ax.minorticks_off()


def make_listed_cmap(cmap_name, ncolors, extend="both"):
    io = None
    iu = None
    if extend == "neither":
        pass

    if extend in ("max", "both"):
        io = -1
        ncolors += 1

    if extend in ("min", "both"):
        iu = 1
        ncolors += 1

    cmap = plt.get_cmap(cmap_name, ncolors)
    lcmap = mc.ListedColormap(cmap.colors[iu:io])

    if extend in ("min", "both"):
        lcmap.set_under(cmap.colors[0])
    if extend in ("max", "both"):
        lcmap.set_over(cmap.colors[-1])

    return lcmap


def add_streams(
    model, rlim, r0=None, use_mu0=False, equal_theta0=False, equal_mu0=True
):
    r0 = r0 or rlim
    if model.tc_ax[-1] > np.pi/2 :
        xau = np.linspace(0, r0, 1000)
        yau = np.linspace(-r0, r0, 2000)
        xx, yy = np.meshgrid(xau, yau)
        linedens = (10, 100)

    else:
        xau = np.linspace(0, r0, 1000)
        yau = np.linspace(0, r0, 1000)
        xx, yy = np.meshgrid(xau, yau)
        linedens = 5

    newgrid = np.stack([np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)], axis=-1)

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
        r0arg = np.argmin(np.abs(r0 * nc.au - model.rc_ax))
        mu0_arr = model.mu0[r0arg, :, 0]
        if equal_theta0:
            theta_func = interpolate.interp1d(
                np.arccos(mu0_arr), model.tc_ax, fill_value="extrapolate", kind="cubic"
            )
            theta0 = theta_func(np.radians(np.linspace(0, 90, 19)[1:-1]))

        elif equal_mu0:
            mu0_to_theta_func = interpolate.interp1d(
                mu0_arr, model.tc_ax, fill_value="extrapolate", kind="linear"
            )
            theta0 = mu0_to_theta_func(np.linspace(0, 1, 11)[1:-1])

    else:
        theta0 = np.radians(np.linspace(0, 90, 19))

    start_points = r0 * np.array([np.sin(theta0), np.cos(theta0)]).T
    opt = {"density": linedens, "linewidth": 0.5, "color": "w", "arrowsize": 0.7}
    plt.streamplot(xau, yau, vR, vz, start_points=start_points, **opt)


def add_trajectries(km):
    r0 = km.ppar.cs * km.ppar.t  # / nc.au
    # start_points = [(r0, th0) for th0 in np.radians(np.linspace(0, 90, 10))]
    start_points = [
        (r0, th0) for th0 in np.radians([89.9, 85, 80, 75, 70, 65, 60, 55, 50, 44.9])
    ]
    t_eval = np.arange(1e3, 1e6, 100) * nc.year
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
        plt.plot(sl.R / nc.au, sl.z / nc.au, c="orange", lw=0.7, marker=".", ms=1.5)
    # streamline.save_data(sls)


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
        ret = interpolate.griddata(points, value_rt.ravel(), npoints, method="linear")
    s = np.array([integrate.simps(r, z_ax) for r in np.nan_to_num(ret)])
    return s


def draw_center_line():
    draw_cross_pointer(0, 0, c="k", lw=1, ls="-", s=0, alpha=0.5, zorder=1)


def draw_cross_pointer(
    x, y, c="k", lw=2, ls=":", s=10, alpha=1, zorder=1, marker_lw=1, fill=True
):
    plt.axhline(y=y, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    plt.axvline(x=x, lw=lw, ls=ls, c=c, alpha=alpha, zorder=zorder)
    if fill:
        plt.scatter(x, y, c=c, s=s, alpha=alpha, linewidth=0, zorder=zorder, ec=None)
    else:
        plt.scatter(
            x,
            y,
            s=s,
            alpha=alpha,
            zorder=zorder,
            facecolors="none",
            ec=c,
            linewidth=marker_lw,
        )


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

    if mode == "pv":
        if pangle_deg is None:
            beamx = 0.5 * (beam_maj_au + beam_min_au)
        else:
            cross_angle = (beam_pa_deg - pangle_deg) * np.pi / 180
            beam_crosslength_au = (
                (np.sin(cross_angle) / beam_min_au) ** 2
                + (np.cos(cross_angle) / beam_maj_au) ** 2
            ) ** (-0.5)
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
    LocalPeak_Pax=False,
    LocalPeak_Vax=False,
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
    xau,
    vkms,
    Ipv,
    Ms_Msun,
    rCR_au,
    incl=90,
    f_crit=None,
    f_crit_list=None,
    mass_ip=False,
    mass_vp=False,
    rangex=None,
    rangev=None,
    quadrant=None,
):
    def calc_M(xau, vkms, fac=1):
        # calc xau*nc.au * (vkms*nc.kms)**2 / (nc.G*nc.Msun)
        return 0.001127 * xau * vkms ** 2 * fac

    if quadrant is not None:
        xx, vv = np.meshgrid(xau, vkms, indexing="ij")
        if quadrant == 1:
            cond = (xx > 0) & (vv > 0)
            #print(Ipv.shape, xx.shape,  ((xx > 0) & (vv > 0)).shape )
    #        _Ipv = Ipv[(xx > 0) & (vv > 0)]
            #print( Ipv[ (xx > 0) & (vv > 0) ].shape )
    #        _xau = xau[xau>0]
    #        _vkms = vkms[vkms>0]
        elif quadrant == 2:
            cond = (xx > 0) & (vv < 0)
    #        _Ipv = Ipv[(xx > 0) & (vv < 0)]
    #        _xau = xau[xau>0]
    #        _vkms = vkms[vkms<0]
        elif quadrant == 3:
            cond = (xx < 0) & (vv < 0)
    #        _Ipv = Ipv[(xx < 0) & (vv > 0)]
    #        _xau = xau[xau<0]
    #        _vkms = vkms[vkms>0]
        elif quadrant == 4:
            cond = (xx < 0) & (vv > 0)
    #        _Ipv = Ipv[(xx < 0) & (vv < 0)]
    #        _xau = xau[xau<0]
    #        _vkms = vkms[vkms<0]
        _xau = xau
        _vkms = vkms
        _Ipv = np.where(cond, Ipv, 0)
        #_Ipv = Ipv.reshape(len(_xau), len(_vkms) )
    else:
        _Ipv = Ipv
        _xau = xau
        _vkms = vkms

    if mass_ip:
        # M_ipeak
        #print(_xau, _vkms, _Ipv)
        #exit()
        #x_vmax, v_vmax = get_coord_vmax(_xau, _vkms, _Ipv, f_crit)
        res = get_coord_ipeak(_xau, _vkms, _Ipv, mode="quadrant")
        if res is not None:
            xau_peak, vkms_peak = res
            draw_cross_pointer(xau_peak, vkms_peak, color_def[1], lw=1.5, s=18, ls=":")
            M_CR = calc_M(abs(xau_peak), vkms_peak, fac=1)
        else:
            xau_peak = 0.0
            vkms_peak = 0.0
            M_CR = 0.0
        # M_CR = calc_M(abs(xau_peak), vkms_peak / np.sin(np.deg2rad(incl)), fac=1)
        txt_Mip = rf"$M_{{\rm ipeak}}$={M_CR:.3f}"
        logger.info("Mass estimation with intensity peak:")
        logger.info(f" x_ipeak = {xau_peak} au")
        logger.info(f" V_ipeak = {vkms_peak} km/s")
        logger.info(f" V_ipeak/sin i = {vkms_peak/np.sin(np.deg2rad(incl))} km/s")
        logger.info(f" M_ipeak = {M_CR} Msun")

    if mass_vp:
        # M_vpeak
        f_crit_list = [f_crit] if f_crit is not None else f_crit_list
        txt_Mvp_list = []
        for f_crit in f_crit_list:
            x_vmax, v_vmax = get_coord_vmax2(_xau, _vkms, _Ipv, Icrit = f_crit * np.max(Ipv))
            M_CB = calc_M(abs(x_vmax), v_vmax / np.sin(np.deg2rad(incl)), fac=1 / 2)
            draw_cross_pointer(
                x_vmax, v_vmax, color_def[0], lw=1.5, s=18, ls=":", fill=False
            )
            txt_Mvp_list.append(rf"$M_{{\rm vmax,\,{f_crit*100:.0f}\%}}$={M_CB:.3f}")
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


def get_coord_ipeak(xau, vkms, Ipv, mode="quadrant"): # --> (xau, vkms) or None
    peaks = peak_local_max(Ipv, threshold_rel=0.7)
    if len(peaks) == 0:
        # imaxpeak, jmaxpeak = np.unravel_index(np.argmax(Ipv), Ipv.shape)
        # print(xau[imaxpeak], vkms[jmaxpeak])
        # return xau[imaxpeak], vkms[jmaxpeak]
        return None

    if mode=="quadrant":
        xx, vv = np.meshgrid(xau, vkms, indexing="ij")
        print("get_coord_vmax for get quadrant")
        x_vmax, v_vmax = get_coord_vmax2(xau, vkms, Ipv, 0.5*np.max(Ipv) )
#        print(x_vmax, v_vmax)
        quadrant = get_quadrant(x_vmax, v_vmax)
        if quadrant == 1:
            cond = (xx > 0) & (vv > 0)
        elif quadrant == 2:
            cond = (xx > 0) & (vv < 0)
        elif quadrant == 3:
            cond = (xx < 0) & (vv < 0)
        elif quadrant == 4:
            cond = (xx < 0) & (vv > 0)
        _Ipv = np.where(cond, Ipv, 0)
    elif mode =="max":
        _Ipv = Ipv


    ipeak, jpeak = np.unravel_index(np.argmax(_Ipv), _Ipv.shape)
    xau_peak = xau[ipeak]
    vkms_peak = vkms[jpeak]
    fun = interpolate.RectBivariateSpline(xau, vkms, Ipv)
    dx = xau[1] - xau[0]
    dv = vkms[1] - vkms[0]
    #return (xau_peak, vkms_peak)

    res = optimize.minimize(
        lambda x: 1 / fun(x[0], x[1])[0, 0],
        [xau_peak, vkms_peak],
        bounds=[
            (xau_peak - 1 * dx, xau_peak + 1 * dx),
            (vkms_peak - 1 * dv, vkms_peak + 1 * dv),
        ],
    )
    return res.x[0], res.x[1]
def get_coord_vmax2(xau, vkms, Ipv, Icrit, quadrant=None):
#    _Ipv = np.where( Ipv == Ipv, Ipv, 0)
#    _Ipv = np.where( Ipv >= 0.0, Ipv, 0)
    fun = interpolate.RectBivariateSpline(xau, vkms, Ipv)

    x = np.linspace(xau[0], xau[-1], 1000)
    v = np.linspace(vkms[0], vkms[-1], 1000)

   # x_fun = optimize.minimize_scalar(lambda x: fun(x,v) )
    x_vmax = [ optimize.minimize_scalar(lambda x: - fun(x, _v)[0,0] ).x for _v in v]

    #print(x_vmax)
    #exit()

#    x_vmax = np.apply_along_axis(lambda Ip: get_maximum_position(xau, Ip), 0, _Ipv)
    #x_vmax = np.apply_along_axis(lambda Ip: get_maximum_position(xau, Ip), 0, _Ipv)
    #I_vmax = np.apply_along_axis(np.max, 0, _Ipv)
    I_vmax = np.array([fun(_x, _v)[0,0] for _x, _v in zip(x_vmax, v)])
    #print(I_vmax)
    #mask = I_vmax >  0.0
#    plt.plot(x_vmax, v)
    #plt.plot(x_vmax, vkms)
    #print(np.array([vkms, x_vmax, I_vmax]).T , Icrit)
    #print(np.array([vkms[mask], x_vmax[mask], I_vmax[mask]]).T , Icrit)
    #v_crit = tools.find_roots(vkms[mask], I_vmax[mask], Icrit)
    v_crit = tools.find_roots(v, I_vmax, Icrit)

    #print(v_crit)
    #exit()
    if len(v_crit) == 0:
        v_crit = vkms[0]
    else:
        i = np.argmax( np.abs(v_crit) )
        v_crit = v_crit[i] # v_crit[-1]
    x_crit = tools.find_roots(x_vmax, v, v_crit)[-1]
    return x_crit, v_crit

def get_coord_vmax(xau, vkms, Ipv, Icrit, quadrant=None):
#    _Ipv = np.where( Ipv == Ipv, Ipv, 0)
    _Ipv = np.where( Ipv >= 0.0, Ipv, 0)
    fun = interpolate.RectBivariateSpline(xau, vkms, Ipv)

    x_vmax = np.apply_along_axis(lambda Ip: get_maximum_position(xau, Ip), 0, _Ipv)
    #x_vmax = np.apply_along_axis(lambda Ip: get_maximum_position(xau, Ip), 0, _Ipv)
    #I_vmax = np.apply_along_axis(np.max, 0, _Ipv)
    I_vmax = np.array([fun(_x, _v)[0,0] for _x, _v in zip(x_vmax, vkms)])
    #print(I_vmax)

    mask = I_vmax >  0.0
    plt.plot(x_vmax[mask], vkms[mask])
    #plt.plot(x_vmax, vkms)
    #print(np.array([vkms, x_vmax, I_vmax]).T , Icrit)
    #print(np.array([vkms[mask], x_vmax[mask], I_vmax[mask]]).T , Icrit)
    v_crit = tools.find_roots(vkms[mask], I_vmax[mask], Icrit)
    if len(v_crit) == 0:
        v_crit = vkms[0]
    else:
        v_crit = v_crit[-1]
    x_crit = tools.find_roots(x_vmax, vkms, v_crit)[-1]
    #print(x_crit, v_crit)
    return x_crit, v_crit


def get_maximum_position(x, y):
    return find_local_peak_position(x, y, np.argmax(y))


def find_local_peak_position(x, y, i):
    if 2 <= i <= len(x) - 3:
        i_s = i - 2
        i_e = i + 3
        grad_y = interpolate.InterpolatedUnivariateSpline(
            x[i_s:i_e], y[i_s:i_e]
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

def get_quadrant(x, y):
    return int( (np.rad2deg( np.arctan2(x, y) ) ) % 360 // 90 + 1 )

def savefig(filename):
    gpath.make_dirs(fig=gpath.fig_dir)
    filepath = os.path.join(gpath.fig_dir, filename)
    plt.savefig(filepath)
    print("saved ", filepath)
    plt.clf()
