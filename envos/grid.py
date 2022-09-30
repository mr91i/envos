import numpy as np
from . import nconst as nc
from .log import logger

#logger = set_logger(__name__)


class Grid:
    def __init__(
        self,
        ri_ax=None,
        ti_ax=None,
        pi_ax=None,
        *,
        rau_lim=None,
        theta_lim=(0, np.pi / 2),
        phi_lim=(0, 2 * np.pi),
        nr=None,
        ntheta=None,
        nphi=1,
        dr_to_r=None,
        aspect_ratio=1.0,
        logr=True,
        ringhost=False,
    ):

        if (
            (ri_ax is not None) and
            (ti_ax is not None) and
            (pi_ax is not None)
        ):
            self.ri_ax = ri_ax
            self.ti_ax = ti_ax
            self.pi_ax = pi_ax

        elif (
            (rau_lim is not None) and
            (theta_lim is not None) and
            (phi_lim is not None)
        ):
            self.calc_interface_coord(
                rau_lim=rau_lim,
                theta_lim=theta_lim,
                phi_lim=phi_lim,
                nr=nr,
                ntheta=ntheta,
                nphi=nphi,
                dr_to_r=dr_to_r,
                aspect_ratio=aspect_ratio,
                logr=logr,
            )

        else:
            return None

        if ringhost:
            print(self.ri_ax)
            self.ri_ax= np.insert(self.ri_ax, 0, np.linspace(1.001*4*nc.Rsun, self.ri_ax[0], 4)[:-1] )
            print(self.ri_ax)


        self.set_cellcenter_axes()
        self.set_meshgrid()
        self.set_cylyndrical_coord()
        self.show_grid_info()

    def set_cellcenter_axes(self):
        self.rc_ax = 0.5 * (self.ri_ax[0:-1] + self.ri_ax[1:])
        self.tc_ax = 0.5 * (self.ti_ax[0:-1] + self.ti_ax[1:])
        self.pc_ax = 0.5 * (self.pi_ax[0:-1] + self.pi_ax[1:])

    def set_meshgrid(self):
        axes = (self.rc_ax, self.tc_ax, self.pc_ax)
        self.rr, self.tt, self.pp = np.meshgrid(*axes, indexing="ij")

    def set_cylyndrical_coord(self):
        self.R = self.rr * np.sin(self.tt)
        self.z = self.rr * np.cos(self.tt)

    def calc_interface_coord(
        self,
        rau_lim,
        theta_lim,
        phi_lim,
        nr=None,
        ntheta=None,
        nphi=1,
        dr_to_r=None,
        aspect_ratio=1.0,
        logr=True,
    ):
        thax_ver = 1
        ismirror = True if abs(theta_lim[1] - np.pi/2) < 1e-8 else False

        if dr_to_r is not None:
            nr = int(np.log(rau_lim[1] / rau_lim[0]) / dr_to_r)
            ntheta_float = (theta_lim[1] - theta_lim[0]) / ( dr_to_r * aspect_ratio)
            if ismirror:
                ntheta = int(round(ntheta_float))
            else:
                ntheta = int(round(ntheta_float/2)) * 2
            #ntheta = ntheta_upper if ismirror else ntheta_upper

        if logr:
            self.ri_ax = np.geomspace(*rau_lim, nr+1) * nc.au
        else:
            self.ri_ax = np.linspace(*rau_lim, nr+1) * nc.au

        if thax_ver == 1:
            self.ti_ax = np.linspace(*theta_lim, ntheta + 1)

        elif thax_ver == 2:
            dtheta = (theta_lim[1] - theta_lim[0])/ntheta
            eps = 0.001
            ti_ax = np.linspace(theta_lim[0], theta_lim[1] - dtheta*eps, ntheta + 1)
            self.ti_ax = np.concatenate([ti_ax, [ theta_lim[1] ]])
            ntheta += 1

        elif thax_ver == 3:
            print(theta_lim)
            self.ti_ax = compressed_x2(theta_lim[0], theta_lim[1], 0.95, ntheta+1)
            print(self.ti_ax)
            self.ti_ax[-1] = theta_lim[1]
            ti_ax = np.linspace(*theta_lim, ntheta + 1)
            print(ti_ax)

        self.pi_ax = np.linspace(*phi_lim, nphi + 1)

    def show_grid_info(self):
        ri = self.ri_ax / nc.au
        ti = np.rad2deg(self.ti_ax)
        pi = np.rad2deg(self.pi_ax)
        logger.info(f"Grid:")
        logger.info(f"    r  = [{ri[0]:.2f}:{ri[-1]:.2f}] au")
        logger.info(f"    Nr = {len(ri)-1}")
        logger.info(f"    θ  = [{ti[0]:.2f}:{ti[-1]:.2f}] ")
        logger.info(f"    Nθ = {len(ti)-1}")
        logger.info(f"    φ  = [{pi[0]:.2f}:{pi[-1]:.2f}] ")
        logger.info(f"    Nφ = {len(pi)-1}")
        logger.info("")


def get_interface_coord(
    rau_lim=None,
    theta_lim=(0, np.pi / 2),
    phi_lim=(0, 2 * np.pi),
    nr=None,
    ntheta=None,
    nphi=1,
    dr_to_r=None,
    aspect_ratio=1.0,
    logr=True,
):

    if dr_to_r is not None:
        nr = int(np.log(rau_lim[1] / rau_lim[0]) / dr_to_r)
        ntheta_float = (theta_lim[1] - theta_lim[0]) / dr_to_r / aspect_ratio
        ntheta_upper = int(round(ntheta_float))
        ntheta = ntheta_upper * 2 if theta_lim > np.pi/2 + 1e-8 else ntheta_upper

    if logr:
        ri_ax = np.geomspace(*rau_lim, nr + 1) * nc.au
    else:
        ri_ax = np.linspace(*rau_lim, nr + 1) * nc.au

    dtheta = (theta_lim[1] - theta_lim[0])/ntheta
    eps = 1e-5
    ti_ax = np.linspace(theta_lim[0], theta_lim[1] - dtheta*eps, ntheta + 1)
    ti_ax = np.concatenate([ti_ax, [ theta_lim[1] ]])
    ntheta += 1

    pi_ax = np.linspace(*phi_lim, nphi + 1)

    return ri_ax, ti_ax, pi_ax


def compressed_x2(xmin, xmax, xrat_root, nfaces):
    x2rat = np.abs(xrat_root)
    xmid = 0.5*np.pi
    #x = np.arange(nfaces)/(nfaces-1)
    x = np.linspace(xmin, xmax, nfaces)/np.pi
    def func(x):
        if x<=0.5: #
            ratn = x2rat**(0.5*nfaces)
            rnx = x2rat**(x*nfaces)
            lw = (rnx-ratn)/(1.0-ratn)
            rw = 1.0-lw
            return xmin*lw + xmid*rw

        else:
            ratn = (1.0/x2rat)**(0.5*nfaces)
            rnx = (1.0/x2rat)**((x-0.5)*nfaces)
            lw = (rnx-ratn)/(1.0-ratn)
            rw = 1.0-lw
            return xmid*lw + xmax*rw

    return np.vectorize(func)(x)

