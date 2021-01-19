import numpy as np

class Grid:
    def __init__(self, ri_ax, ti_ax, pi_ax):
        self.ri_ax = ri_ax
        self.ti_ax = ti_ax
        self.pi_ax = pi_ax
        self.set_cellcenter_axes()
        self.set_meshgrid()
        self.set_cylyndrical_coord()

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


def get_grid(
        rau_lim=None,
        theta_lim=(0, np.pi/2),
        phi_lim=(0, 2 * np.pi),
        nr=None,
        ntheta=None,
        nphi=1,
        dr_to_r=None,
        aspect=1.0,
        logr=True
        ):

    if dr_to_r is not None:
        nr = int(np.log(rau_lim[1] / rau_lim[0]) / dr_to_r)
        ntheta_float = (theta_lim[1] - theta_lim[0]) / dr_to_r / aspect_ratio
        ntheta = np.rint(ntheta_float, dtype=int)

    if logr:
        ri_ax = np.logspace(*np.log10(rau_lim), nr + 1) * nc.au
    else:
        ri_ax = np.linspace(*rau_lim, nr + 1) * nc.au

    ti_ax = np.linspace(*theta_lim, ntheta + 1)
    pi_ax = np.linspace(*phi_lim, nphi + 1)

    return Grid(ri_ax, ti_ax, pi_ax)
