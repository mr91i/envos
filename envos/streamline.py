import os
import numpy as np
from dataclasses import dataclass, field
from scipy import interpolate, integrate
from envos import log
from envos.gpath import run_dir
import envos.nconst as nc

logger = log.set_logger(__name__)

def calc_physical_values_along_streamlines(
    model, start_points, **kwargs
):
    calc_streamlines_from_model(
        model, ["rhogas", "Tgas"], ["g cm^-3", "K"], start_points, **kwargs
    )



def calc_streamlines_from_model(
    model, name_list, unit_list, start_points, **kwargs
):
    values = [(n, getattr(model, n), u) for n, u in zip(name_list, unit_list)]
    calc_streamlines(
        model.rc_ax,
        model.tc_ax,
        model.vr[:, :, 0],
        model.vt[:, :, 0],
        start_points,
        values=values,
        **kwargs,
    )


def calc_streamlines(
    r_ax,
    t_ax,
    vr,
    vt,
    pos0list,
    values=[],
    t_span=(1, 1e30),
    nt=500,
    rtol=1e-4,
    filename="stream",
    method="RK23",
    dpath=None,
    save=False,
):
    slc = StreamlineCalculator(
        r_ax,
        t_ax,
        vr,
        vt,
        pos0list,
        t_span=t_span,
        nt=nt,
        rtol=rtol,
        method=method,
    )
    for name, value, unitname in values:
        slc.add_value(name, value, unitname)
    slc.calc_streamlines()
    if save:
        save_data(slc.streamlines, filename=filename, dpath=dpath)
    return slc.streamlines


@dataclass
class Streamline:
    pos0: np.ndarray
    t: np.ndarray
    R: np.ndarray
    z: np.ndarray
    vR: np.ndarray
    vz: np.ndarray
    values: list = field(default_factory=list)

    def add_value(self, name, value, unit):
        self.values.append([name, value, unit])

    def get_values(self):
        return self.values


class StreamlineCalculator:
    def __init__(
        self,
        r_ax,
        t_ax,
        vr,
        vt,
        pos0list,
        t_span=(1, 1e30),
        nt=500,
        rtol=1e-8,
        method="RK45",
    ):
        self.r_ax = r_ax
        self.t_ax = t_ax
        self.vr_field = interpolate.RegularGridInterpolator(
            (r_ax, t_ax), vr, bounds_error=False, fill_value=None
        )
        self.vt_field = interpolate.RegularGridInterpolator(
            (r_ax, t_ax), vt, bounds_error=False, fill_value=None
        )
        self.pos0list = pos0list
        self.t_eval = np.geomspace(t_span[0], t_span[-1], nt)
        self.rtol = rtol
        self.method = method
        self.streamlines = []
        self.value_list = []
        self._hit_midplane.terminal = True

    def add_value(self, name, value, unitname):
        self.value_list.append([name, value, unitname])

    def calc_streamlines(self):
        for pos0 in self.pos0list:
            self.calc_streamline(pos0)

    def calc_streamline(self, pos0):
        if pos0[0] > self.r_ax[-1]:
            logger.info(
                f"Too large starting radius (r0 = {pos0[0]/nc.au:.2f} au). "
                + f"Use r0 = max(r_ax) = {self.r_ax[-1]/nc.au:.2f} au instead."
            )
            pos0 = [self.r_ax[-1], pos0[1]]

        res = integrate.solve_ivp(
            self._func,
            (self.t_eval[0], self.t_eval[-1]),
            pos0,
            method=self.method,
            events=self._hit_midplane,
            t_eval=self.t_eval,
            rtol=self.rtol,
        )
        self.add_streamline(res)

    def _func(self, t, pos):
        vr = self.vr_field(pos)[0]
        vt = self.vt_field(pos)[0]
        if np.isnan(pos[0]):
            raise Exception
        if  np.pi / 2 < pos[1]:
            vt *= -1
        return np.array([vr, vt / pos[0]])

    @staticmethod
    def _hit_midplane(t, pos):
        return np.pi / 2 - pos[1]

    def add_streamline(self, res):
        R = res.y[0] * np.sin(res.y[1])
        z = res.y[0] * np.cos(res.y[1])
        vr = self.vr_field(res.y.T)
        vt = self.vt_field(res.y.T)
        vR = np.sin(res.y[1]) * vr + np.cos(res.y[1]) * vt
        vz = np.cos(res.y[1]) * vr - np.sin(res.y[1]) * vt
        sl = Streamline(res.y[:, 0], res.t, R, z, vR, vz)
        for name, value, unitname in self.value_list:
            vint = self._interpolate_along_streamline(value, res.y)
            sl.add_value(name, vint, unitname)
        self.streamlines.append(sl)

    def _interpolate_along_streamline(self, value, points):
        return interpolate.interpn(
            (self.r_ax, self.t_ax),
            value,
            points.T,
            bounds_error=False,
            fill_value=None,
        )


def save_data(streamlines, filename="stream", dpath=None):
    global run_dir
    if dpath is None:
        dpath = run_dir
    os.makedirs(dpath, exist_ok=True)

    for sl in streamlines:
        label = f"r{sl.pos0[0]/nc.au:.0f}_th{np.rad2deg(sl.pos0[1]):.0f}"
        header = "t [s]  R [cm]  z [cm]  vr [cm/s]  vt [cm/s] "
        values = []
        for name, value, unitname in sl.get_values():
            header += f"{name} [{unitname}]"
            values.append(value[...,0])

        print( sl.vR, sl.vz, *values )
        stream_data = np.stack(
            (sl.t, sl.R, sl.z, sl.vR, sl.vz, *values), axis=-1
        )

        np.savetxt(
            f"{dpath}/{filename}_{label}.txt",
            stream_data,
            header=header,
            fmt="%.18e",
        )


def make_streamline_data(model, r0, theta0, t_eval, rtol=1e-4, method="RK23"):
    slc = StreamlineCalculator2(model, t_eval=t_eval, rtol=rtol, method=method, save=True)
    slc.calc_streamline(r0, theta0)
    slc.save_data()

class StreamlineCalculator2:
    """

    example
    ------------
    slc = StreamlineCalculator2(model, t_eval=np.arange(0, 1e12, 1e8), rtol=1e-4, method="RK23", save=True)
    slc.calc_streamline(r0=1000 au, theta0=80)
    slc.save_data()

    """
    def __init__(
        self,
        model,
        t_eval=np.geomspace(1, 1e30, 500),
        rtol=1e-8,
        method="RK45",
        save=False
    ):
        self.r_ax = model.rc_ax
        self.t_ax = model.tc_ax
        self.vr_field = interpolate.RegularGridInterpolator(
            (self.r_ax, self.t_ax), model.vr[...,0], bounds_error=False, fill_value=None
        )
        self.vt_field = interpolate.RegularGridInterpolator(
            (self.r_ax, self.t_ax), model.vt[...,0], bounds_error=False, fill_value=None
        )
        self.t_eval = t_eval
        self.rtol = rtol
        self.method = method
        self.streamlines = []
        self._hit_midplane.terminal = True
        self.save = save

        self.value_list = []
        if hasattr(model, "rhogas"):
            self.add_value("rhogas", model.rhogas, "g cm^-3")
        if hasattr(model, "Tgas"):
            self.add_value("Tgas", model.Tgas, "K")

    def add_value(self, name, value, unitname):
        self.value_list.append([name, value, unitname])

    def calc_streamline(self, r0, theta0):
        if r0 > self.r_ax[-1]:
            logger.info(
                f"Too large starting radius (r0 = {r0/nc.au:.2f} au). "
                + f"Use r0 = max(r_ax) = {self.r_ax[-1]/nc.au:.2f} au instead."
            )
            pos0 = (self.r_ax[-1], theta0)
        else:
            pos0 = (r0, theta0)

        if self.t_eval is None:
            t_range = (1, 1e30)
        else:
            t_range = (self.t_eval[0], self.t_eval[-1])

        res = integrate.solve_ivp(
            self._func,
            t_range,
            np.array(pos0),
            method=self.method,
            events=self._hit_midplane,
            t_eval=self.t_eval,
            rtol=self.rtol,
        )
        self.add_streamline(res)

    def _func(self, t, pos):
        vr = self.vr_field(pos)[0]
        vt = self.vt_field(pos)[0]
        if np.isnan(pos[0]):
            raise Exception
        if np.pi / 2 < pos[1]:
            vt *= -1
        return np.array([vr, vt / pos[0]])

    @staticmethod
    def _hit_midplane(t, pos):
        return np.pi / 2 - pos[1]

    def add_streamline(self, res):
        R = res.y[0] * np.sin(res.y[1])
        z = res.y[0] * np.cos(res.y[1])
        vr = self.vr_field(res.y.T)
        vt = self.vt_field(res.y.T)
        vR = np.sin(res.y[1]) * vr + np.cos(res.y[1]) * vt
        vz = np.cos(res.y[1]) * vr - np.sin(res.y[1]) * vt
        sl = Streamline(res.y[:, 0], res.t, R, z, vR, vz)
        for name, value, unitname in self.value_list:
            vint = self._interpolate_along_streamline(value, res.y)
            sl.add_value(name, vint, unitname)
        self.streamlines.append(sl)

    def _interpolate_along_streamline(self, value, points):
        return interpolate.interpn(
            (self.r_ax, self.t_ax),
            value,
            points.T,
            bounds_error=False,
            fill_value=None,
        )

    def save_data(self, filename="stream", dpath=None):
        global run_dir
        if dpath is None:
            dpath = run_dir
        os.makedirs(dpath, exist_ok=True)

        for sl in self.streamlines:
            label = f"r{sl.pos0[0]/nc.au:.0f}_th{np.rad2deg(sl.pos0[1]):.0f}"
            header_list = ["t [s]", "R [cm]", "z [cm]", "vr [cm/s]", "vt [cm/s]"]
            values = []
            for name, value, unitname in sl.get_values():
                header_list.append(f"{name} [{unitname}]")
                values.append(value[...,0])
            header = " ".join([hd.rjust(19) for hd in header_list])
            stream_data = np.stack(
                (sl.t, sl.R, sl.z, sl.vR, sl.vz, *values), axis=-1
            )

            np.savetxt(
                f"{dpath}/{filename}_{label}.txt",
                stream_data,
                header=header[2:],
                fmt="%19.12e",
            )
