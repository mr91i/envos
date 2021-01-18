import numpy as np
from dataclasses import dataclass

def calc_streamlines(r_ax, t_ax, vr, vt, physical_value_dict):
    """
    physical_value_dict: dict
        key: str
            name of physical value
        value: list
            values
    """



class Streamlines:
    def __init__(self, r_ax, t_ax, vr, vt ):
        self.r_ax = r_ax
        self.t_ax = t_ax
        self.slines = []
        self.val_list = []

    def set_pos0list(self, r0_list, theta0_list):
        self.pos0list = [(r0, th0) for r0 in r0_list for th0 in theta0_list]

    def set_vfield(self, r_ax, t_ax, vr, vt):
        self.vr_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vr, bounds_error=False, fill_value=None)
        self.vt_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vt, bounds_error=False, fill_value=None)

    def add_value(self, name, val, unit):
        valfunc = interpolate.RegularGridInterpolator((self.r_ax, self.t_ax), val, bounds_error=False, fill_value=None)
        self.val_list.append([ name, valfunc, unit])

    def calc_streamlines(self, dpath):
        sl = Streamline(self.r_ax, self.t_ax, self.vr_field, self.vt_field, self.t_span)
        for name, valfunc, unit in self.val_list:
            sl.add_value(name, valfunc, unit)

        for _pos0 in self.pos0_list:
            sl.calc_streamline(_pos0)
            sl.save_data(dpath)

class Streamline:
    def __init__(self, r_ax, th_ax, vrf, vtf, pos0, t_span=(0, 1e30), nt=500, rtol=1e-4):
        self.name = "{pos0[0]/nc.au:.2f}_{np.deg2rad(pos0[1]):.2f}"
        self.vr_field = vrf
        self.vt_field = vtf
        self.value_dict = {}
        self.t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), nt)
        self.rtol = rtol


    def add_value(self, value_dict):
        #self.vnames.append(name)
        #self.vfuncs.append(interpfunc)
        #self.vunits.append(unit)
        self.value_dict.update(value_dict)

    def calc_streamline(self):
        if pos0[0] > self.r_ax[-1]:
            looger.info(f'Too large position:r0 = {pos0[0]/nc.au} au. r0 must be less than {r_ax[-1]/nc.au} au. I use r0 = {r_ax[-1]/nc.au} au instead of r0 = {pos0[0]/nc.au} au')
            pos0 = [self.r_ax[-1], pos0[1]]

        def func(t, pos, hit_flag=0):
            if hit_midplane(t, pos) < 0:
                hit_flag = 1
            vr = self.vr_field(pos)
            vt = self.vt_field(pos)
            return np.array([vr, vt/pos[0]])

        def hit_midplane(t, pos):
            return np.pi/2 - pos[1]
        hit_midplane.terminal = True

        pos = integrate.solve_ivp(
                func,
                (self.t_eval[0], self.t_eval[-1]),
                self.pos0,
                method='RK45',
                events=hit_midplane,
                t_eval=self.t_eval,
                rtol=self.rtol)
        pos.R = pos.y[0] * np.sin(pos.y[1])
        pos.z = pos.y[0] * np.cos(pos.y[1])
        self.pos = pos
        #self.values = [f(pos) for f in self.vfuncs]
        for v in self.value_dict.values():
            vline = self._interpolate_along_streamline(v, pos.y)
            setattr(

    def _interpolate_along_streamline(self, value, points):
         return interpolate.interpn((self.r_ax, self.t_ax), value, points, bounds_error=False, fill_value=None)

    def save_data(self, filename="stream", dpath=None):
        if dpath is None:
            dpath = config.dp_run
        os.makedirs(dpath, exist_ok=True)

        #vlist = [f"{v}["u"]" for v, u in zip(self.vnames, self.sunits)]
        header = " ".join('t [s]', "R [cm]", "z [cm]", *self.value_dict.keys())
        stream_data = np.stack((self.pos.t, self.pos.R, self.pos.z, *self.values()), axis=-1)
        np.savetxt(f'{dpath}/{filename}_{self.name}.txt', stream_data, header=header, fmt="%.18e")


def calc_streamlines(r_ax, th_ax, vr, vt, pos0list, valueinfo, t_span=(0, 1e30), nt=500, rtol=1e-4, filename="stream", dpath=None):
    slc = StreamlineCalculator(r_ax, th_ax, vr, vt, pos0list, t_span=t_span, nt=nt, rtol=rtol):
    for name, value, unitname in valueinfo:
        slc.add_value(self, name, value, unitname)
    slc.calc_streamlines()
    slc.save_data(self, filename=filename, dpath=dpath)

@dataclass
class Streamline:
    pos0: np.ndarray
    t: np.ndarray
    R: np.ndarray
    z: np.ndarray
    vR: np.ndarray
    vz: np.ndarray
    def add_value(self, name, value):
        setattr(self, name, value)

class StreamlineCalculator:
    def __init__(self, r_ax, th_ax, vr, vt, pos0list, t_span=(0, 1e30), nt=500, rtol=1e-4, method='RK45'):
        self.r_ax = r_ax
        self.t_ax = t_ax
        self.vr_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vr, bounds_error=False, fill_value=None)
        self.vt_field = interpolate.RegularGridInterpolator((r_ax, t_ax), vt, bounds_error=False, fill_value=None)
        self.pos0list = pos0list
        self.t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), nt)
        self.rtol = rtol
        self.method = method
        self.names = []
        self.values = []
        self.unitnames = []
        self.streamlines = []
        self.hit_midplane.terminal = True

    def add_value(self, name, value, unitname):
        #valfunc = interpolate.RegularGridInterpolator((self.r_ax, self.t_ax), val, bounds_error=False, fill_value=None)
        #self.val_list.append([ name, valfunc, unit])
        self.names.append(name)
        self.values.append(value)
        self.unitnames.append(unitname)

    def calc_streamlines(self):
        for pos0 in self.pos0list:
            self.calc_streamline(pos0)

    def calc_streamline(self, pos0):
        if pos0[0] > self.r_ax[-1]:
            looger.info(f'Too large position:r0 = {pos0[0]/nc.au} au. r0 must be less than {r_ax[-1]/nc.au} au. I use r0 = {r_ax[-1]/nc.au} au instead of r0 = {pos0[0]/nc.au} au')
            pos0 = [self.r_ax[-1], pos0[1]]

        res = integrate.solve_ivp(
                self._func,
                (self.t_eval[0], self.t_eval[-1]),
                self.pos0,
                method=self.method,
                events=self.hit_midplane,
                t_eval=self.t_eval,
                rtol=self.rtol)
        self.add_streamline(res)

    def _func(t, pos, hit_flag=0):
        if hit_midplane(t, pos) < 0:
            hit_flag = 1
        vr = self.vr_field(pos)
        vt = self.vt_field(pos)
        return np.array([vr, vt/pos[0]])

    def _hit_midplane(t, pos):
        return np.pi/2 - pos[1]

    def add_streamline(self, res):
        R = res.y[0] * np.sin(res.y[1])
        z = res.y[0] * np.cos(res.y[1])
        vr = self.vr_field(res.y)
        vt = self.vt_field(res.y)
        vR = np.sin(res.y[1])  *vr + np.cos(res.y[1]) * vt
        vz = np.cos(res.y[1]) * vr - np.sin(res.y[1]) * vt
        sl = Streamline(res.y[:,0], res.t, R, z, vR, vz)
        for name, value in zip(self.names, self.values):
            vint = self._interpolate_along_streamline(value, res.y)
            sl.add_value(name, vint)
        self.streamlines.append(sl)

    def _interpolate_along_streamline(self, value, points):
         return interpolate.interpn((self.r_ax, self.t_ax), value, points, bounds_error=False, fill_value=None)

    def save_data(self, filename="stream", dpath=None):
        if dpath is None:
            dpath = config.dp_run
        os.makedirs(dpath, exist_ok=True)

        for sl in self.streamlines:
            label = "{sl.pos0[0]/nc.au:.2f}_{np.deg2rad(sl.pos0[1]):.2f}"
            vlist = [f"{v} ["u"]" for v, u in zip(self.vnames, self.unitnames)]
            header = " ".join('t [s]', "R [cm]", "z [cm]", *self.value_dict.keys())
            stream_data = np.stack((self.pos.t, self.pos.R, self.pos.z, *self.values()), axis=-1)
            np.savetxt(f'{dpath}/{filename}_{label}.txt', stream_data, header=header, fmt="%.18e")
