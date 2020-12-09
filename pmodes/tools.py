import subprocess
import os
import numpy as np
from pmodes import cst

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

## in input files

class Parameters:
    def __init__(self, subclass_name_list=None):
        if subclass_name_list is not None:
            for name in subclass_name_list:
                setattr(self, name, Subclass() )
    class Subclass:
        pass

def set_arguments(cls, locals_dict):
    for k, v in locals_dict.items():
        if (k != 'self') and (k!="kwargs"):
            setattr(cls, k, v)
            # logger.debug(f"{k:20} is {str(v):20}")


def freq_to_vkms_array(freq, freq0):
    return cst.c/1e5* (freq0 - freq)/freq0

def freq_to_vkms(freq0, dfreq):
    return cst.c/1e5* dfreq/freq0



def make_array_center( xi ):
    return 0.5 * ( xi[0:-1] + xi[1:] )

def make_array_interface( xc ):
        return np.concatenate([[xc[0]-0.5*(xc[1]-xc[0])],
                                0.5*(xc[0:-1]+xc[1:]),
                               [xc[-1]+0.5*(xc[-1]-xc[-2])]
                              ], axis=0)

def make_meshgrid_center(xxi, yyi, indexing="xy"):
    if indexing=="xy":
        xc = make_array_center(xxi[0,:])
        yc = make_array_center(yyi[:,0])
        return np.meshgrid(xc, yc)

    elif indexing=="ij":
        xc = make_array_center(xxi[:,0])
        yc = make_array_center(yyi[0,:])
        return np.meshgrid(xc, yc, indexing="ij")

def make_meshgrid_interface( xxc, yyc , indexing="xy"):
    if indexing=="xy":
        xi = make_array_interface(xxc[0,:])
        yi = make_array_interface(yyc[:,0])
        return np.meshgrid(xi, yi)

    elif indexing=="ij":
        xi = make_array_interface(xxc[:,0])
        yi = make_array_interface(yyc[0,:])
        return np.meshgrid(xi, yi, indexing="ij")

def x_cross_zero(x1, x2, y1, y2):
    return x1 + y1 * (x2 - x1)/(y1 - y2)

def find_roots(x, y1, y2):
    dy = np.array(y1) - np.array(y2)
    n = len(x) - 1
    return np.array([x_cross_zero(x[i], x[i+1], dy[i], dy[i+1])
                     for i in range(n)
                     if dy[i]*dy[i+1] <= 0])

def isnan_values(values):
    if np.isscalar(values):
        return np.any(np.isnan(values))
    else:
        return np.any([np.any(np.isnan(v)) for v in values])

class Message:
    def __init__(self, filename=None, debug=None):
        self.filename = os.path.basename(filename)
        self.debug = debug
#        print("Message is created.", filename, debug)

    def __call__(self, *s, **args ):
        if ("debug" in args) and args["debug"]:
            if self.debug:
                print("[debug]", end='')
            else:
                return
        else:
            if self.filename:
                print("[%s] "%self.filename, end='')
        print(*s)
        if ("exit" in args) and args["exit"]:
            exit()

def _msg(*s, **args):
    if ("debug" in args) and args["debug"]:
        if inp.debug:
            print("[debug]", end='')
        else:
            return
    else:
        print("[plotter.py]", end='')
    print(*s)
    if ("exit" in args) and args["exit"]:
        exit()


class Exe:
    def __init__(self, debug=False, dryrun=False, skiperror=False):
        self.debug = debug
        self.dryrun = dryrun
        self.skiperror = skiperror

    def __call__(self, cmd):
        try:
            if self.dryrun:
                print("[dryrun]",cmd)
                return 0
            else:
                subprocess.check_call(r'echo `date "+%Y/%m/%d-%H:%M:%S"` "      " "{}" >> .executed_cmd.txt'.format(cmd), shell=True)
                print("Execute: {}".format(cmd))
                if not self.debug:
                    retcode = subprocess.check_call(cmd, shell=True)
                return 0
                #retcode = subprocess.check_call( cmd.split() )
        except subprocess.CalledProcessError as e:
            if self.skiperror:
                print("Skipper error:")
                print("    %s"%e )
            else:
                print('Error generated:')
                print(' - return code is %s'%e.returncode)
                print(' - cmd is \'%s\''%e.cmd)
                print(' - output is %s'%e.output)
                raise Exception( e )

exe = Exe()


def interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new):
    from scipy.interpolate import interpn, RectBivariateSpline, RegularGridInterpolator
    ret0 = interpn((x_ori, y_ori, z_ori), value, np.stack([xx_new, yy_new, zz_new], axis=-1), bounds_error=False, fill_value=np.nan)
    return ret0

def _interpolator2d(value, x_ori, y_ori, x_new, y_new, logx=False, logy=False, logv=False):
    xo = np.log10(x_ori) if logx else x_ori
    xn = np.log10(x_new) if logx else x_new
    yo = np.log10(y_ori) if logy else y_ori
    yn = np.log10(y_new) if logy else y_new
    vo = np.log10(np.abs(value)) if logv else value
    fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
    ret0 = fv(xn, yn)
    if logv:
        if (np.sign(value)!=1).any():
            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
            sgn = np.sign(fv_sgn(xn, yn))
            ret = np.where(sgn!=0, sgn*10**ret0, 0)
        else:
            ret = 10**ret0
    else:
        ret = ret0
    return np.nan_to_num(ret0)

def _interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new, logx=False, logy=False, logz=False, logv=False):
    if len(z_ori) == 1:
        value = _interpolator2d(value, x_ori, y_ori, xx_new, yy_new, logx=False, logy=False, logv=False)
        return value
#        return

    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
    def points(*xyz):
        return [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
                       for r in self.xau ] for v in self.vkms]
        #return np.array(list(itertools.product(xyz[0], xyz[1], xyz[2])))
    xo = np.log10(x_ori) if logx else x_ori
    yo = np.log10(y_ori) if logy else y_ori
    zo = np.log10(z_ori) if logz else z_ori
    xn = np.log10(x_new) if logx else xx_new
    yn = np.log10(y_new) if logy else yy_new
    zn = np.log10(z_new) if logz else zz_new
    vo = np.log10(np.abs(value)) if logv else value
    print(np.stack([xn, yn, zn], axis=-1), xo, yo, zo )


    ret0 = RegularGridInterpolator((xo, yo, zo), vo, bounds_error=False, fill_value=-1 )( np.stack([xn, yn, zn], axis=-1))
    print(ret0, np.max(ret0) )
    exit()
    #fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
    ret0 = fv(xn, yn)
    if logv:
        if (np.sign(value)!=1).any():
            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
            sgn = np.sign(fv_sgn(xn, yn))
            ret = np.where(sgn!=0, sgn*10**ret0, 0)
        else:
            ret = 10**ret0
    else:
        ret = ret0
    return np.nan_to_num(ret0)





#  def exe(cmd, debug=False, dryrun=False, skiperror=False):
#      try:
#          if dryrun:
#              print(cmd)
#              return 0
#          subprocess.check_call(r'echo `date "+%Y/%m/%d-%H:%M:%S"` "      " "{}" >> .executed_cmd.txt'.format(cmd), shell=True )
#  #       subprocess.check_call('echo `date \"+\%Y/\%m/\%d-\%H:\%M:\%S\" ` \'%s\' >> .executed_cmd.txt'%cmd, shell=True )
#          print("Execute: {}".format(cmd))
#          if not debug:
#              retcode = subprocess.check_call( cmd, shell=True )
#          return 0
#          #retcode = subprocess.check_call( cmd.split() )
#      except subprocess.CalledProcessError as e:
#          if skiperror:
#              print("Skipper error:")
#              print("    %s"%e )
#          else:
#              print('Error generated:')
#              print(' - return code is %s'%e.returncode)
#              print(' - cmd is \'%s\''%e.cmd)
#              print(' - output is %s'%e.output)
#              raise Exception( e )
