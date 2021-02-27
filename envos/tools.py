import subprocess
import os
import sys
import shutil
import numpy as np
import envos.nconst as nc
import pandas

from envos.log import set_logger

logger = set_logger(__name__)

## in input files

# def mkdir(dpath):
#    os.mkdirs(dpath, exist_ok=True)



def read_pickle(filepath):
    logger.info("Reading pickle data")
    cls = pandas.read_pickle(filepath)
    logger.info(cls)
    return cls


#######

def freq_to_vkms_array(freq, freq0):
    return nc.c / 1e5 * (freq0 - freq) / freq0


def freq_to_vkms(freq0, dfreq):
    return nc.c / 1e5 * dfreq / freq0


def make_array_center(xi):
    return 0.5 * (xi[0:-1] + xi[1:])


def make_array_interface(xc):
    return np.concatenate(
        [
            [xc[0] - 0.5 * (xc[1] - xc[0])],
            0.5 * (xc[0:-1] + xc[1:]),
            [xc[-1] + 0.5 * (xc[-1] - xc[-2])],
        ],
        axis=0,
    )

def x_cross_zero(x1, x2, y1, y2):
    return x1 + y1 * (x2 - x1) / (y1 - y2)


def find_roots(x, y1, y2):
    dy = np.array(y1) - np.array(y2)
    n = len(x) - 1
    return np.array(
        [
            x_cross_zero(x[i], x[i + 1], dy[i], dy[i + 1])
            for i in range(n)
            if dy[i] * dy[i + 1] <= 0
        ]
    )

def show_used_memory():
    import psutil
    mem = psutil.virtual_memory()
    logger.info("Used memory = %.3f GiB", mem.used/(1024**3))

def compute_object_size(o, handlers={}):
    import sys
    from itertools import chain
    from collections import deque

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def shell(
    cmd,
    cwd=None,
    log=True,
    dryrun=False,
    skip_error=False,
    error_keyword=None,
    logger=logger,
    log_prefix="",
    simple=False,
):

    error_flag = 0
    msg = ""

    if dryrun:
        logger.info(f"(dryrun) {cmd}")
        return 0

    if cwd is not None:
        pass
        #msg += f"Change the working directory from {os.getcwd()} to {cwd}"
        #msg += " while executing the command"
    else:
        cwd = os.getcwd()

    logger.info(f'Running shell command at {cwd}:\n    "{cmd}"')

    if simple:
        return subprocess.run(cmd, shell=True, cwd=cwd)
    else:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    while True:
        _line = proc.stdout.readline().rstrip()  # .decode("utf-8").rstrip()
        if _line:
            if log:
                logger.info(log_prefix + _line)
            if (error_keyword is not None) and (error_keyword in _line):
                error_flag = 1

        if (not _line) and (proc.poll() is not None):
            if log:
                logger.info("")
            break

    retcode = proc.wait()

    if (retcode != 0) or (error_flag == 1):
        e = subprocess.CalledProcessError(retcode, cmd)
        if skip_error:
            logger.warning("Skip error:")
            logger.warning("    %s" % e)
        else:
            logger.error(e)
            raise e


def filecopy(src, dst, error_already_exist=False):
    dpath_src = os.path.abspath(os.path.dirname(src))
    fname_src = os.path.basename(src)
    dpath_dst = os.path.abspath(os.path.dirname(dst))

    logger.info(f"Copying {fname_src} from {dpath_src} into {dpath_dst}")

    if os.path.exists(dst):
        msg = f"File already exists: {dst}"
        if error_already_exist:
            logger.error(msg)
            raise Exception(msg)
        else:
            logger.warn(msg)
            logger.warn("Do nothing.")

    try:
        shutil.copy2(src, dst)

    except FileNotFoundError as err:
        msg = ""
        if not os.path.exists(dpath_src):
            msg = f"Source directory not found: {dpath_src}"
        elif not os.path.exists(src):
            fpath_src = os.path.abspath(src)
            msg = f"Source file not found: {fpath_src}"
        elif not os.path.exists(dpath_dst):
            msg = f"Destination directory not found: {dpath_dst}"
        logger.error(msg)
        raise

    except Exception as e:
        logger.error(e)
        raise

    else:
        logger.debug("Sucsess copying")

# def _instance_pickle(filepath, base=None):
#     if ".pkl" in filepath:
#         dtype = "pickle"
#
#     if dtype == "pickle":
#         cls = pd.read_pickle(filepath)
#         if base is None:
#             return cls
#         else:
#             for k, v in cls.__dict__.items():
#                 setattr(base, k, v)

def setattr_from_pickle(cls, filepath):
    readcls = pandas.read_pickle(filepath)
    for k, v in readcls.__dict__.items():
        setattr(cls, k, v)


#  def shell(
#      cmd,
#      cwd=None,
#      log=True,
#      dryrun=False,
#      skiperror=False,
#      realtime=False,
#      error_keyword=None,
#  ):
#
#      if cwd is not None:
#          logger.info(f"Move the working directory from {os.getcwd()} to {cwd} temporarily.")
#      else:
#          cwd = os.getcwd()
#      try:
#          if dryrun:
#              logger.info("(dryrun) %s", cmd)
#              return 0
#
#          logger.info("shell: %s", cmd)
#
#          if realtime:
#              for line in run_and_get_lines(cmd, cwd=cwd):
#                  if log:
#                      print(line)
#                      #logger.info(line)
#                  else:
#                      pass
#              #logger.info("")
#              return 0
#
#          if log:
#              retcode = subprocess.run(cmd, shell=True, cwd=cwd, check=True)
#          else:
#              retcode = subprocess.run(
#                  cmd,
#                  shell=True,
#                  cwd=cwd,
#                  check=True,
#                  stdout=subprocess.PIPE,
#                  stderr=subprocess.PIPE,
#              )
#
#          logger.info("")
#          return retcode
#
#      except subprocess.CalledProcessError as e:
#
#          if skiperror:
#              logger.warning("Skip error:")
#              logger.warning("    %s" % e)
#
#          else:
#              logger.exception("Error generated:")
#              logger.exception(" - return code is %s" % e.returncode)
#              logger.exception(" - cmd is '%s'" % e.cmd)
#              logger.exception(" - output is %s" % e.output)
#              exit()
#
#
#  # def run_and_capture(cmd):
#  def run_and_get_lines(cmd, **kwargs):
#
#      """
#      cmd: str running command
#      :return standard output
#      """
#
#      # Start process asynchronously with the python process
#      proc = subprocess.Popen(
#          cmd,
#          shell=True,
#          stdout=subprocess.PIPE,
#          stderr=subprocess.STDOUT,
#          **kwargs
#      )
#      while True:
#          line = proc.stdout.readline() #.decode("utf-8")
#          if line:
#              yield line
#
#          if (not line) and (proc.poll() is not None):
#              break


# def interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new):
#    from scipy.interpolate import interpn # , RectBivariateSpline, RegularGridInterpolatour
#    ret0 = interpn((x_ori, y_ori, z_ori), value, np.stack([xx_new, yy_new, zz_new], axis=-1), bounds_error=False, fill_value=np.#nan)
#    return ret0

# def _interpolator2d(value, x_ori, y_ori, x_new, y_new, logx=False, logy=False, logv=False):
#    xo = np.log10(x_ori) if logx else x_ori
#    xn = np.log10(x_new) if logx else x_new
#    yo = np.log10(y_ori) if logy else y_ori
#    yn = np.log10(y_new) if logy else y_new
#    vo = np.log10(np.abs(value)) if logv else value
#    fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
#    ret0 = fv(xn, yn)
#    if logv:
#        if (np.sign(value)!=1).any():
#            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
#            sgn = np.sign(fv_sgn(xn, yn))
#            ret = np.where(sgn!=0, sgn*10**ret0, 0)
#        else:
#            ret = 10**ret0
#    else:
#        ret = ret0
#    return np.nan_to_num(ret0)
#
# def _interpolator3d(value, x_ori, y_ori, z_ori, xx_new, yy_new, zz_new, logx=False, logy=False, logz=False, logv=False):
#    if len(z_ori) == 1:
#        value = _interpolator2d(value, x_ori, y_ori, xx_new, yy_new, logx=False, logy=False, logv=False)
#        return value
##        return
#
#    from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
#    def points(*xyz):
#        return [[(v, r*np.sin(posang_PV_rad), r*np.cos(posang_PV_rad))
#                       for r in self.xau ] for v in self.vkms]
#        #return np.array(list(itertools.product(xyz[0], xyz[1], xyz[2])))
#    xo = np.log10(x_ori) if logx else x_ori
#    yo = np.log10(y_ori) if logy else y_ori
#    zo = np.log10(z_ori) if logz else z_ori
#    xn = np.log10(x_new) if logx else xx_new
#    yn = np.log10(y_new) if logy else yy_new
#    zn = np.log10(z_new) if logz else zz_new
#    vo = np.log10(np.abs(value)) if logv else value
#    print(np.stack([xn, yn, zn], axis=-1), xo, yo, zo )
#
#
#    ret0 = RegularGridInterpolator((xo, yo, zo), vo, bounds_error=False, fill_value=-1 )( np.stack([xn, yn, zn], axis=-1))
#    print(ret0, np.max(ret0) )
#    exit()
#    #fv = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
#    ret0 = fv(xn, yn)
#    if logv:
#        if (np.sign(value)!=1).any():
#            fv_sgn = np.vectorize(interp2d(xo, yo, value.T, fill_value=0))
#            sgn = np.sign(fv_sgn(xn, yn))
#            ret = np.where(sgn!=0, sgn*10**ret0, 0)
#        else:
#            ret = 10**ret0
#    else:
#        ret = ret0
#    return np.nan_to_num(ret0)
#
#


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
