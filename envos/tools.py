import subprocess
import os
import sys
import numpy as np
import envos.nconst as nc

from envos.log import set_logger

logger = set_logger(__name__)

## in input files

# def mkdir(dpath):
#    os.mkdirs(dpath, exist_ok=True)

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


def shell(
    cmd,
    cwd=None,
    log=True,
    dryrun=False,
    skiperror=False,
    error_keyword=None,
    logger=logger,
    log_prefix="",
    simple=False,
):

    error_flag = 0

    if cwd is not None:
        logger.info(
            f"Move the working directory from {os.getcwd()} to {cwd} temporarily."
        )
    else:
        cwd = os.getcwd()

    logger.info(f'Running shell command: "{cmd}"')

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

    if dryrun:
        logger.info(f"(dryrun) {cmd}")
        return 0

    while True:
        line = proc.stdout.readline().rstrip()  # .decode("utf-8").rstrip()
        if line:
            logger.info(log_prefix + line)
            if (error_keyword is not None) and (error_keyword in line):
                error_flag = 1

        if (not line) and (proc.poll() is not None):
            break

    retcode = proc.wait()

    if (retcode != 0) or (error_flag == 1):
        e = subprocess.CalledProcessError
        if skip_error:
            logger.warning("Skip error:")
            logger.warning("    %s" % e)
        else:
            logger.exception(e)
            raise e


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
