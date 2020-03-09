from __future__ import print_function,  absolute_import, division
import subprocess
import os
import numpy as np

## in input files
def get_square_ntheta(r_range, nr):
    dlogr = np.log10(r_range[1]/r_range[0])/nr
    dr_over_r = 10**dlogr -1
    return int(round(0.5*np.pi/dr_over_r))

class Parameters:
    def __init__(self, subclass_name_list=None):
        if subclass_name_list is not None:
            for name in subclass_name_list:
                setattr(self, name, Subclass() )
    class Subclass:
        pass

def set_arguments(cls, locals_dict, printer=print):
    for k, v in locals_dict.items():
        if (k != 'self') and (k!="kwargs"):
            setattr(cls, k, v)
            printer(f"{k:20} is {str(v):20}")

def freq_to_vkms(freq0, dfreq):
    import cst
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
