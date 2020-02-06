from __future__ import print_function,  absolute_import, division
import subprocess
import os


#def msg_maker(filename):
#    return m 
def freq_to_vkms(freq0, dfreq):
    import cst
    return cst.c/1e5* dfreq/freq0


class Message:
    def __init__(self, filename=None, debug=None):
        self.filename = os.path.basename(filename)
        self.debug = debug
        print("Message is created.", filename, debug)

    def __call__(self, *s, **args ):
        if ("debug" in args) and args["debug"]:
            if self.debug:
                print("[debug] ", end='')
            else:
                return
        else:
            if self.filename:
                print("[%s] "%self.filename, end='')
        print(*s)
        if ("exit" in args) and args["exit"]:
            exit()

def _msg(*s, **args):
#    import inspect
#    import os
#    print(os.path.basename() )
    if ("debug" in args) and args["debug"]:
        if inp.debug:
            print("[debug] ", end='')
        else:
            return
    else:
        print("[plotter.py] ", end='')
    print(*s)
    if ("exit" in args) and args["exit"]:
        exit()



def exe(cmd, debug=False, dryrun=False, skiperror=False):
    try:
        if dryrun:
            print(cmd)
            return 0
        subprocess.check_call(r'echo `date "+%Y/%m/%d-%H:%M:%S"` "      " "{}" >> .executed_cmd.txt'.format(cmd), shell=True )
#       subprocess.check_call('echo `date \"+\%Y/\%m/\%d-\%H:\%M:\%S\" ` \'%s\' >> .executed_cmd.txt'%cmd, shell=True )
        print("Execute: {}".format(cmd))
        if not debug:
            retcode = subprocess.check_call( cmd, shell=True )
        return 0
        #retcode = subprocess.check_call( cmd.split() )
    except subprocess.CalledProcessError as e:
        if skiperror:
            print("Skipper error:")
            print("    %s"%e )
        else:
            print('Error generated:')
            print(' - return code is %s'%e.returncode)
            print(' - cmd is \'%s\''%e.cmd)
            print(' - output is %s'%e.output)
            raise Exception( e )
