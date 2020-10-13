# Note:
#  to supress cache files :
#      write "PYTHONDONTWRITEBYTECODE = 1" to ~/.bashrc or ~/.bash_profile
import os
import sys

## Color
class pycolor:
    BLACK = '\033[30m'
    GRAY = '\033[96m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'

## logging
import logging
class MyFormatter(logging.Formatter):

    dbg_fmt  = pycolor.GRAY + "[%(filename)s] Debug: %(message)s" + pycolor.END
    info_fmt = "[%(filename)s] %(message)s"
    warn_fmt = pycolor.YELLOW + "\n[%(filename)s] Warning: %(message)s\n" + pycolor.END
    err_fmt = pycolor.RED + "\n[%(filename)s] Error: %(message)s\n" + pycolor.END

    def __init__(self):
        #super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s", datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level

        if record.levelno == logging.DEBUG:
            self._style._fmt = MyFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = MyFormatter.warn_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = MyFormatter.err_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result

logger = logging.getLogger(__name__)
fmt = MyFormatter()
hdlr = logging.StreamHandler(sys.stdout)
hdlr.setFormatter(fmt)
logging.root.addHandler(hdlr)
logger.setLevel(logging.DEBUG)
#logging.root.setLevel(logging.DEBUG)


## argparse
import argparse
dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + '/../')
dn_radmc = dn_home + "/radmc"
dn_fig = dn_home + "/fig"
py_version = sys.version_info
if py_version[0] == 2:
    logging.error('''\n
This module does not work with Python2.
At 1 Jan 2020, most packages stopped to support Python2 (see https://python3statement.org).
Please use and update your Python3 to the newest version.
At this writing, Python 3.8.1 is the newest.''')
    exit()

parser = argparse.ArgumentParser(description='')
parser.add_argument('input_file_name', nargs='?', default="L1527.in")
parser.add_argument('-e','--edit', action='store_true')
args = parser.parse_args()
logger.info("Input file path is "+dn_home+"/"+args.input_file_name)
if args.edit:
    print("Enter edit mode with vim...")
    os.system("vim "+dn_home+"/"+args.input_file_name)

## Reading input file
#class read_inp:
#    def __init__(self, path):
#        with open(path) as f:
#            exec(f.read(), {}, self.__dict__)

#inp = read_inp(dn_home+"/"+args.input_file_name)


import types
import importlib.machinery
loader = importlib.machinery.SourceFileLoader('inp',  dn_home+"/"+args.input_file_name)
inp = types.ModuleType(loader.name)
loader.exec_module(inp)






