# header
# Note:
#  to supress cache files : 
#      write "PYTHONDONTWRITEBYTECODE = 1" to ~/.bashrc or ~/.bash_profile
import os
dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + '/../')
dn_radmc = dn_home + "/radmc"
dn_fig = dn_home + "/fig"
import sys
py_version = sys.version_info
if py_version[0] == 2:
    print('''\n
This module does not work with Python2. 
At 1 Jan 2020, most packages stopped to support Python2 (see https://python3statement.org). 
Please use and update your Python3 to the newest version.
At this writing, Python 3.8.1 is the newest.''')
    exit()

class read_inp:
    def __init__(self, path):
        with open(path) as f:
            exec(f.read(), {}, self.__dict__)
inp = read_inp(dn_home+"/L1527.in")

#or 
#from .. import "L1527.in" as inp 

#or
#import importlib
#inp = importlib.import_module("..","L1527.in")
#inp = importlib.util.find_spec("L1527.in", path=dn_home)
#print(inp)

#print("Here is %s"%dn_here)
#print("Home directory is %s"%dn_home)
