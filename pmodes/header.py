# header
# Note:
#  to supress cache files : 
#      write "PYTHONDONTWRITEBYTECODE = 1" to ~/.bashrc or ~/.bash_profile
import os
dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + '/../')
dn_radmc = dn_home + "/radmc"
dn_fig = dn_home + "/fig"

class read_inp:
    def __init__(self, path):
        with open(path) as f:
            exec(f.read(), {}, self.__dict__)
#or 
# import ../L1527.in as inp 

inp = read_inp(dn_home+"/L1527.in")
#print("Here is %s"%dn_here)
#print("Home directory is %s"%dn_home)
