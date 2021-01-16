import os
import re
import glob
import shutil

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_SRC = DIR_BASE + "/envos"
DIR_RADMC = DIR_BASE + "/radmc"

def remove_glob_re(pattern, pathname, recursive=False, exclude=[]):
    paths =  glob.glob(pathname, recursive=recursive)
    for p in paths:
#    for p in glob.glob(pathname, recursive=recursive):
        if any([ (ex in p) for ex in exclude ]):
            print("skip")
        else:
            if os.path.isfile(p) and re.search(re.escape(pattern), p):
                os.remove(p)
                #print("rm file")
                #print(p)

            elif os.path.isdir(p) and re.search(re.escape(pattern), p):
                shutil.rmtree(p)
                #print("rm dir")
                #print(p)

remove_glob_re('proc', f'{DIR_RADMC}/*')
remove_glob_re('inp', f'{DIR_RADMC}/*', exclude=["molecule", "dustkappa"])
remove_glob_re('pkl', f'{DIR_RADMC}/*')
remove_glob_re('fits', f'{DIR_RADMC}/*')
remove_glob_re('out', f'{DIR_RADMC}/*')
remove_glob_re('dat', f'{DIR_RADMC}/*')
