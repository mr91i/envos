#
#OBUIOB
#        print(target_list)
#
#        if targets in ["*", "/","/*"]:
#            raise Exception("Danger!!", targets)
#
#        if targets == "":
#            print("No targets")
#            return
#
#def getpaths(target_dir,patterns=[],exclude=[]):
#    def match_exclude(x):
#        return any([re.fullmatch(exc, x) for exc in exclude ])
#    p = Path(target_dir)
#    answer=[]
#    for pat in patterns:
#        answer += [ str(x) for x in p.glob(pat) if not match_exclude(x.name)]
#    return answer
#
#

import os
import re
import glob
import shutil

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_SRC = DIR_BASE + "/pmodes"
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




#taget_list =

#for target in target_list:
#    if os.path.isfile(target):
#        os.remove(target)









#<F2>def clean(clean_target, rm_cmd=RMCMD, base_dir=BASEDIR, src_dir=SRCDIR, radmc_dir=RADMCDIR):
#<F2>    def getpaths(target_dir,patterns=[],exclude=[]):
#<F2>        def match_exclude(x):
#<F2>            return any([re.fullmatch(exc, x) for exc in exclude ])
#<F2>        p = Path(target_dir)
#<F2>        answer=[]
#<F2>        for pat in patterns:
#<F2>            answer += [ str(x) for x in p.glob(pat) if not match_exclude(x.name)]
#<F2>        return answer
#<F2>
#<F2>    def rm(target_list, option=""):
#<F2>        print(target_list)
#<F2>        targets = " ".join(target_list)
#<F2>    def getpaths(target_dir,patterns=[],exclude=[]):
#<F2>        def match_exclude(x):
#<F2>            return any([re.fullmatch(exc, x) for exc in exclude ])
#<F2>        p = Path(target_dir)
#<F2>        answer=[]
#<F2>        for pat in patterns:
#<F2>            answer += [ str(x) for x in p.glob(pat) if not match_exclude(x.name)]
#<F2>        return answer
#<F2>        if targets in ["*", "/","/*"]:
#<F2>            raise Exception("Danger!!", targets)
#<F2>        if targets == "":
#<F2>            print("No targets")
#<F2>            return
#<F2>
#<F2>        for target in target_list:
#<F2>            if os.path.isfile(target):
#<F2>                os.remove(target)
#<F2>        #tools.exe(f'{RMCMD} {option} {targets}')
#<F2>
#<F2>    if clean_target == "cache":
#<F2>        rm(getpaths(src_dir, ["*.pyc", "__pycache__"]))
#<F2>
#<F2>    elif clean_target == "subproc":
#<F2>        rm(getpaths(radmc_dir, ["proc*"] ), option="-r")
#<F2>
#<F2>    elif clean_target == "radmc":
#<F2>        rm(getpaths(radmc_dir, ["*.inp"], exclude=[r"molecule_.*", r"dustkappa_.*"]))
#<F2>
#<F2>    elif clean_target == "fits":
#<F2>        rm(getpaths(radmc_dir, ["*.fits"]))
#<F2>
#<F2>    elif clean_target == "pkl":
#<F2>        rm(getpaths(radmc_dir, ["*.pkl"]))
#<F2>
#<F2>    elif clean_target == "tmpinp":
#<F2>        rm(getpaths(base_dir, ["*.in.tmp*"]))
#<F2>        rm(getpaths(base_dir, ["fig_*"]), option="-r")
#<F2>
#<F2>    else:
#<F2>        raise Exception("Unknown clean target name: ", clean_target)
