#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess, sys
import argparse
import pmodes.mytools as mytools
import argcomplete
import numpy as np
import re
from pathlib import Path
import itertools

parser = argparse.ArgumentParser(description='This code executes a pipeline making from a model to figures. This works like Makefile.')
parser.add_argument('targets', nargs='*', default="default")
#parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('-r','--dryrun',action='store_true')
parser.add_argument("-n",'--nohup',action='store_true')
argcomplete.autocomplete(parser)
args = parser.parse_args()

PYCMD = "python"
SRCDIR = "./pmodes"
RADMCDIR = "./radmc"
INPUT_FILE = "L1527.in"
RMCMD = "rm -fv"

mytools.exe.dryrun = True

def main():
    for target in args.targets:
    
        if target == "default":
            Execute("all")
    
        elif target in ["all", "mkmodel", "setradmc", "sobs", "visualize"]:
            Execute(target)
    
        elif target == "params":
            #params_list = []
            #params_list.append( make_params("model.ire_model","ire", ["Simple", "CM"]))
            #params_list.append( make_params("radmc.nr","nr", [64, 128, 256]))
            #params_list.append( make_params("fitsa.convolution_pvdiagram", "conv", [0,1]))
            params_list = make_params_list([["model.ire_model", "ire", ["Simple", "CM"]],
                                           ["radmc.nr", "nr", [64, 128, 256]],
                                           ["beam_scale", "bsc", [0.1,0.3,1,3]], 
                 #                          ["fitsa.convolution_pvdiagram", "conv", [0,1]]],
                                          )
    
            parameter_survey("all", params_list, INPUT_FILE )
    
        elif target == "clean":
            for tar in ["cache","subproc","radmc","fits","pkl"]:
              clean(tar)

        elif "clean_" in target:
            clean(target.replace("clean_","")) 
    
        elif target == "kill_process":
            pass

        else:
            exit()



class Parameter:
    def __init__(self, name, short_name, value):
        self.name = name
        self.short_name = short_name
        self.value = value

def make_params(name, shortname, values):
    return [ Parameter(name, shortname, v) for v in values ]

def make_params_list( param_info_list ):
    return [ make_params(name, shortname, values) for name, shortname, values in param_info_list]

#### somethoiung worng 
def parameter_survey( mode , params, ori_inpfn ):
 #       [ [param_a_1,..], [param_b_1,..], [param_c_1,..] ]
    for param_set in itertools.product(*params):
        target_chars = [ par.name.replace(".","\.") + "\s*=\s*" for par in param_set ]
        new_chars = [f"{par.name} = {par.value}  # original:" for par in param_set ]
        tmp_fname = f"{ori_inpfn}.tmp_" + "_".join([ f"{par.short_name}={par.value}" for par in param_set ])
        replace_inputfile2(target_chars, new_chars, ori_inpfn, tmp_fname)
        Execute(mode, inputfile=tmp_fname)
        mytools.exe('cp -r fig fig_'+"_".join([f"{par.short_name}={par.value}" for par in param_set]) )

class Execute:
#    def __init__(self, py="python", src_dir="", inputfile="", nohup=False):    
    def __init__(self, mode=None, py=PYCMD, src_dir=SRCDIR, inputfile=INPUT_FILE, nohup=args.nohup):
        self.py = py
        self.src_dir = src_dir
        self.inputfile = inputfile
        self.nohup_cmd = ["nohup","&"] if nohup else ["",""]

        if mode is not None:
            if mode == "mkmodel":
                self.mkmodel()

            elif mode == "setradmc":
                self.setradmc()

            elif mode == "sobs":
                self.sobs()

            elif mode == "visualize":
                self.visualize()

            elif mode == "all":
                self.all()

            else:
                raise Exception("No sunch mode.")
    

    def _cmd(self, command):
         mytools.exe(f'{self.nohup_cmd[0]} {self.py} {self.src_dir}/{command}.py {self.inputfile} {self.nohup_cmd[1]}')

    def mkmodel(self):
        self._cmd("mkmodel")

    def setradmc(self):
        self._cmd("setradmc")

    def sobs(self):
        self._cmd("sobs")

    def visualize(self):
        self._cmd("visualize")

    def all(self):
        self.mkmodel()
        self.setradmc()
        self.sobs()
        self.visualize()



def replace_inputfile2(target_chars, new_chars, original_file, temp_file):
    with open(original_file, mode='r') as f:
        filetxt = f.read()
    for tch, nch in zip(target_chars, new_chars):
        filetxt = re.sub(tch, nch, filetxt)
    with open(temp_file, mode='w') as f:
        f.write(filetxt)


def clean(target, rm_cmd=RMCMD, src_dir=SRCDIR, radmc_dir=RADMCDIR):
    def getpaths(target_dir,patterns=[],exclude=[]):
        def match_exclude(x):
            return any([re.fullmatch(exc, x) for exc in exclude ])
        p = Path(target_dir)
        answer=[]
        for pat in patterns:
            answer += [ str(x) for x in p.glob(pat) if not match_exclude(x.name)] 
        return answer    

    def rm(target_list, option=""):
        target = " ".join(target_list)
        mytools.exe(f'echo {rm_cmd} {option} {target}')

    if target == "cache":
        rm(getpaths(src_dir, ["*.pyc", "__pycache__"]))

    if target == "subproc":
        rm(getpaths(radmc_dir, ["proc*"] ), option="-d")

    if target == "radmc":
        rm(getpaths(radmc_dir, ["*.inp"], exclude=[r"molecule_.*", r"dustkappa_.*"]))

    if target == "fits":
        rm(getpaths(radmc_dir, ["*.fits"]))

    if target == "pkl":
        rm(getpaths(radmc_dir, ["*.pkl"]))

if __name__=='__main__':
    main()
