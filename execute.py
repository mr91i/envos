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
import pmodes.cst as cst
import os


parser = argparse.ArgumentParser(description='This code executes a pipeline making from a model to figures. This works like Makefile.')
parser.add_argument('targets', nargs='*', default="default")
#parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('-r','--dryrun',action='store_true')
parser.add_argument("-n",'--nohup',action='store_true')
parser.add_argument("-i",'--input',default="L1527.in")
parser.add_argument("-s",'--show',action='store_true')
parser.add_argument('--skipifexist',action='store_true')
parser.add_argument("-c",'--clean',action='store_true')
argcomplete.autocomplete(parser)
args = parser.parse_args()
targets = args.targets
if isinstance(targets, str):
    targets = [targets]

if args.show:
    r = subprocess.check_output("cat execute.py|grep -oP '(?<=if target == \").+(?=\":)'", shell=True).decode("utf-8").replace("\n"," ")
    print("Variable target name:", r)
    targets = [input("Input target name:")]

PYCMD = "python"
BASEDIR = "."
SRCDIR = BASEDIR + "/pmodes"
RADMCDIR = BASEDIR + "/radmc"
INPUT_FILE = args.input
RMCMD = "rm -fv"
mytools.exe.dryrun = True if args.dryrun else False

def main():
    global targets
    if args.clean:
        targets = ["clean"] + targets

    if args.nohup:
        opt_line = " ".join([f"--{k} {'' if v is True else v}" for k, v in vars(args).items()
                            if (k not in ["targets","nohup"]) and (v is not False)])
        cmd = f"{__file__} "+  opt_line + " "+" ".join(targets)
        mytools.exe(f"nohup {PYCMD} {cmd} >> nohup.out &")
        exit()

    for target in targets:
        if target == "default":
            Execute("all")

        elif target in ["all", "mkmodel", "setradmc", "sobs", "visualize"]:
            Execute(target)

        elif target == "resolution":
            dpc =137
            params_list = make_params_list([["radmc.nr", "nr", [128, 256, 512]],
                                            ["sobs.dv_kms", "dv", [0.02,0.04,0.08,0.16]],
                                            ["sobs.pixsize_au", "pix", [0.02*dpc,0.04*dpc,0.08*dpc, 0.16*dpc] ],
                                            ["beam_scale", "bs",[0.1]],
                                           ])

            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "iline":
            params_list = make_params_list([["radmc.mol_name", "", ["c18o"]],
                                            ["iline", "il", [1,2,3,4,5,6]],
                                            ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "test_temp":
            params_list = make_params_list([["model.ire_model", "", ["CM"]],
                                            ["radmc.temp_mode", "", ["lambda"]],
                                            ["T0_lam", "T0", [100]],
                                            ["qT_lam", "qT", [-.5]],
                                            ["iline", "il", [3]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

            exit()

            params_list = make_params_list([["model.ire_model", "", ["Simple", "CM"]],
                                            ["radmc.temp_mode", "", ["lambda"]],
                                            ["T0_lam", "T0", [3, 30, 100, 300]],
                                            ["qT_lam", "qT", [-.5]],
                                            ["iline", "il", [3]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)
            params_list = make_params_list([["model.ire_model", "", ["Simple", "CM"]],
                                            ["radmc.temp_mode", "", ["lambda"]],
                                            ["T0_lam", "T0", [30]],
                                            ["qT_lam", "qT", [0, -.25, -.5, -1, -2]],
                                            ["iline", "il", [3]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)
            params_list = make_params_list([["model.ire_model", "", ["Simple", "CM"]],
                                            ["radmc.temp_mode", "", ["lambda"]],
                                            ["T0_lam", "T0", [30]],
                                            ["qT_lam", "qT", [-.5]],
                                            ["iline", "il", [1, 3, 5, 7, 20]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "PS_CM":
            Tenv_K = [30]
            rCR_au = round_sig_array(100 * 10**np.linspace(-0.5, 0.5, 6), 3)
            Mstar_Msun = round_sig_array(1 * 10**np.linspace(-1, 0, 6), 3)
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["CM"]],
                                            ["beam_scale", "bs", [0.125, 0.25, 0.5, 2]],
                                            #["fitsa.vwidth_kms", "vw", [0.05, 0.1, 0.2, 0.8]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [45]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)


        elif target == "model_survey":
            #Tenv_K = 30 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #rCR_au = 200 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #Mstar_Msun = 0.18 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            Tenv_K = 30 * 10**np.linspace(-0.5, 0.5, 11)
            rCR_au = 200 * 10**np.linspace(-0.5, 0.5, 11)
            Mstar_Msun = 0.2 * 10**np.linspace(-0.5, 0.5, 11)
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [0]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)
        elif target == "model_survey0":
            #Tenv_K = 30 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #rCR_au = 200 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #Mstar_Msun = 0.18 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            Tenv_K = [30]
            rCR_au = [200]
            Mstar_Msun = [0.2]
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [0]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "model_survey2":
            #Tenv_K = 30 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #rCR_au = 200 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            #Mstar_Msun = 0.18 * 10**np.array([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
            Tenv_K = [30]
            rCR_au = [200]
            Mstar_Msun = [0.2]
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["CM"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [45]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

            Tenv_K = [30]
            rCR_au = round_sig_array(200 * 10**np.linspace(-0.5, 0.5, 11), 2)
            Mstar_Msun = round_sig_array(0.2 * 10**np.linspace(-0.5, 0.5, 11), 2)
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [80]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "model_survey3":
            Tenv_K = [30]
            rCR_au = round_sig_array(200 * 10**np.linspace(-1, 1, 5), 2)
            Mstar_Msun = round_sig_array(0.2 * 10**np.linspace(-1, 1, 5), 2)
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [80]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "fit_L1527":
            Tenv_K = [20] # 30 * 10**np.linspace(-0.5, 0.5, 11)
            rCR_au = [140, 150, 160]
            Mstar_Msun = [0.20, 0.22, 0.24]  #0.2 * 10**np.linspace(-0.5, 0.5, 11)
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.Tenv_K", "T", Tenv_K],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [0]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)



        elif target == "model_survey_edon": # for edge-on
            # 200 params takes 3 hours. ~> 2000 params takes ~ 1 day.
            rCR_au = round_sig_array(200 * 10**np.linspace(-1, 1, 21), 3)
            Mstar_Msun = round_sig_array(0.2 * 10**np.linspace(-1, 1, 21), 3)
            params_list = make_params_list([["model.ire_model", "", ["Simple","CM"]],
                                            ["model.Tenv_K", "T", [30]],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [45]],
                                            ["sobs.incl", "incl", [85]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "model_survey_incl": # for inclination
            rCR_au = round_sig_array(200 * 10**np.linspace(-1, 1, 11), 3)
            Mstar_Msun = round_sig_array(0.2 * 10**np.linspace(-1, 1, 11), 3)
            params_list = make_params_list([["model.ire_model", "", ["Simple","CM"]],
                                            ["model.Tenv_K", "T", [30]],
                                            ["model.rCR_au", "CR", rCR_au],
                                            ["model.Mstar_Msun", "M", Mstar_Msun],
                                            ["model.cavity_angle", "cav", [0, 45, 80]],
                                            ["sobs.incl", "incl", [30] ],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)


        elif target == "params3":
            params_list = make_params_list([["radmc.temp_mode", "", ["const", "mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [0, 45, 60, 80, 85]],
                                            ["model.simple_density","sden",[0,1]],
                                            ["radmc.mol_name", "", ["c18o","cch"]],
                                           ])
            subparam = make_params_list([["beam_scale", "bsc", [0.3, 1, 3]],
                                         ["fitsa.normalize", "norm", ["None","peak"]],
                                        ])
            parameter_survey("all", params_list, INPUT_FILE, submodes="visualize", subparams=subparam)

        elif target == "params4":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [0, 45, 80]],
                                            ["model.simple_density","sden",[0,1]],
                                            ["radmc.mol_name", "", ["c18o"]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)


        elif target == "params5":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [45, 80]],
                                            ["model.simple_density","sden",[0]],
                                            ["radmc.mol_name", "", ["c18o"]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "inclination_perp":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [45, 80]],
                                            ["fitsa.posang_PV", "pa", [95+90] ],
                                           ])
            subparam = make_params_list([ ["sobs.incl", "i", [80, 85, 90, 95, 100]] ])

            parameter_survey("all", params_list, INPUT_FILE, submodes=["sobs","visualize"], subparams=subparam)


        elif target == "cavangle":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [45, 80]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE)

        elif target == "opacity":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["CM"]],
                                            ["model.cavity_angle", "cav", [0, 45]],
                                            ["radmc.opac", "op",
                                            ["0.1_micron", "100_micron", "amorph_mix", "carbon", "cryst_mix",
                                                 "forsterite", "forsterite_sogawa", "h2oice", "silbeta", "silicate"]
                                            ],
                                           ])
            parameter_survey("setradmc", params_list, INPUT_FILE)


        elif target == "yoffset":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple", "CM"]],
                                            ["model.cavity_angle", "cav", [0, 45 , 80]]
                                            ])
            subparam = make_params_list([ ["fitsa.P_offset_yau", "ofs", [0, 20, 40, 80, 100, 200, 400]],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE, submodes="visualize", subparams=subparam)


        elif target == "pangle":
            params_list = make_params_list([["radmc.temp_mode", "", ["mctherm"]],
                                            ["model.ire_model", "", ["Simple","CM"]],
                                            ["model.cavity_angle", "cav", [0, 45, 80]],])
            subparam = make_params_list([ ["fitsa.posang_PV", "pa", 95 + np.array([0, 5, 10, 15, 30, 60, 90])],
                                           ])
            parameter_survey("all", params_list, INPUT_FILE, submodes="visualize", subparams=subparam)


        elif target == "clean":
            for tar in ["cache","subproc","radmc","fits","pkl","tmpinp"]:
              clean(tar)

        elif "clean_" in target:
            print(target.replace("clean_",""))
            clean(target.replace("clean_",""))

        elif target == "kill_process":
            kill_process()

        else:
            r = subprocess.check_output("cat execute.py | grep -oP '(?<=if target == \").+(?=\":)'", shell=True).decode("utf-8").replace("\n", " ")
            print(r)
            raise Exception("Unknown target name: ", target)

def round_sig(x, sig):
    return round(x,  sig - int(np.floor(np.log10(np.abs(x)))) - 1)

def round_sig_array(x, sig):
    return np.array([  round_sig(xx, sig) for xx in x ])

class Parameter:
    def __init__(self, name, short_name, value):
        self.name = name
        self.short_name = short_name
        self.value = value

def make_params(name, shortname, values):
    return [ Parameter(name, shortname, v) for v in values ]

def make_params_list( param_info_list ):
    return [ make_params(name, shortname, values) for name, shortname, values in param_info_list]

def parameter_survey(modes, params, ori_inpfn, submodes=[], subparams=[[]]):
 #       [ [param_a_1,..], [param_b_1,..], [param_c_1,..] ]
    def fmt(x):
        if isinstance(x, float):
            return "%.3g" % x
        else:
            return x

    def calc_one_paramset(modes, param_set, ori_inpfn):
        modes = modes if isinstance(modes, list) else [modes]
        target_chars = [ par.name.replace(".","\.") + "\s*=\s*" for par in param_set ]
        new_chars = [f"{par.name} = {repr(par.value)}  # original:" for par in param_set ]
        label = "_".join([ f"{par.short_name}{fmt(par.value)}" for par in param_set ])
        tmp_fname = f"{ori_inpfn}.tmp_" + label
        replace_inputfile2(target_chars, new_chars, ori_inpfn, tmp_fname)
        if args.skipifexist and os.path.exists('fig_'+label+'/PV.fits'):
            return
        for mode in modes:
            print(mode)
            Execute(mode, inputfile=tmp_fname)
        mytools.exe('cp -ar fig -T fig_'+label)
        mytools.exe(f'cp -a {tmp_fname} fig_{label}/')

    for param_set in itertools.product(*params):
        calc_one_paramset(modes, param_set, ori_inpfn)
        for sub_param_set in itertools.product(*subparams):
            if len(sub_param_set) != 0:
                calc_one_paramset(submodes, param_set + sub_param_set, ori_inpfn)


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
    if not args.dryrun:
        with open(temp_file, mode='w') as f:
            f.write(filetxt)


def clean(clean_target, rm_cmd=RMCMD, base_dir=BASEDIR, src_dir=SRCDIR, radmc_dir=RADMCDIR):
    def getpaths(target_dir,patterns=[],exclude=[]):
        def match_exclude(x):
            return any([re.fullmatch(exc, x) for exc in exclude ])
        p = Path(target_dir)
        answer=[]
        for pat in patterns:
            answer += [ str(x) for x in p.glob(pat) if not match_exclude(x.name)]
        return answer

    def rm(target_list, option=""):
        print(target_list)
        targets = " ".join(target_list)

        if targets in ["*", "/","/*"]:
            raise Exception("Danger!!", targets)
        if targets == "":
            print("No targets")
            return

        for target in target_list:
            if os.path.isfile(target):
                os.remove(target)
        #mytools.exe(f'{RMCMD} {option} {targets}')

    if clean_target == "cache":
        rm(getpaths(src_dir, ["*.pyc", "__pycache__"]))

    elif clean_target == "subproc":
        rm(getpaths(radmc_dir, ["proc*"] ), option="-r")

    elif clean_target == "radmc":
        rm(getpaths(radmc_dir, ["*.inp"], exclude=[r"molecule_.*", r"dustkappa_.*"]))

    elif clean_target == "fits":
        rm(getpaths(radmc_dir, ["*.fits"]))

    elif clean_target == "pkl":
        rm(getpaths(radmc_dir, ["*.pkl"]))

    elif clean_target == "tmpinp":
        rm(getpaths(base_dir, ["*.in.tmp*"]))
        rm(getpaths(base_dir, ["fig_*"]), option="-r")

    else:
        raise Exception("Unknown clean target name: ", clean_target)


def kiml_process():
    msg("\nMay I kill \"radmc3d\" ? ")
    mytools.exe("ps")
    input("")
    mytools.exe("kill all radmc3d")

    msg("\nMay I kill \"python\" ? ")
    mytools.exe("ps")
    input("")
    mytools.exe("killall python python2 python3 ")

    #echo -e "\nMay I kill \"radmc3d\" ? : OK[Enter]"
    #ps
    #read
    #killall radmc3d
    #echo -e "\nMay I kill \"python\" ? : OK[Enter]"
    #ps
    #read
    #-killall python python2 python3

if __name__=='__main__':
    main()
