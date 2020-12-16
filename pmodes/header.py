# Note:
#  to supress cache files :
#      write "PYTHONDONTWRITEBYTECODE = 1" to ~/.bashrc or ~/.bash_profile


## Set directory paths
import os
import sys
#dpath_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
#dpath_home = os.path.abspath(dpath_here + '/../')
#dpath_radmc = dpath_home + "/radmc"
#dpath_fig = dpath_home + "/fig"

#logger.setLevel(logging.DEBUG)
#logging.root.setLevel(logging.DEBUG)


#1## argparse
#import argparse
#parser = argparse.ArgumentParser(description='')
##parser.add_argument('input_file_name', nargs='?', default="inp")
#parser.add_argument('input_file_path', nargs='?', default="./inp.py")
#parser.add_argument('-e','--edit', action='store_true')
#args = parser.parse_args()
#logger.info("Input file path is "+args.input_file_path)
#if args.edit:
#    print("Enter edit mode with vim...")
#    os.system(f"vim {args.input_file_path}")
#

## Dynamically reading inputfile
# ver. >= 3.5
#import types
#import importlib
import importlib.util
spec = importlib.util.spec_from_file_location("InputParams", args.input_file_path)
inpfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inpfile)
inp = inpfile.gen_input()

#print(inp)
#inp = importlib.import_module(args.input_file_path)
#loader = importlib.machinery.SourceFileLoader('inp',  dpath_home+"/"+args.input_file_name)
#inp = types.ModuleType(loader.name)
#loader.exec_module(inp)
