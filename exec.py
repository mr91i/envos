#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess, sys, argparse

parser = argparse.ArgumentParser(description='This code executes a pipeline making from a model to figures')
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--dryrun',action='store_true')
parser.add_argument('--genpyc',action='store_true')
parser.add_argument('-1','--one',action='store_true')
args = parser.parse_args()

sys.dont_write_bytecode = True if args.genpyc else False
count=0
def main():
	if args.one:
		pexe(ca=85)
	else:
		for ca in [0,30,45,60,75,80,85]:
			pexe(ca=ca)
	
		for cr in [50,100,300,400]:
			pexe(cr=cr)
	
		for mass in [0.1,0.15,0.3]:
			pexe(mass=mass)

		for abun in [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-20]:
			pexe(mol_abun=abun)

	print("\nTotal number of runs is %i."%count)

###################################################

def exe(cmd):
	try:
		if args.dryrun:
			print(cmd)
			return 0
		subprocess.check_call('echo `date` \"%s\" >> .executed_cmd.txt'%cmd, shell=True )
		retcode = subprocess.check_call( cmd, shell=True )
		return 0
		#retcode = subprocess.check_call( cmd.split() )
	except subprocess.CalledProcessError as e:
		print("Error generated:")
		print(" - return code is %s"%e.returncode)
		print(" - cmd is \"%s\""%e.cmd)
		print(" - output is %s"%e.output)
		#exit()
		raise Exception()
		
def pexe( cr=None, ca=None, mass=None, mol_abun=None):
	global count
	count += 1
#	name,  =  params(cr,ca,mass)
	def agen(d):
		l=[]
		for k, v in d.items():
			if v is not None:
				l.append("--%s %g"%(k,v))
		return " ".join(l)

	def ngen(d):
		l=[]
		for k, v in d.items():
			if v is not None:
				l.append("%s%g"%(k,v))
		return "_".join(l)
		
#	d = {"cr":cr,"cavity_angle":ca,"mass":mass}i#	prinit("".join( [ "--%s %f"%(k,v) if not v is  for k, v in d.items() ])	 )
	try:
		## Make a model : cr, ca, mass, ...
		exe('python calc/mkmodel/pmodes.py '+agen({"cr":cr,"cavity_angle":ca,"mass":mass}))	
		## Make a setting for radmc3d : opacity, d/g, molecule, ...
		exe('python calc/radmc/set.py '+agen({"mol_abun":mol_abun} ))
		## Execute radmc3d : number of threads
		exe('cd calc/radmc; radmc3d mctherm setthreads 16')
		## Make a 3D-data cube (x-y-freq data): inclination, slicing angle, ... 
		exe('yes | python calc/radmc/mkfits.py')
		## Make figures of data in radmc3d : None
		exe('cd calc/radmc ;  python plot.py')
		## Make a PV diagram
		exe('python calc/sobs/sobs.py')	
		exe('cp -r fig fig_'+ngen({"cr":cr,"ca":ca,"m":mass,"ma":mol_abun}) )
		## This figure filenames may contain period ".", so please care about it.
	except Exception:
		print("Error...")
		return 1
		

if __name__=='__main__':
	main()
