#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,	absolute_import, division
import pickle, os, sys,argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d, interp1d
dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + '/../../')
sys.path.append(dn_home)
dn_radmc = dn_home + '/calc/radmc/'
print('Execute %s:\n'%__file__)
from calc import cst

parser = argparse.ArgumentParser(description='This code sets input files for calculating radmc3d.')
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--tgas',default=False)
parser.add_argument('--turb',default=False)
parser.add_argument('--line',default=True)
parser.add_argument('--lowreso',default=False,type=int)
parser.add_argument('--fdg',default=0.01,type=float)
parser.add_argument('--mol_abun',default=2e-7,type=float)
args = parser.parse_args()

def convert_cyl_to_sph( v_cyl , R_ori , z_ori , RR , zz):
	if 1:
		f = interp2d( np.log10( R_ori[1:] ) , np.log10( z_ori[1:] ) , np.log10( v_cyl[1:,1:] )	)
		fv = np.vectorize(f)
		v_sph = 10**fv(  np.log10( RR ) ,  np.log10( zz ) )
	else:
		f = interp2d( R_ori , z_ori , v_cyl )
		fv = np.vectorize(f)
		v_sph = fv( RR , zz)
	return v_sph

def interpolator2( value, x_ori , y_ori , x_new, y_new ,logx=False, logy=False, logz=False, pm=False):

	if logx:
		x_ori = np.log10(x_ori)
		x_new = np.log10(x_new)

	if logy:
		y_ori = np.log10(y_ori)
		y_new = np.log10(y_new)

	if logz:
		sign  = np.sign(value)
		value = np.log10(np.abs(value))		

	f = interp2d( x_ori , y_ori , value.T , fill_value = 0 )
	fv = np.vectorize(f)
	ret =  fv( x_new  , y_new ) 
	if logz:
		if pm:
			f2 = interp2d( x_ori , y_ori , sign.T , fill_value = np.nan )
			fv2 = np.vectorize(f2)
			sgn =  fv2( x_new  , y_new )
			return np.where( sgn!=0, np.sign(sgn)*10**ret, 0 ) 
		return 10**ret #np.where( sgn!=0, np.sign(sgn)*10**ret, 0 ) 
	else:
		return ret

def main():
	D = pd.read_pickle(dn_radmc+'res_L1527.pkl')

	nphot	 = 100000
	
	#
	# Grid parameters
	#
	nr		 = 128 if args.lowreso else 512
	ntheta	 = 64 if args.lowreso else 256
	nphi	 = 1
	rin		 = 10*cst.au
	rout	 = 1000*cst.au
	thetaup  = 0.0 / 180.0 * np.pi 
	#
	# Disk parameters
	#
	#
	# Make the coordinates
	#
	ri	 = np.logspace(np.log10(rin),np.log10(rout),nr+1)
#	thetai	 = np.linspace(thetaup,0.5e0*np.pi,ntheta+1)
	thetai1	 = np.linspace(thetaup,0.25*np.pi,ntheta/4.+1)
	thetai2  = np.linspace(0.25*np.pi,0.5*np.pi,ntheta*3/4.+1)[1:]
	thetai	 = np.concatenate([thetai1,thetai2])
	phii	 = np.linspace(0,2*np.pi,nphi+1)
	rc		 = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
	thetac	 = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
	phic	 = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
	#
	# Make the grid
	#
	rr,tt	 = np.meshgrid(rc,thetac,phic,indexing='ij')
	RR		 = rr*np.sin( tt )
	zz		 = rr*np.cos( tt )
	
	rhog = interpolator2( D.den_tot[:,:,0] , D.r_ax , D.th_ax	, rr, tt , logx=True, logz=True)
	rhod = rhog * args.fdg
	vr	= interpolator2( D.ur[:,:,0] , D.r_ax , D.th_ax  , rr, tt , logx=True, logz=True) 
	vth	= interpolator2( D.uth[:,:,0] , D.r_ax , D.th_ax	, rr, tt , logx=True, logz=True, pm=True) 
	vph  = interpolator2( D.uph[:,:,0] , D.r_ax , D.th_ax	, rr, tt , logx=True, logz=True)
	vturb	= np.zeros((nr,ntheta,nphi)) 

	rhog = np.nan_to_num(rhog)
	vr = np.nan_to_num(vr)
	vth = np.nan_to_num(vth)
	vph = np.nan_to_num(vph)

	if args.tgas:
		tgas = 30*np.ones_like(rhog)
	
	#
	# 1. Write the wavelength_micron.inp file
	#
	#
	# Star parameters
	#
	mstar	 = D.Ms #cst.ms
	rstar	 = D.Rs
	tstar	 = D.Ts   #1.2877 * cst.Tsun ## gives 2.75 Ls
	pstar	 = np.array([0.,0.,0.])
	#
	lam1	 = 0.1
	lam2	 = 7
	lam3	 = 25
	lam4	 = 1e4
	n12		 = 20
	n23		 = 100
	n34		 = 30
	lam12	 = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
	lam23	 = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
	lam34	 = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
	lam		 = np.concatenate([lam12,lam23,lam34])
	nlam	 = lam.size
	#
	# Write the wavelength file
	#
	with open(dn_radmc+'wavelength_micron.inp','w+') as f:
		f.write('%d\n'%(nlam))
		for value in lam:
			f.write('%13.6e\n'%(value))

	#
	# 2. Write the stars.inp file
	#
	with open(dn_radmc+'stars.inp','w+') as f:
		f.write('2\n')
		f.write('1 %d\n\n'%(nlam))
		f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
		for value in lam:
			f.write('%13.6e\n'%(value))
		f.write('\n%13.6e\n'%(-tstar))
	

	#
	# 3. Write the grid file
	#
	with open(dn_radmc+'amr_grid.inp','w+') as f:
		f.write('1\n')						 # iformat
		f.write('0\n')						 # AMR grid style  (0=regular grid, no AMR)
		f.write('100\n')					 # Coordinate system: spherical
		f.write('0\n')						 # gridinfo
		f.write('1 1 0\n')					 # Include r,theta coordinates
		f.write('%d %d %d\n'%(nr,ntheta,nphi))	# Size of grid
		for value in ri:
			f.write('%13.6e\n'%(value))		 # X coordinates (cell walls)
		for value in thetai:
			f.write('%13.6e\n'%(value))		 # Y coordinates (cell walls)
		for value in phii:
			f.write('%13.6e\n'%(value))		 # Z coordinates (cell walls)
	

	if args.line:
		mol_name = 'cch' #'c18o'	
		mol_abun = args.mol_abun ## c18o
		#
		# 4.1 Write the molecule number density file. 
		#	
		n_mol  = rhog * mol_abun/(2.34*cst.amu)
		with open(dn_radmc+'numberdens_%s.inp'%(mol_name),'w+') as f:
			f.write('1\n')						 # Format number
			f.write('%d\n'%(nr*ntheta*nphi))			 # Nr of cells
			data = n_mol.ravel(order='F')			 # Create a 1-D view, fortran-style indexing
			data.tofile(f, sep='\n', format='%13.6e')
			f.write('\n')
	
		#
		# 4.2 Write the lines.inp control file
		#
		with open(dn_radmc+'lines.inp','w') as f:
			f.write('2\n')# 
			f.write('1\n')# number of molecules
			f.write('%s	 leiden	0 0 0\n'%(mol_name)) # molname1 inpstyle1 iduma1 idumb1 ncol1

		#
		# 4.3 Write the gas velocity field
		#
		with open(dn_radmc+'gas_velocity.inp','w+') as f:
			f.write('1\n')						 # Format number
			f.write('%d\n'%(nr*ntheta*nphi))			 # Nr of cells
			for ip in range(nphi):
				for it in range(ntheta):
					for ir in range(nr):
						f.write('%13.6e %13.6e %13.6e\n'%(vr[ir,it,ip],vth[ir,it,ip],vph[ir,it,ip]))
	
		#
		# 4.4 Write the microturbulence file
		#
		if args.turb:
			with open(dn_radmc+'microturbulence.inp','w+') as f:
				f.write('1\n')						 # Format number
				f.write('%d\n'%(nr*ntheta*nphi))			 # Nr of cells
				data = vturb.ravel(order='F')		   # Create a 1-D view, fortran-style indexing
				data.tofile(f, sep='\n', format='%13.6e')
				f.write('\n')
	
		#
		# 4.5 Write the gas temperature
		#
		if args.tgas:
			with open(dn_radmc+'gas_temperature.inp','w+') as f:
				f.write('1\n')						 # Format number
				f.write('%d\n'%(nr*ntheta*nphi))			 # Nr of cells
				data = tgas.ravel(order='F')		  # Create a 1-D view, fortran-style indexing
				data.tofile(f, sep='\n', format='%13.6e')
				f.write('\n')

		#	with open(dn_radmc+'dust_temperature.dat','w+') as f:
		#		f.write('1\n')						 # Format number
		#		f.write('%d\n'%(nr*ntheta*nphi))			 # Nr of cells
		#		data = tgas.ravel(order='F')		  # Create a 1-D view, fortran-style indexing
		#		data.tofile(f, sep='\n', format='%13.6e')
		#		f.write('\n')

	#
	# Write the dust temperature file
	# NOTE: You can also remove this, and compute the dust
	#		temperature self-consistently with the shell 
	#		command 'radmc3d mctherm'. Here, however, we 
	#		simply make a guess and write it to file, so
	#		that you can immediately make the images.
	#
	#	with open('dust_temperature.dat','w+') as f:
	#		f.write('1\n')						 # Format number


	#
	# 5. Write the density file
	#
	with open(dn_radmc+'dust_density.inp','w+') as f:
		f.write('1\n')						 # Format number
		f.write('%d\n'%(nr*ntheta*nphi))	 # Nr of cells
		f.write('1\n')						 # Nr of dust species
		data = rhod.ravel(order='F')		 # Create a 1-D view, fortran-style indexing
		data.tofile(f, sep='\n', format='%13.6e')
		f.write('\n')

	#
	# 6. Dust opacity control file
	#
	with open(dn_radmc+'dustopac.inp','w+') as f:
		opacs = ['silicate']
		f.write('2				 Format number of this file\n')
		f.write('{}				 Nr of dust species\n'.format( len(opacs) ))
		f.write('============================================================================\n')
		for op in opacs:
			f.write('1				 Way in which this dust species is read\n')
			f.write('0				 0=Thermal grain\n')
			f.write('{}		 Extension of name of dustkappa_***.inp file\n'.format( op ))
			f.write('----------------------------------------------------------------------------\n')

	#
	# 7. Write the radmc3d.inp control file
	#
	with open(dn_radmc+'radmc3d.inp','w+') as f:
		f.write('nphot = %d\n'%(nphot))
		f.write('scattering_mode_max = 0\n') # 1: with scattering 
		f.write('iranfreqmode = 1\n')
		f.write('mc_scat_maxtauabs = 5.d0\n')
		f.write('tgas_eq_tdust = %d'%(1 if args.tgas else 0))
	del f
		
	pkl = {'rr':rr,'tt':tt,'rhod':rhod}
	savefile = dn_radmc+os.path.basename(__file__)+'.pkl'
	pd.to_pickle(pkl, savefile ,protocol=2)
	print('Saved : %s\n'%savefile )

if __name__=='__main__':
	main()
