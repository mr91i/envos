#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, copy, pickle, argparse, sys
import numpy	  as np
import pandas	  as pd 
import matplotlib as mpl
import matplotlib.cm as cm
from scipy import optimize, interpolate, integrate
from numba import jit

dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + "/../../")
sys.path.append(dn_home)
dn_pkl = dn_home + '/calc/radmc'
print("\nExecute %s:\n"%__file__)
import calc.plotter as mp
from calc import cst 
from calc import CubicSolver


parser = argparse.ArgumentParser(description='This code calculates the kinetic structure based on a model.')
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--disk',action='store_true')
parser.add_argument('--norot',action='store_true')
parser.add_argument('--no_param_tuning',action='store_true')
parser.add_argument('-m','--model',choices=['CM','Simple','TSC'],default='CM')
parser.add_argument('--submodel',choices=[None,'CM','Simple','TSC'],default=None)
#parser.add_argument('--CM',default=True)
#parser.add_argument('--simbal',action='store_true')
#parser.add_argument('--TSC',default=False)
parser.add_argument('--obj',choices=['L1527','None'],default='L1527') 
parser.add_argument('--cr', type=float)
parser.add_argument('--mass',type=float)
parser.add_argument('--cavity_angle',type=float)
parser.add_argument('--pkl_name')
args = parser.parse_args()
if args.obj=='L1527':
   def_cr	= 200
   def_mass = 0.18 #0.18 
   def_cavity_angle = 45
   def_pkl_name = 'L1527'
else:
   def_cr	= 50  
   def_mass = 1.0	
   def_cavity_angle = 0
   def_pkl_name = None
parser.set_defaults(cr=def_cr, 
					mass=def_mass, 
					cavity_angle=def_cavity_angle, 
					pkl_name=def_pkl_name)
args = parser.parse_args()

## Global Parameters ##
use_solution_0th = 0
mp.dbg = args.debug

r_in = 1 *cst.au
r_out = 1e4 *cst.au
nr	= 601 

th_in = 1e-6
th_out = np.pi/2 
nth = 91 #181 #73	

ph_in = -np.pi
ph_out = np.pi
nph = 1 #181 #73 

t_in  = 5e5 *cst.yr
t_out  = 5e5 *cst.yr
nt	= 1
 
#######################
def main():
	r_ax  = np.logspace( np.log10( r_in )  , np.log10( r_out ) , nr	)
	th_ax = np.linspace( th_in	 , th_out , nth	  )
	ph_ax = np.linspace( ph_in	 , ph_out , nph   )
	t_ax = np.linspace( t_in , t_out , nt)
	for t in t_ax:
		Cm = Calc_2d_map( t , r_ax , th_ax	, ph_ax )
		Cm.calc()
		Plots( Cm.res )

def slice_at_midplane( res, tt,*keys_rtp):
	iphi=0
	if len(keys_rtp) >= 2:
		return np.array( [	(res[key_rtp]).take(iphi,2)[ tt.take(iphi,2)==np.pi/2 ]	 for key_rtp in keys_rtp	  ]	 )
	else:
		return	res[keys_rtp].take(iphi,2)[ tt==np.pi/2 ]


def Plots( re , r_lim=500):
		tstamp = args.pkl_name if args.pkl_name else '{:.0e}'.format(re['t']/cst.yr)
		if nph==1:
			r_mg , th_mg , ph_mg =	np.meshgrid( re['r_ax'] , re['th_ax'] , np.linspace(-np.pi,np.pi,31  ) , indexing='ij' )
		else:
			r_mg , th_mg , ph_mg =	np.meshgrid( re['r_ax'] , re['th_ax'] , re['ph_ax'] , indexing='ij' )
		R_mg , z_mg  = r_mg * [ np.sin( th_mg ) ,  np.cos( th_mg ) ]
		x_mg , y_mg =  R_mg * [ np.cos(ph_mg)	, np.sin(ph_mg) ]

		def draw_map( v , name , rang , log=True, V=None, cbl=None,  **kwargs):
			mp.map( R_mg.take(0,2)/cst.au , z_mg.take(0,2)/cst.au , v.take(0,2)	, 'map_{}_{}'.format(name,tstamp), 
						xl='Radius [au]' , yl='Height [au]', cbl=cbl,
						logx=False, logy=False, logcb=log, leg=False, 
						xlim=[0,r_lim], ylim=[0,r_lim], cblim=rang, **kwargs )

		def draw_plane_map( v , name , rang , log=True, V=None, cbl=None, **kwargs):
			if v.shape[2]==1:
				v = np.concatenate([v]*31,axis=2)
			mp.map( x_mg.take(-1,1)/cst.au , y_mg.take(-1,1)/cst.au , v.take(-1,1) , 'plmap_{}_{}'.format(name,tstamp),
						xl='Radius [au]' , yl='Height [au]', cbl=cbl,
						logx=False, logy=False, logcb=log, leg=False,
						xlim=[-1000,1000], ylim=[-1000,1000], cblim=rang,seeds_angle=[0,2*np.pi], **kwargs )		

		# Density and velocity map
		Vec = np.array( [ re[ 'uR' ].take(0,2) , re[ 'uz' ].take(0,2) ] )
		draw_map( re['den'], 'den', [1e-21, 1e-16] , cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec,n_sl=40)

		# Ratio between mu0 and mu : where these gas come from
		draw_map( re['mu0']/np.cos( re['tt'] ), 'mu0_mu',[0, 10] , cbl=r'$\mu_{0}/\mu$', div=10, Vector=Vec,n_sl=40,log=False)

		## Analyze radial profiles at the midplane
		# Slicing 
		V_LS = x_mg/r_mg * re["uph"] - y_mg/r_mg*re["ur"]
		ux = re["ur"] *np.cos(ph_mg) - re["uph"]*np.sin(ph_mg)
		uy = re["ur"] *np.sin(ph_mg) + re["uph"]*np.cos(ph_mg)
		# dV/dy = dV/dr * dr/dy + dV/dth * dth/dy
		#		= dV/dr * 1/sin + dV/dth * 1/r*cos
		#dVdy = V_LS/y_mg #np.gradient( V_LS, re['r_ax'], axis = 0 )/np.sin(ph_mg) + np.gradient( V_LS, re['ph_ax'] , axis = 2)/x_mg
		Vec = np.array( [ ux.take(-1,1) , uy.take(-1,1) ] )
		draw_plane_map( V_LS/1e5, 'Vls', [-2.0, 2.0] , cbl=r'V_LS [km s$^{-1}$]', div=20, n_sl=40, log=False, cmap=cm.get_cmap("seismic"),Vector=Vec)
		draw_plane_map( re['den'], 'den', [1e-18,1e-16] , cbl=r'log Density [g/cm$^{3}$]', div=4, n_sl=40)

		
		den0, uR0 , uph0, den_tot0	= slice_at_midplane( re , th_mg , 'den' , 'uR' , 'uph' , 'den_tot' )  

#		den0, uR0 , uph0  = slice_at_midplane( tt_p , re['den'] , re['uR'] , re['uph'] )
#		den_tot0 = slice_at_midplane( tt_p ,  re['den_tot'] )
	
		mp.set( x = re['r_ax']/cst.au , xlim=[0,500] )
		# Density as a function of distance from the center
		mp.plot( { 'nH2':den0/cst.mn , 'nH2_tot':den_tot0/cst.mn }	,
					'dens_%s'%tstamp , ylim=[1e4,1e15] ,  xlim=[1,1000] ,
					lw=[3,6], c=[None,'k'], ls=['--','-'], 
					loglog=True, vl=[ 2*re['r_CB_0']/cst.au ])


		# Make a "balistic" orbit similar procedure to Oya+2014
#		uph_bal = np.sqrt( re['zeta']*cst.G*re['M']/re['r_ax']	 )
#		uR_bal = np.sqrt(2*cst.G*re['M']/re['r_ax'] - uph_bal**2 ) 
		mp.plot( {	'-uR':-uR0/cst.kms ,
					'uph':uph0/cst.kms }	,
					'v_%s'%tstamp  , ylim=[-1,3] ,	xlim=[0,500] , 
					lw=[ 2,2,4,4], ls=['-','-','--','--'])

		mp.plot( {	'-uR':-uR0/max(uph0) ,
					'uph':uph0/max(uph0) }	,
					'vnorm_%s'%tstamp  , ylim=[0,1.5] ,	x=re['r_ax']/re['r_CB_0'], xlim=[0,3] , 
					lw=[ 2,2,4,4], ls=['-','-','--','--'])

		if args.submodel is not None:
			# see when and how much the results is different
			den0_TSC, uR0_TSC , uph0_TSC = slice_at_midplane( re, th_mg , 'den_TSC' , 'uR_TSC' , 'uph_TSC' )
			mp.plot( { 'log nH2 - 6':np.log10(den0/cst.mn) -6, 
						'-uR':-uR0/cst.kms , 
						'uph':uph0/cst.kms ,	
						'log nH2_TSC - 6':np.log10(den0_TSC/cst.mn) -6,
						'-uR_TSC':-uR0_TSC/cst.kms , 
						'uph_TSC':uph0_TSC/cst.kms }	, 
						'vden_compare_%s'%tstamp	  , x = re['r_ax']/cst.au ,xlim=[0,500],ylim=[-2,10] , 
						lw=[ 3, 3,3,6,6,6], c=['k', ], ls=['-','-','-','--','--','--'], 
						vl=[re['r_CB_0']*2/cst.au, re['r_in_lim']/cst.au])
		return 

#
# Calculator
#
class Calc_2d_map:
	
	def __init__(self , t , r_ax , th_ax , ph_ax ):
		## Non Variable aramters Through Calculation
		self.t	  = t  
		self.r_ax = r_ax
		self.th_ax = th_ax
		self.ph_ax = ph_ax
		self.mu	  = np.round( np.cos( self.th_ax ) , 15)
		self.si	  = np.sqrt( 1 - self.mu**2 )	
		self.M	  = 0
		self.m0 = 0.975
		self.P2			= lambda x: 0.5*(3*x**2-1)
		self.dP2_dtheta = lambda x: -3*x*np.sqrt(1-x**2)
		self.res = {}
		self.Tenv	=	10
		self.cs	= np.sqrt( cst.kB * self.Tenv/cst.mn )
		self.Mdot = self.cs**3*self.m0/cst.G 
		self.M, self.t = self.give_Mass_and_time(self.Mdot, Mstar = args.mass*cst.Msun)	
		self.Omg			   = self.give_Omega( self.cs , self.t , self.M , mode='centrifugal_radius', r_CR=args.cr*cst.au)
		self.j0	 = self.Omg * ( self.m0 * self.cs * self.t * 0.5 )**2
		self.rCB  = self.get_rCB(self.M, self.Omg, self.cs, self.t ) # 50 * cst.au
		self.r_in_lim = self.cs*self.Omg**2* self.t**3
		self.req  = self.rCB * 2 * np.sin( args.cavity_angle/180 * np.pi )**2
		self.Md	= 0.1*self.M
		self.Td	= 30

		self.print_params( self.t, self.M, self.Tenv, self.cs, self.Omg, self.Mdot, self.r_in_lim, self.j0, self.rCB)	

	def get_rCB(self, M, Omg, cs, t , j0=None):
		j0 =  Omg * ( self.m0 * cs * t * 0.5 )**2  if ( j0 is None ) else j0
		return	j0**2 / ( cst.G * M * 2 )

	def print_params(self, t, M, T, cs, Omg, Mdot, r_in_lim, j0, rCB):
		print("Parameters:")
		def print_format( name , val , unit):
			print( name.ljust(10) + 'is    {:10.2g}   '.format(val) + unit.ljust(10)	   )
		print_format( 'T' , T , 'K' )
		print_format( 'cs' , cs/cst.kms , 'km/s' )
		print_format( 't' , t/cst.yr , 'yr' )
		print_format( 'M' , M/cst.Msun , 'Msun' )
		print_format( 'Omega' , Omg , 's^-1' )
		print_format( 'dM/dt' , Mdot/(cst.Msun/cst.yr) , 'M/yr' )
		print_format( 'r_lim' , r_in_lim/cst.au , 'au' )
		print_format( 'j0' , j0/(cst.kms*cst.au)  , 'au*km/s' )
		print_format( 'j0' , j0/(cst.kms*cst.pc)  , 'pc*km/s' )
		print_format( 'rCB' , rCB/cst.au , 'au' )
		print_format( 'c.a.' , args.cavity_angle , '' )
		print("")

##
	def calc_Kinematics(self, r ,model='CM'):
		if model=='CM':
			rho, ur, uth, uph, zeta, mu0  = \
				self.get_Kinematics_CM( r , self.mu , self.M , self.Mdot , self.j0 , cavity_angle=args.cavity_angle) 

		elif model=='TSC':
			rho, ur, uth, uph = \
				self.get_Kinematics_TSC( r , self.mu , Omg , cs , self.t , func=funcs)

		elif model=='Simple':
			rho, ur, uth, uph = self.get_Kinematics_SimpleBalistic( r, self.rCB, self.mu, self.M, self.j0 , Mdot=self.Mdot) 
			zeta = 0 #j0**2 / ( cst.G * self.M * r )
			mu0 = np.zeros_like(self.mu)

		uR = ur * np.sin(self.th_ax) + uth * np.cos(self.th_ax) 
		uz = ur * np.cos(self.th_ax) - uth * np.sin(self.th_ax)
		return rho, ur, uth, uph, zeta, mu0, uR, uz

	def get_mu0(self, mu , zeta , solver='roots'):
#		from scipy.optimize import fsolve
		import sympy
		from scipy import optimize
		if 0 :
			def f(x):
				return zeta*x**3 + (1-zeta)*x - mu
			sol = optimize.root( f , mu , method='broyden2', tol=1e-15 )
			return sol.x
			return optimize.newton( f , mu	,  maxiter=1500, fprime = lambda x :3*zeta*x**2 + 1-zeta,	tol=1.48e-016  )
			
		else:
			def eq(x,m):
				return zeta*x**3 + (1-zeta)*x - m
			def sol1():
				return np.roots([zeta, 0 ,1-zeta ,-m]).real 
			def sol2():
				return CubicSolver.solve(zeta, 0 , 1-zeta ,-m).real
			def sol3():
				return np.array([ optimize.brentq( eq , 0, 1 , args=( m )	, xtol=2e-15) ])
			def sol4():
				return optimize.bisect(eq, 0, 1 , args=( m )  , xtol=2e-15, maxiter=1000)
			
			solver = sol2
			root = [None]*len(mu)
			for i, m in enumerate(mu):
				sol = [ round(a,10) for a in solver() if 0<= round(a,10) <=1]	#if m!=0 else [0]
#				if len(sol)>1:
#					print( zeta, m, "{:.16f}".format(sol[0]))	
				root[i] = sol[0] ## Choose a solution where gas comes vertically
			return np.array(root)

		
	def get_Kinematics_CM( self, r , mu , M , Mdot , j0 , cavity_angle=None ):
		zeta =	j0**2 / ( cst.G * M * r	)
		mu0  = self.get_mu0(mu, zeta) 
		sin0 = np.sqrt( 1 - mu0**2 ) 
		sin = np.where( mu==1 , 1e-100 , np.sqrt( 1 - mu**2 ) )
		v0 = np.sqrt( cst.G * M / r ) 
		mu_to_mu0 = 1 - zeta*(1 - mu0**2) #np.where( np.logical_and(mu0==0,mu==0) , 1-zeta, mu/mu0 )
		ur	= - v0 * np.sqrt( 1 + mu_to_mu0 )
		uth =	v0 * zeta*sin0**2*mu0/sin * np.sqrt( 1 + mu_to_mu0 )
		uph =	v0 * sin0**2/sin		  * np.sqrt( zeta )
		rho = - Mdot / (4 * np.pi * r**2 * ur) / (1 + 2*zeta*self.P2(mu0)	)
		if cavity_angle is not None:
			mask = np.where( mu0 < np.cos( cavity_angle/180.0*np.pi ) , 1 , 0  )
			rho *= mask
			ur *= mask; uth *= mask; uph *= mask
		return rho, ur, uth, uph , zeta, mu0
	
	def get_Kinematics_TSC( self, r , mu , Omg , cs , t  , func=None):
		x =  r/(cs*t)
		tau = Omg*t
		a_0 = (0.5*self.m0/x**3)**(1/2) if func is None else func[0](x)
		V_0 = -(2*self.m0/x)**(1/2)		if func is None else func[1](x)
		a_Q = -13/12*(self.m0/2)**(7/2)*x**(-5/2)
		V_Q = -1/6	*(self.m0/2)**(7/2)*x**(-3/2)
		W_Q = -1/3	*(self.m0/2)**(7/2)*x**(-3/2)
		a_M =  1/12 *(self.m0/2)**(7/2)*x**(-5/2)
		V_M =  1/6	*(self.m0/2)**(7/2)*x**(-3/2)		
		a  = a_0 + tau**2 * ( a_M + a_Q * self.P2(mu) )
		V  = V_0 + tau**2 * ( V_M + V_Q * self.P2(mu) )
		W  =	   tau**2 *   W_Q * self.dP2_dtheta(mu)
		m0x   = x**2 * a_0 * (x-V_0)
		Gamma = tau**2 * (0.5*m0x)**2*(1-mu**2)
		rho = Omg**2 /(4*np.pi*cst.G*tau**2) * a 
		ur	= cs*V
		uth = cs*W
		uph = cs**2*Gamma/(Omg*r*np.sqrt(1-mu**2))
		return rho.clip(0), ur, uth, uph 

	def get_Kinematics_SimpleBalistic(self, r, rCB, mu, M, j , p=-1.5, r0=None, rho0=None , Mdot=None, h=0.1, cavangle=0 , fillv=0):
		vff = np.sqrt( 2 * cst.G * M / r )
		x = r/rCB
#		rho = rho0*(r/r0)**p *np.ones_like(mu)
		b_env = np.logical_and( r*np.sqrt(1-mu**2)>=rCB , mu <= np.cos(cavangle/180.0*np.pi)	)
		rho = np.where( b_env, Mdot/(4*np.pi*r**2 *vff ), fillv )
		ur = - vff * np.sqrt( np.where( b_env, 1 - 1./x , fillv)	)
		uth = np.where( b_env, 0, fillv)
		uph = np.where( b_env, vff/np.sqrt(x), fillv)
		return rho, ur, uth, uph 

	def get_0th_order_function(self,x):
		def f0(x, y):
			ret = np.zeros_like(y)
			V0	= y[0]
			al0 = y[1]
			ret[0] =	 (al0*(x - V0) - 2/x)*(x-V0)/((x-V0)**2 -1)
			ret[1] = al0*(al0 - 2/x*(x - V0))*(x-V0)/((x-V0)**2 -1)
			return ret
#		t = 1e-2
#		x = np.logspace( np.log10(1/t) , np.log10(t**2) , 10000 ) ## determines size of MC 
		x = np.logspace( 3 , -6  , 1000 )
		y0 = [ 0 , 2/x[0]**2 ]
		sol = integrate.solve_ivp( f0 , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True, method='BDF',rtol = 1e-12,	atol = [1e-8,1e-15], vectorized=True)
		V0	= sol.y[0]
		al0 = sol.y[1]
		return interpolate.interp1d(x,al0) , interpolate.interp1d(x,V0)
		

	def put_Disk_sph(self, r , th , M , Md , Rd , Td , CM=False, mode='exponential_cutoff' , ind=-1):
		if not args.disk:
			return np.zeros_like( th )
		R = r*np.sin(th)
		z = r*np.cos(th)
#		if CM:
#			
#			A = 4*a/(m0 * Omg0 * cst.Rsun)
#			u = R/Rd
#			V = 
#			Sigma = (1-u)**0.5 /(2*A*u*t**2*V)
#
			
		if mode=="exponential_cutoff":
			Sigma_0 = Md/(2*np.pi*Rd**2)/(1-2/np.e)
			Sigma = Sigma_0 * (R/cst.au)**ind  * np.exp( -(R/Rd)**(2+ind) )
		elif mode=="tapered_cutoff":
			Sigma_0 = Md/(2*np.pi*Rd**2) * (ind+3)
			Sigma = Md/(2*np.pi) * (ind+3)*Rd**(-ind-3)  * (R/cst.au)**ind
		H  = np.sqrt( cst.kB * Td / cst.mn ) / np.sqrt(cst.G*M/R**3)
		rho = Sigma/np.sqrt(2*np.pi)/H * np.exp( - 0.5*z**2/H**2	)
		return rho


	def calc(self):
		#
		# Set parameters
		#
		funcs = self.get_0th_order_function(self.r_ax) if use_solution_0th else None
		self.res = {'r_ax':self.r_ax, 'th_ax':self.th_ax, 'ph_ax':self.ph_ax, 't':self.t ,'M':self.M, 'r_in_lim':self.r_in_lim, 'r_CB_0': self.rCB}	
		# for Disk

		vals_rt = {} 
		vals_prt = {}
		#	
		# Main loop in r-axis
		#
		for ph in self.ph_ax:
			for r in self.r_ax:
				R_ax, z_ax	= r*self.si , r*self.mu
				rho, ur, uth, uph, zeta, mu0, uR, uz = self.calc_Kinematics( r , model=args.model)
				rho_disk = self.put_Disk_sph( r , self.th_ax , self.M , self.Md , self.rCB , self.Td)	
				disk_reg = rho_disk > np.nan_to_num(rho)
				v_Kep	 = np.sqrt(cst.G * self.M / R_ax )
				vals = {  'R':R_ax, 'z': z_ax, 'den':rho,
							'ur':ur,'uph':uph,'uth':uth,'uR':uR,'uz':uz ,
							'rr':np.full_like(self.th_ax , r) , 
							'tt': self.th_ax ,
							'zeta':zeta, 
							'mu0':mu0 ,
							'den_tot':rho + rho_disk,  
							'vr_tot':np.where( disk_reg    , 0	   , ur  ), 
							'vphi_tot':np.where( disk_reg  , v_Kep , uph ), 
							'vth_tot':np.where( disk_reg   , 0	   , uth )	
						}  
	
				if args.submodel is not None:
					rho_sub, ur_sub, uth_sub, uph_sub, zeta_sub, mu0_sub, uR_sub, uz_sub = self.calc_Kinematics( r , model=args.submodel)
					vals.update({'den_sub':rho_sub, 'ur_sub':ur_sub,'uph_sub':uph_sub,'uth_sub':uth_sub,'uR_sub':uR_sub,'uz_sub':uz_sub }  )
		
				# Accumurate list to res 
				# Stack: stacked_value = stack( vals )
				# [ r1:[ t1 t2 ... tn  ], ... rn:[	]  ]
				self.stack( vals, vals_rt	)
			self.stack( vals_rt , vals_prt	)

		# Convert to ndarray
		for k,v_prt in vals_prt.items():
			a = np.array(v_prt)
			self.res[k] = a.transpose(1,2,0)
	
		# Save to pickle
		self.Save_pickle(self.t) 
		return self.res

	def Save_pickle(self, t):
		stamp = '{:.0e}'.format( t/cst.yr ) if args.pkl_name == 'time' else args.pkl_name
		savefile = '%s/res_%s.pkl'%(dn_pkl, stamp)
		pd.to_pickle(self.res , savefile)
		print('Saved : %s\n'%savefile )	

	def give_Mass_and_time(self, Mdot ,t=None, Mstar=None):
		if args.no_param_tuning :
			return Mdot * t , t
		else: 
			return Mstar , Mstar / Mdot
		
	def give_Omega(self, cs,t,M , mode='const', v_CR=None , r_CR=None):
		if args.norot:
			return 0
		GM = cst.G * M
		r_col = self.m0*0.5*cs*t
		if mode=='const':
			return 1e-14
		if mode=='velocity_peak':
			## v_CR = velocity peak
			##		= sqrt(2) * GM / l
			## l  = (0.5*cs*t)**2 * Omega
			## --> Omega = sqrt(2) * GM / vpeak / (0.5*cs*t)**2 
			return sqrt(2) * GM / v_CR / r_col**2
		if mode=='centrifugal_radius':
			## r_CR = given
			##		= l**2 /(GM)
			##	l  = (0.5*cs*t)**2 * Omega
			return np.sqrt( GM * r_CR )/r_col**2


	def stack(self, dict_vals, dict_stacked ):
		for k, v in dict_vals.items():
			if not k in dict_stacked:
				dict_stacked[k] = []
			if not isinstance(v,(list,np.ndarray)):
				v=[v]
			dict_stacked[k].append( v )
		dict_vals.clear()

##########################################################################################################################

###########
if __name__ == '__main__':
	main()
###########
