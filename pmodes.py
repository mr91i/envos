#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, copy, pickle, argparse
import numpy	  as np
import pandas	  as pd 
import matplotlib as mpl
from scipy import optimize, interpolate, integrate
import plotter as mp
import cst
parser = argparse.ArgumentParser(description='This code calculates the kinetic structure based on a model.')
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--disk',action='store_true')
parser.add_argument('--norot',action='store_true')
parser.add_argument('--no_param_tuning',action='store_true')
parser.add_argument('--CM',default=True)
parser.add_argument('--TSC',default=True)
parser.add_argument('--obj',choices=['L1527','None'],default='L1527') 
print( parser.parse_known_args() )
if parser.parse_known_args()[0].obj == 'L1527':
   def_cr	= 200 
   def_mass = 0.18 #0.18 
   def_cavity_angle = 45
   def_pkl_name = 'L1527'
else:
   def_cr	= 50  
   def_mass = 1.0	
   def_cavity_angle = 0
   def_pkl_name = None
parser.add_argument('--cr',  default=def_cr)
parser.add_argument('--mass',default=def_mass)
parser.add_argument('--cavity_angle',default=def_cavity_angle,type=float)
parser.add_argument('--pkl_name',default=def_pkl_name)
args = parser.parse_args()


## Global Parameters ##
use_solution_0th = 0
mp.dbg = args.debug

r_in = 1 *cst.au
r_out = 10000 *cst.au
nr	= 401

th_in = 0		 + 1e-6
th_out = np.pi/2 
nth = 181 #73  

t_in  = 5e5 *cst.yr
t_out  = 5e5 *cst.yr
nt	= 1
 
#######################

def main():
	r_ax  = np.logspace( np.log10( r_in )  , np.log10( r_out ) , nr	)
	th_ax = np.linspace( th_in	 , th_out , nth	  )
	t_ax = np.linspace( t_in , t_out , nt)
	for t in t_ax:
		Cm = Calc_2d_map( t , r_ax , th_ax	)
		Cm.calc()
		Plots( Cm.res )

def slice_at_midplane( tt,	*a_rths  ):
	if len(a_rths) >= 2: 
		return np.array( [	a_rth[ tt==np.pi/2 ]	 for a_rth in a_rths	  ]	 )
	else:
		return	a_rths[0][ tt==np.pi/2 ] 


def Plots( re , r_lim=500):
		tstamp = args.pkl_name if args.pkl_name else '{:.0e}'.format(re['t']/cst.yr)
		rr_p , tt_p =  np.meshgrid( re['r_ax'] , re['th_ax'] , indexing='ij' )
		RR_p , zz_p  = rr_p * [ np.sin( tt_p ) ,  np.cos( tt_p ) ]

		def draw_map( v , name , rang , log=True, V=None, cbl=None, **kwargs):
			mp.map( RR_p/cst.au , zz_p/cst.au , v	, 'map_{}_{}'.format(name,tstamp), 
						xl='Radius [au]' , yl='Height [au]', cbl=cbl,
						logx=False, logy=False, logcb=log, leg=False, 
						xlim=[0,r_lim], ylim=[0,r_lim], cblim=rang, **kwargs )

		# Density and velocity map
		Vec = np.array( [ re[ 'uR' ] , re[ 'uz' ] ] )
		draw_map( re['den'], 'den', [1e-21, 1e-16] , cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec,n_sl=40)

		# Ratio between mu0 and mu : where these gas comes from
		draw_map( re['mu0']/np.cos( re['tt'] ), 'mu0_mu',[0, 10] , cbl=r'$\mu_{0}/\mu$', div=10, Vector=Vec,n_sl=40,log=False)

		## Analyze radial profiles at the midplane
		# Slicing 

		den0, uR0 , uph0  = slice_at_midplane( tt_p , re['den'] , re['uR'] , re['uph'] )
		den_tot0 = slice_at_midplane( tt_p ,  re['den_tot'] )
	
		mp.set( x = re['r_ax']/cst.au , xlim=[0,500] )
		# Density as a function of distance from the center
		mp.plot( { 'nH2':den0/cst.mn , 'nH2_tot':den_tot0/cst.mn }	,
					'dens_%s'%tstamp , ylim=[1e4,1e15] ,  xlim=[1,1000] ,
					lw=[3,6], c=[None,'k'], ls=['--','-'], 
					loglog=True, vl=[ 2*re['r_CB_0']/cst.au ])

		# Make a "balistic" orbit similar procedure to Oya+2014
		uph_bal = np.sqrt( re['zeta']*cst.G*re['M']/re['r_ax']	 )
		uR_bal = np.sqrt(2*cst.G*re['M']/re['r_ax'] - uph_bal**2 ) 
		mp.plot( {	'-uR':-uR0/cst.kms ,
					'uph':uph0/cst.kms ,
					'-uR_bal':uR_bal/cst.kms ,
					'uph_bal':uph_bal/cst.kms ,
					'u_max':np.sqrt(uR0**2+ uph0**2)/cst.kms }	,
					'v_%s'%tstamp  , ylim=[-2,7] ,	xlim=[0,500] , 
					lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'])

		if args.TSC:
			# see when and how much the results is different
			den0_TSC, uR0_TSC , uph0_TSC = slice_at_midplane( tt_p , re['den_TSC'] , re['uR_TSC'] , re['uph_TSC']  )
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

	def __init__(self , t , r_ax , th_ax ):
		self.t	  = t  
		self.r_ax = r_ax
		self.th_ax = th_ax
		self.mu	  = np.cos( self.th_ax )
		self.si	  = np.sqrt( 1 - self.mu**2 )	
		self.M	  = 0
		self.m0 = 0.975
		self.P2			= lambda x: 0.5*(3*x**2-1)
		self.dP2_dtheta = lambda x: -3*x*np.sqrt(1-x**2)
		self.res = {}

	def get_mu0(self, mu_ax , zeta , solver='roots'):
#		solver='brentq'
		
		if solver=='roots':	
			root = [None]*len(mu_ax)
			for i, mu in enumerate(mu_ax):
				sol = np.roots( [ zeta, 0 , 1-zeta , -mu ] )	
			#	mp.say(sol)
				sol = sol[ 0<=sol ]
				sol = sol[ sol<=1+1e-10 ]

				if len(sol)==1 and np.isreal(sol[0]) :
					root[i] = np.real( sol[0] )
				else:
					raise Exception(sol)
			mp.say(root)
			return np.array(root)

		if solver=='brentq':
			def eq(mu0,mu):
				if mu == 1:
			#		return zeta*mu0*(1 + mu0) + 1
					return mu0 - 0.5*( -1 + np.sqrt( 1 - 4/zeta ))
				elif mu == 0:
			#		return	zeta - 1/(1 - mu0**2)
					return 0#mu0 - np.sqrt(1 - 1/zeta)
				else:
					return zeta - ( 1 - mu/mu0)/(1 - mu0**2)  
			return np.array( [ optimize.brentq(eq, 0, 1 , args=( mu ) )  for mu in mu_ax ] ) 

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
		

	def if_one(self, cond , val ):
		return np.where( cond  , np.ones_like(val) , val )

	def if_zero(self, cond , val ):
		return np.where( cond  , np.zeros_like(val) , val )
	
	def get_Kinematics_CM( self, r , mu , M , Mdot , j0 , cavity_angle=None ):
		SMALL0	= 1e-50
		SMALL1	= 1e-49
		zeta =	j0**2 / ( cst.G * M * r	)
		mu0  = self.get_mu0( mu ,zeta) 
		sin, sin0 = np.sqrt( 1 - mu**2 ) ,	np.sqrt( 1 - mu0**2 ) 
		vff	= np.sqrt( cst.G * M / r )
		
		sin = np.where( sin==0 , SMALL0 , sin )
		if cavity_angle is not None:
			Mdot *= ( mu0 < np.cos( cavity_angle/180*np.pi ) ) * np.ones_like( mu0 )
	
		mu_to_mu0 = mu/mu0
		ur	= - vff * np.sqrt( 1 + mu_to_mu0 )
		uth =	vff * self.if_zero( sin<SMALL1 , ( mu0 - mu )/sin ) * np.sqrt( 1 + mu_to_mu0 ) 
		uph =	vff * self.if_one(	sin<SMALL1 , sin0/sin		 ) * np.sqrt( 1 - mu_to_mu0 )
		rho = - Mdot / (4 * np.pi * r**2 * ur) / (1 + 2*zeta*self.P2(mu0))
		return rho, ur, uth, uph , zeta, mu0
	
	def get_rCB(self, M, Omg, cs, t , j0=None):
		j0 =  Omg * ( self.m0 * cs * t * 0.5 )**2  if ( j0 is None ) else j0
		return	j0**2 / ( cst.G * M * 2 )

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

	def put_Disk_sph(self, r , th , M , Md , Rd , Td , CM=False, mode='exponential_cutoff' , ind=-1):
		if not args.disk:
			return np.zeros_like( th )
		R = r*np.sin(th)
		z = r*np.cos(th)
		#if CM:
		#	pass
		if mode=="exponential_cutoff":
			Sigma_0 = Md/(2*np.pi*Rd**2)/(1-2/np.e)
			Sigma = Sigma_0 * (R/cst.au)**ind  * np.exp( -(R/Rd)**(2+ind) )
		elif mode=="tapered_cutoff":
			Sigma_0 = Md/(2*np.pi*Rd**2) * (ind+3)
			Sigma = Md/(2*np.pi) * (ind+3)*Rd**(-ind-3)  * (R/cst.au)**ind
		H  = np.sqrt( cst.kB * Td / cst.mn ) / np.sqrt(cst.G*M/R**3)
		rho = Sigma/np.sqrt(2*np.pi)/H * np.exp( - 0.5*z**2/H**2	)
		return rho

	def urth_to_uRz( self, ur , ut , th_ax ):
		uR = ur * np.sin(th_ax) + ut * np.cos(th_ax) 
		uz = ur * np.cos(th_ax) - ut * np.sin(th_ax)
		return uR, uz

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

	def print_params(self, t, M, T, cs, Omg, Mdot, r_in_lim, j0, rCB):
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

	def calc(self):
		#
		# Set parameters
		#
		T	=	10
		cs	= np.sqrt( cst.kB * T/cst.mn )
		Mdot = cs**3*self.m0/cst.G 
		Mdot *=	1 - args.cavity_angle/90
		self.M, self.t = self.give_Mass_and_time(Mdot, Mstar=args.mass*cst.Msun)	
		Omg			   = self.give_Omega( cs , self.t , self.M , mode='centrifugal_radius', r_CR=args.cr*cst.au)
		r_in_lim = cs*Omg**2* self.t**3
		j0	 = Omg * ( self.m0 * cs * self.t * 0.5 )**2
		rCB  = self.get_rCB(self.M, Omg, cs, self.t ) # 50 * cst.au
		req  = rCB * 2 * np.sin( 45/180 * np.pi )**2
		self.print_params( self.t, self.M, T, cs, Omg, Mdot, r_in_lim, j0, rCB)	
		funcs = self.get_0th_order_function(self.r_ax) if use_solution_0th else None
		self.res = {'r_ax':self.r_ax, 'th_ax':self.th_ax, 't':self.t ,'M':self.M, 'r_in_lim':r_in_lim, 'r_CB_0': rCB}	
		# for Disk
		Md	= 0.1*self.M
		Td	= 30
		#	
		# Main loop in r-axis
		#
		for r in self.r_ax:
			R_ax  = r * self.si
			z_ax  = r * self.mu
			if args.CM:
				rho, ur, uth, uph, zeta, mu0  = \
					self.get_Kinematics_CM( r , self.mu , self.M , Mdot ,j0 , cavity_angle=args.cavity_angle) 
				uR, uz = self.urth_to_uRz( ur , uth , self.th_ax )

			rho_disk = self.put_Disk_sph( r , self.th_ax , self.M , Md , rCB , Td)	
			disk_reg = rho_disk > rho
			v_Kep	 = np.sqrt(cst.G * self.M / R_ax )
			vphi_tot = np.where( disk_reg , v_Kep  , uph  )

			vals = {  'R':R_ax, 'z': z_ax, 'den':rho,
						'ur':ur,'uph':uph,'uth':uth,'uR':uR,'uz':uz ,
						'rr':np.full_like(self.th_ax , r) , 'tt': self.th_ax ,
						'zeta':zeta, 'mu0':mu0 ,
						'den_tot':rho + rho_disk,  'vr_tot':np.where( disk_reg , 0	, ur  ) , 
						'vphi_tot':np.where(  disk_reg , v_Kep	, uph  ), 'vth_tot':np.where( disk_reg , 0	, uth  )	}  

			if args.TSC:
				rho_TSC, ur_TSC, uth_TSC, uph_TSC = \
					self.get_Kinematics_TSC( r , self.mu , Omg , cs , self.t , func=funcs)
				uR_TSC, uz_TSC = self.urth_to_uRz( ur_TSC , uth_TSC , self.th_ax )	
				vals.update( {	'den_TSC':rho_TSC, 'ur_TSC':ur_TSC,'uph_TSC':uph_TSC,'uth_TSC':uth_TSC,'uR_TSC':uR_TSC,'uz_TSC':uz_TSC }  )
	
			# Accumurate list to res 
			for k, v in vals.items():
				if not k in self.res:
					self.res[k] = []
				self.res[k].append( v )
		
		# Convert to ndarray
		for k,v in self.res.items():
			self.res[k] = np.array(v)
		
		# Save to pickle
		stamp = '{:.0e}'.format( self.t/cst.yr ) if args.pkl_name == 'time' else args.pkl_name
		pd.to_pickle( self.res , 'res_%s.pkl'%stamp)	
		print('Saved: %s'%'res_%s.pkl'%stamp )	
		return self.res
	
##########################################################################################################################

###########
if __name__ == '__main__':
	main()
###########


