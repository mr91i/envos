#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
#import natconst as cst
import matplotlib as mpl
import os, copy, pickle
from scipy import integrate, special, optimize
from scipy.interpolate import interp1d,interp2d,griddata
from scipy.integrate import solve_ivp
from sympy import *
import plotter as mp
import cst

## Global Parameters ##
CM = 1
TSC = 0
use_solution_0th = 0
mp.dbg = 0

r_in = 1 *cst.au
r_out = 10000 *cst.au
nr	= 301

th_in = 0
th_out = np.pi/2
nth = 73  

t_in  = 5e5 *cst.yr
t_out  = 5e5 *cst.yr
nt	= 1
 
#######################

class Param:
	def __init__(self):
		self.Disk = 0
		self.Rotation = 1
		self.AutoMassTuning = 1
		self.pickle_name = "L1527"

par = Param()

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


def Plots( re ):
		tstamp = '{:.0e}'.format(re['t']/cst.yr)

		rr_p , tt_p =  np.meshgrid( re['r_ax'] , re['th_ax'] , indexing='ij' )
		RR_p = rr_p * np.sin( tt_p )  
		zz_p = rr_p * np.cos( tt_p )  

		r_lim = 500 

		def draw_map( v , name , rang , log=True, V=None, cbl=None, **kwargs):
			mp.map( RR_p/cst.au , zz_p/cst.au , v	, "map_{}_{}".format(name,tstamp), 
						xl='Radius [au]' , yl='Height [au]', cbl=cbl,
						logx=False, logy=False, logcb=log, leg=False, 
						xlim=[0,r_lim], ylim=[0,r_lim], cblim=rang, **kwargs )

		def draw_map_name( vname ,*args, **kwargs):
			draw_map( re[vname] , vname , *args, **kwargs)


		# Density and velocity map
		Vec = np.array( [ re[ 'uR' ] , re[ 'uz' ] ] )
		draw_map( re['den'], 'den', [1e-21, 1e-16] , cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec,n_sl=40)

		# Ratio between mu0 and mu : where these gas comes from
		draw_map( re['mu0']/np.cos( re['tt'] ), 'mu0_mu',[0, 10] , cbl=r'$\mu_{0}/\mu$', div=10, Vector=Vec,n_sl=40,log=False)


		den0, uR0 , uph0  = slice_at_midplane( tt_p , re['den'] , re['uR'] , re['uph'] )

		if TSC:
			den0_TSC, uR0_TSC , uph0_TSC = slice_at_midplane( tt_p , re['den_TSC'] , re['uR_TSC'] , re['uph_TSC']  )
			mp.plot( { 'log nH2 - 6':np.log10(den0/cst.mn) -6, 
						'-uR':-uR0/cst.kms , 
						'uph':uph0/cst.kms ,	
						'log nH2_TSC - 6':np.log10(den0_TSC/cst.mn) -6,
						'-uR_TSC':-uR0_TSC/cst.kms , 
						'uph_TSC':uph0_TSC/cst.kms }	, 
						'vden_compare_%s'%tstamp	  , x = re['r_ax']/cst.au ,xlim=[0,500],ylim=[-2,10] , 
						lw=[ 3, 3,3,6,6,6], c=['k', ], ls=['-','-','-','--','--','--'], 
						vl=[re["r_CB_0"]*2/cst.au, re["r_in_lim"]/cst.au])

		uph_bal = np.sqrt( re['zeta']*cst.G*re['M']/re['r_ax']	 )
		uR_bal = np.sqrt(2*cst.G*re['M']/re['r_ax'] - uph_bal**2 ) 


		den_tot0 = slice_at_midplane( tt_p ,  re['den_tot'] )
		mp.plot( { 'nH2':den0/cst.mn ,
					'nH2_tot':den_tot0/cst.mn }	,
					'v_dens_%s'%tstamp	, x = re['r_ax']/cst.au , xlim=[1,1000],ylim=[1e4,1e15] ,
					lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'], loglog=True, vl=[re["r_CB_0"]/cst.au*2 ])
		exit()
		mp.plot( { 'log nH2 - 6':np.log10(den0/cst.mn) -6, 
					'-uR':-uR0/cst.kms , 
					'uph':uph0/cst.kms , 
					'-uR_bal':uR_bal/cst.kms , 
					'uph_bal':uph_bal/cst.kms , 
					'u_max':np.sqrt(uR0**2+ uph0**2)/cst.kms }	, 
					'v_dens_%s'%tstamp	, x = re['r_ax']/cst.au , xlim=[0,500],ylim=[-2,7] , 
					lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'])

		mp.plot( { 'nH2/1e6':den0/cst.mn *1e-6, 
					'-uR':-uR0/cst.kms , 
					'uph':uph0/cst.kms , 
					'-uR_bal':uR_bal/cst.kms , 
					'uph_bal':uph_bal/cst.kms, 
					'zeta':re['zeta'] ,	
					'u_max':np.sqrt(uR0**2+ uph0**2)/cst.kms	}	,
					 'v_dens_loglog_%s'%tstamp	, x = re['r_ax']/cst.au , xlim=[0.3,500] , ylim=[1e-4,1e5] ,
					 lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'], loglog=True)

		mp.plot( { 'uph/uR':-uph0/uR0 }	, 
					'uph2uR_%s'%tstamp, x = re['r_ax']/cst.au , 
					logy=True,	xlim=[0,500],ylim=[1e-4,100] )
	
		return 




#
# Calculator
#
class Calc_2d_map:

	def __init__(self , t , r_ax , th_ax ):
		self.t	  = t  
		self.r_ax = r_ax
		self.th_ax = th_ax
		self.M	  = 0
		self.m0 = 0.975
		self.P2			= lambda x: 0.5*(3*x**2-1)
		self.dP2_dtheta = lambda x: -3*x*np.sqrt(1-x**2)

	def get_mu0(self, mu_ax , zeta):
		brentq=0
		cubic =1
		if cubic:	
#			mu0=Symbol('mu0')			
#			root = np.array( [ solve(zeta*mu0**3 + (1-zeta)*mu0 - mu , mu0)[0] for mu in mu_ax ])
			root = [None]*len(mu_ax)
			for i,mu in enumerate(mu_ax):
				sol = np.roots( [ zeta, 0 , 1-zeta , - mu ] )	
				mp.say(sol)
				sol = sol[ 0<=sol ]
				sol = sol[ sol<=1+1e-5 ]

				if len(sol)==1 and np.isreal(sol[0]) :
					root[i] = np.real( sol[0] )
					#raise Exception(sol)
#					mp.say(root[i])
				else:
					raise Exception(sol)
#					print(sol)
#			root = np.array( [ np.real( np.roots( [ zeta, 0 , 1-zeta , - mu ] ) ) for mu in mu_ax ])
#			mp.say(root)
			root = np.array(root)

		if brentq:
			def eq(mu0,mu):
#				if mu == 1:
#					return zeta*mu0*(1 + mu0) + 1
#					return mu0 - 0.5*( -1 + np.sqrt( 1 - 4/zeta ))
#				elif mu == 0:
#					return	zeta - 1/(1 - mu0**2)
#					return mu0 - np.sqrt(1 - 1/zeta)
#				else:
					return zeta - ( 1 - mu/mu0)/(1 - mu0**2)  
			root = np.array( [ optimize.brentq(eq, 0, 1 , args=( mu ) )  for mu in mu_ax ] ) 
		return root

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
		sol = solve_ivp( f0 , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True, method='BDF',rtol = 1e-12,	atol = [1e-8,1e-15], vectorized=True)

		V0	= sol.y[0]
		al0 = sol.y[1]
		return interp1d(x,al0) , interp1d(x,V0)
		

	def if_one(self, cond , val ):
		return np.where( cond  , np.ones_like(val) , val )
	def if_zero(self, cond , val ):
		return np.where( cond  , np.zeros_like(val) , val )
	
	def get_Kinematics_CM( self, r , mu , Omg  , cs , t , M , Mdot , j0=None , cavity_angle=None ):
		SMALL0	= 1e-50
		SMALL1	= 1e-49
		j0 =  Omg * ( self.m0 * cs * t/2 )**2  if ( j0 is None ) else j0
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

		if func is None:
			al_0 = (0.5*self.m0/x**3)**(1/2)
			V_0 = -(2*self.m0/x)**(1/2)
		else:
			al_0 = func[0](x)
			V_0  = func[1](x)

		al_Q = -13/12*(self.m0/2)**(7/2)*x**(-5/2)
		V_Q  = -1/6*(self.m0/2)**(7/2)*x**(-3/2)
		W_Q  = -1/3*(self.m0/2)**(7/2)*x**(-3/2)

		al_M = 1/12*(self.m0/2)**(7/2)*x**(-5/2)
		V_M  = 1/6*(self.m0/2)**(7/2)*x**(-3/2)		

		al = al_0 + tau**2 * ( al_M + al_Q * self.P2(mu) )
		V  = V_0  + tau**2 * ( V_M	+ V_Q  * self.P2(mu) )
		W  =		tau**2 *   W_Q * self.dP2_dtheta(mu)
		m0x = x**2 * al_0 * (x-V_0)
		Gamma2 =  (0.5*m0x)**2*(1-mu**2)
		Gamma = tau**2 * Gamma2

		rho = Omg**2 /(4*np.pi*cst.G*tau**2)*al
		ur	= cs*V
		uth = cs*W
		uph = cs**2*Gamma/(Omg*r*np.sqrt(1-mu**2))

		return rho.clip(0), ur, uth, uph 

	def put_Disk_sph(self, r , th , M , Md , Rd , Td , CM=False):
		exponential_cutoff = True
		tapered_cutoff = False
		ind = -1
		R = r*np.sin(th)
		z = r*np.cos(th)
		if not par.Disk:
			return np.zeros_like(z)

		#if CM:
		#	pass
		
		if exponential_cutoff:
			Sigma_0 = Md/(2*np.pi*Rd**2)/(1-2/np.e)
			Sigma = Sigma_0 * (R/cst.au)**ind  * np.exp( -(R/Rd)**(2+ind) )

		if tapered_cutoff:
			Sigma_0 = Md/(2*np.pi*Rd**2) * (ind+3)
			Sigma = Md/(2*np.pi) * (ind+3)*Rd**(-ind-3)  * (R/cst.au)**ind

		H  = np.sqrt( cst.kB * Td / cst.mn ) / np.sqrt(cst.G*M/R**3)
		rho = Sigma/np.sqrt(2*np.pi)/H * np.exp( - 0.5*z**2/H**2	)

		return rho

	def urth_to_uRz( self, ur , up , th_ax ):
		uR = ur * np.sin(th_ax) + up * np.cos(th_ax) 
		uz = ur * np.cos(th_ax) - up * np.sin(th_ax)
		return uR, uz

	def give_Mass_and_time(self, Mdot ,t=None, Mstar=None):
		if not par.AutoMassTuning:
			return Mdot * t , t
		else: 
			return Mstar , Mstar / Mdot
		
	def give_Omega(self, cs,t,M , mode='const', v_CR=None , r_CR=None):
		GM = cst.G * M
		r_col = self.m0*0.5*cs*t
		if not par.Rotation:
			return 0

		if mode=="const":
			Omega = 1e-14

		if mode=='velocity_peak':
			## v_CR = velocity peak
			##		= sqrt(2) * GM / l
			## l  = (0.5*cs*t)**2 * Omega
			## --> Omega = sqrt(2) * GM / vpeak / (0.5*cs*t)**2 
			Omega = sqrt(2) * GM / v_CR / r_col**2
			
		if mode=='centrifugal_radius':
			## r_CR = given
			##		= l**2 /(GM)
			##	l  = (0.5*cs*t)**2 * Omega
			Omega = np.sqrt( GM * r_CR )/r_col**2
		
		return Omega	
	

	def print_params(self, t, M, T, cs, Omg, Mdot, r_in_lim, j0, rCB):
		def print_format( name , val , unit):
			print( name.ljust(10) + "is    {:10.2g}   ".format(val) + unit.ljust(10)	   )
		print_format( "T" , T , "K" )
		print_format( "cs" , cs/cst.kms , "km/s" )
		print_format( "t" , t/cst.yr , "yr" )
		print_format( "M" , M/cst.Msun , "Msun" )
		print_format( "Omega" , Omg , "s^-1" )
		print_format( "dM/dt" , Mdot/(cst.Msun/cst.yr) , "M/yr" )
		print_format( "r_lim" , r_in_lim/cst.au , "au" )
		print_format( "j0" , j0/(cst.kms*cst.au)  , "au*km/s" )
		print_format( "j0" , j0/(cst.kms*cst.pc)  , "pc*km/s" )
		print_format( "rCB" , rCB/cst.au , "au" )

	def calc(self):
		print("start calc")

		#
		# Set parameters
		#
		T	=	10
		cs	= np.sqrt( cst.kB * T/cst.mn )
		Mdot = cs**3*self.m0/cst.G

		self.M, self.t = self.give_Mass_and_time(Mdot, Mstar=0.36*cst.Msun)		
		Omg = self.give_Omega( cs , self.t , self.M , mode='centrifugal_radius', r_CR=100*cst.au)
	
		r_in_lim = cs*Omg**2* self.t**3
		j0 = Omg * ( self.m0 * cs * self.t * 0.5 )**2
		rCB  = self.get_rCB(self.M, Omg, cs, self.t ) # 50 * cst.au
		req  = rCB * 2 * np.sin( 45/180 * np.pi )**2

		self.print_params( self.t, self.M, T, cs, Omg, Mdot, r_in_lim, j0, rCB)	
		self.res = {'r_ax':self.r_ax, 'th_ax':self.th_ax, 't':self.t ,'M':self.M, 'r_in_lim':r_in_lim, 'r_CB_0': rCB}	
		funcs = self.get_0th_order_function(self.r_ax) if use_solution_0th else None

		mu	  = np.cos( self.th_ax )
		si	  = np.sqrt( 1 - mu**2 )	
		Md	= 0.1*self.M
		Td	= 30

		#	
		# Main loop in r-axis
		#
		for r in self.r_ax:
			R_ax  = r * np.sin(self.th_ax)
			z_ax  = r * np.cos(self.th_ax)

			if CM:	
#			if r < r_in_lim or CM:
				rho, ur, uth, uph, zeta, mu0  = self.get_Kinematics_CM( r , mu , Omg  , cs , self.t , self.M , Mdot ,j0=j0, cavity_angle=45) 
				uR, uz = self.urth_to_uRz( ur , uth , self.th_ax )
				dic = {'R':R_ax, 'z': z_ax, 'den':rho, 
						'ur':ur,'uph':uph,'uth':uth,'uR':uR,'uz':uz , 
						'rr':np.full_like(self.th_ax , r) , 'tt': self.th_ax ,
						'zeta':zeta, 'mu0':mu0 }

			elif TSC:
#			else:
				#rho, ur, uth, uph = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t	, func=funcs)
				#zeta = ( Omg*(m0*cs*self.t/2)**2 )**2/(cst.G*self.M*r)
				#mu0 = np.zeros_like(mu)
				rho_TSC, ur_TSC, uth_TSC, uph_TSC = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t , func=funcs)	
				uR_TSC, uz_TSC = self.urth_to_uRz( ur_TSC , uth_TSC , self.th_ax )
				dic.update( {  'den_TSC':rho_TSC, 'ur_TSC':ur_TSC,'uph_TSC':uph_TSC,'uth_TSC':uth_TSC,'uR_TSC':uR_TSC,'uz_TSC':uz_TSC }  )

			rho_disk = self.put_Disk_sph( r , self.th_ax , self.M , Md , rCB , Td)	
			disk_reg = rho_disk > rho
			v_Kep	 = np.sqrt(cst.G * self.M / R_ax )
			vphi_tot = np.where( disk_reg , v_Kep  , uph  )
			
			dic.update( {  'den_tot':rho + rho_disk,  'vr_tot':np.where( disk_reg , 0  , ur  ) , 
								'vphi_tot':np.where(  disk_reg , v_Kep	, uph  ), 'vth_tot':np.where( disk_reg , 0	, uth  )	}  )
	
			for key, vprof in dic.items():
				if not key in self.res:
					self.res[key] = []
				self.res[key].append( vprof )
		
		for k,v in self.res.items():
			self.res[k] = np.array(v)

		# Save to pickle
		stamp = "{:.0e}".format( self.t/cst.yr ) if par.pickle_name == 'time' else par.pickle_name
		pd.to_pickle( self.res , "res_%s.pkl"%stamp)	
		print("Saved: %s"%"res_%s.pkl"%stamp )	
		return self.res
	
##########################################################################################################################

###########
if __name__ == '__main__':
	main()
###########

