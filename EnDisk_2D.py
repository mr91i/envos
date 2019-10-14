#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import natconst as cst
print(vars(cst))
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import os, copy, pickle

#import mytools as my
import plotter as mp

from scipy import integrate, special, optimize
from scipy.interpolate import interp1d,interp2d,griddata
from scipy.integrate import solve_ivp

from sympy import *

mpl.rc('xtick.major',size=6,width=2,pad=6,)
mpl.rc('ytick.major',size=6,width=2,pad=2)

## Global Parameters ##
alpha  = 0.01
#Ms	   = 1	
#Rs	   = 1
f_dg   = 1e-4
M_disk = 0.03
scattering			 = 0
variable_photosphere = 0
SL_vs_Mdot			 = 0
Mdot_err			 = 1
Disk_Structure		 = 1
Calc_2D				 = 1
Simple				 = 1
acc_model = 'Hartmann2016'
#######################

high_res = 0
Ohashi2014 = 0
CM = 1
TSC = 0
use_solution_0th = 0
mp.dbg = 0

class Param:
	def __init__(self):
		self.Rotation = 1

par = Param()

def main():
	global Ms, Ls, alpha, f_dg, acc_model
		
	if high_res :
		r_ax  = np.logspace( 0	, 3 , 1001	)*cst.au
		th_ax = np.linspace( 0	 , np.pi/2 , 721	)
	else:
		r_ax  = np.logspace( 0	 , 3 , 601	 )*cst.au
		th_ax = np.linspace( 0	 , np.pi/2 , 73    )
	print(r_ax)
	if Calc_2D:
		for t in np.linspace( 5e5 , 5e5 , 1):## #year
		#for t in np.linspace( 1e5 , 1e6 , 10):## #year
#		for t in np.logspace( 4 , 6 , 5):## year
			if Ohashi2014:
				t=0
			Cm = Calc_2d_map( t , r_ax , th_ax	)
			Cm.calc()
			Plots( Cm.res )
			if Ohashi2014:
				break

def slice_at_midplane( tt,	*a_rths  ):
	return [  a_rth[ tt==np.pi/2 ]	 for a_rth in a_rths	  ]	

def Plots( re ):
		t_yr = re['t']
		tstamp = f'{t_yr/cst.yr:.0e}'

		rr_p , tt_p =  np.meshgrid( re['r_ax'] , re['th_ax'] , indexing='ij' )
		RR_p = rr_p * np.sin( tt_p )  
		zz_p = rr_p * np.cos( tt_p )  

		r_lim = 500 

		def draw_map( v , name , rang , log=True, V=None, cbl=None, **kwargs):
			mp.map( RR_p/cst.au , zz_p/cst.au , v	, f"map_{name}_{tstamp}", 
						xl='Radius [au]' , yl='Height [au]', cbl=cbl,
						logx=False, logy=False, logcb=log, leg=False, 
						xlim=[0,r_lim], ylim=[0,r_lim], cblim=rang, **kwargs )

		def draw_map_name( vname ,*args, **kwargs):
			draw_map( re[vname] , vname , *args, **kwargs)

		Vec = np.array( [ re[ 'uR' ] , re[ 'uz' ] ] )#/np.sqrt( re[ 'ux' ]**2	  +  re[ 'uy' ]**2 )
#		 draw_map( re['den'], 'den',[1e-20, 1e-15] , cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec,n_sl=40)
		draw_map( re['den'], 'den', [1e-21, 1e-16] , cbl=r'log Density [g/cm$^{3}$]', div=10, Vector=Vec,n_sl=40)
		draw_map( re['mu0']/np.cos( re['tt'] ), 'mu0_mu',[0, 10] , cbl=r'$\mu_{0}/\mu$', div=10, Vector=Vec,n_sl=40,log=False)

		den0, uR0 , uph0 = slice_at_midplane( tt_p , re['den'] , re['uR'] , re['uph']  )
		den0_TSC, uR0_TSC , uph0_TSC = slice_at_midplane( tt_p , re['den_TSC'] , re['uR_TSC'] , re['uph_TSC']  )

		mp.say( den0, uR0 , uph0 )

		uph_bal = np.sqrt( re['zeta']*cst.G*re['M']/re['r_ax']	 )
		uR_bal = np.sqrt(2*cst.G*re['M']/re['r_ax'] - uph_bal**2 ) 



		mp.plot( { 'log nH2 - 6':np.log10(den0/cst.mn) -6, '-uR':-uR0/cst.kms , 'uph':uph0/cst.kms,	
				'log nH2_TSC - 6':np.log10(den0_TSC/cst.mn) -6,'-uR_TSC':-uR0_TSC/cst.kms , 'uph_TSC':uph0_TSC/cst.kms }	, f'vden_compare_{tstamp}'	  , x = re['r_ax']/cst.au ,
				xlim=[0,500],ylim=[-2,10] , lw=[ 3, 3,3,6,6,6], c=['k', ], ls=['-','-','-','--','--','--'], vl=[re["r_CB_0"]*2/cst.au, re["r_in_lim"]/cst.au])


		mp.plot( { 'log nH2 - 6':np.log10(den0/cst.mn) -6, '-uR':-uR0/cst.kms , 'uph':uph0/cst.kms , '-uR_bal':uR_bal/cst.kms , 'uph_bal':uph_bal/cst.kms , 'u_max':np.sqrt(uR0**2+ uph0**2)/cst.kms }	, f'v_dens_{tstamp}'	, x = re['r_ax']/cst.au ,
				xlim=[0,500],ylim=[-2,7] , lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'])
		mp.plot( { 'nH2/1e6':den0/cst.mn *1e-6, '-uR':-uR0/cst.kms , 'uph':uph0/cst.kms , '-uR_bal':uR_bal/cst.kms , 'uph_bal':uph_bal/cst.kms,'zeta':re['zeta'] ,	'u_max':np.sqrt(uR0**2+ uph0**2)/cst.kms	}	, f'v_dens_loglog_{tstamp}'	, x = re['r_ax']/cst.au ,
				xlim=[0.3,500],ylim=[1e-4,1e5] , lw=[ None, 3,3,6,6], c=['k', ], ls=['-','-','-','--','--'], loglog=True)
		mp.plot( { 'uph/uR':-uph0/uR0 }	  , f'uph2uR_{tstamp}'	  , x = re['r_ax']/cst.au , logy=True,	xlim=[0,500],ylim=[1e-4,100] )
	


		return 

		mp.do_map( R_au , z_au ,  np.sqrt( re[ 'ur' ]**2 + re[ 'uph' ]**2 + re[ 'uth' ]**2)	, f"map_u_{t:.0e}" ,
					xl='Radius [au]' , yl='Height [au]', cbl='',
					logx=False, logy=False, logcb=True, leg=False, xlim=[0,r_lim], ylim=[0,r_lim], cblim=[1e5,1e7], data='1+1D',Vector=Vec)



		mp.do_plot( {"V_phi": re[ 'uph' ][:,0]/1e5,  "V_phi_tot": re[ 'vphi_tot' ][:,0]/1e5 }	, f"vphi_{t:.0e}" , xl='Distance From Star [au]' , yl='Rotation Velocity [km/s]',
			logx=True, logy=True, ls=['--','-'], lw=[3] ,
			c=[c_def[0],c_def[1],'salmon','lightskyblue','darkgray','darkgray'] , leg=False, xlim=[1,1000],ylim=[1e-1,1e2],
			lbs=['Our model','Conventional','Only viscous heating','Only irradiation heating',None,None])



class Calc_2d_map:

	def __init__(self , t , r_ax , th_ax ):
		self.t	  = t  * cst.year
		self.r_ax = r_ax
		self.th_ax = th_ax
		self.M	  = 0
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
					raise Exception(sol)
#					mp.say(root[i])
				else:
					print(sol)
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
		



	def rotate(v, th):
		a = np.array([ [np.cos(th), -np.sin(th)] , [np.sin(th),  np.cos(th)] ])
		return np.dot(a, v) 

	def if_one(self, cond , val ):
		return np.where( cond  , np.ones_like(val) , val )
	def if_zero(self, cond , val ):
		return np.where( cond  , np.zeros_like(val) , val )
	
	def get_Kinematics_CM( self, r , mu , Omg  , cs , t , M , Mdot , j0=None	 ):
		SMALL0	= 1e-50
		SMALL1	= 1e-49
		m0 = 0.975
		j0 =  Omg * ( m0 * cs * t/2 )**2  if ( j0 is None ) else j0
		zeta =	j0**2 / ( cst.G * M * r	)
		mu0  = self.get_mu0( mu ,zeta) 
		sin, sin0 = np.sqrt( 1 - mu**2 ) ,	np.sqrt( 1 - mu0**2 ) 
		vff	= np.sqrt( cst.G * M / r )
		
#		mu0 = np.where( mu0==0 , SMALL0 , mu0 )
		sin = np.where( sin==0 , SMALL0 , sin )
		Mdot = Mdot * ( mu0 < 1/np.sqrt(2) ) * np.ones_like( mu0 )
#		Mdot = Mdot * ( mu0 < 0.1 ) * np.ones_like( mu0 )
	
#		mu_to_mu0 = np.where( mu0 < SMALL1 , 1-zeta , mu/mu0 )
#		mu_to_mu0 = np.where( mu0 < SMALL1 , 0 , mu/mu0 )
		mu_to_mu0 = mu/mu0
		ur	= - vff * np.sqrt( 1 + mu_to_mu0 )
		uth =	vff * self.if_zero( sin<SMALL1 , ( mu0 - mu )/sin ) * np.sqrt( 1 + mu_to_mu0 ) 
		uph =	vff * self.if_one(	sin<SMALL1 , sin0/sin		 ) * np.sqrt( 1 - mu_to_mu0 )
		
		rho = - Mdot / (4 * np.pi * r**2 * ur) / (1 + 2*zeta*self.P2(mu0))

		return rho, ur, uth, uph , zeta, mu0
	
	def get_rCB(self, M, Omg, cs, t , j0=None):
		m0 = 0.975
		j0 =  Omg * ( m0 * cs * t * 0.5 )**2  if ( j0 is None ) else j0
		return	j0**2 / ( cst.G * M * 2 )

	def get_Kinematics_TSC( self, r , mu , Omg , cs , t  , func=None):
		m0 = 0.975
		x =  r/(cs*t)
		tau = Omg*t

		if func is None:
			al_0 = (0.5*m0/x**3)**(1/2)
			V_0 = -(2*m0/x)**(1/2)
		else:
			al_0 = func[0](x)
			V_0  = func[1](x)

		al_Q = -13/12*(m0/2)**(7/2)*x**(-5/2)
		V_Q  = -1/6*(m0/2)**(7/2)*x**(-3/2)
		W_Q  = -1/3*(m0/2)**(7/2)*x**(-3/2)

		al_M = 1/12*(m0/2)**(7/2)*x**(-5/2)
		V_M  = 1/6*(m0/2)**(7/2)*x**(-3/2)		

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
		ind = -1
		R = r*np.sin(th)
		z = r*np.cos(th)
		if CM:
			pass
		
		if 1:
			Sigma_0 = Md/(2*np.pi*Rd**2)/(1-2/np.e)
			Sigma = Sigma_0 * (R/cst.au)**ind  * np.exp(- (R/Rd)**(2+ind))
		else:
			Sigma_0 = Md/(2*np.pi*Rd**2) * (ind+3)
			Sigma = Md/(2*np.pi) * (ind+3)*Rd**(-ind-3)  * (R/cst.au)**ind

		H  = np.sqrt( cst.kB * Td / cst.mn ) / np.sqrt(cst.G*M/R**3)
		rho = Sigma/np.sqrt(2*np.pi)/H * np.exp( - 0.5*z**2/H**2	)
		return rho

	def urth_to_uRz( self, ur , up , th_ax ):
		uR = ur * np.sin(th_ax) + up * np.cos(th_ax) 
		uz = ur * np.cos(th_ax) - up * np.sin(th_ax)
		return uR, uz

#	def u_balistic( j , )
#		uph_bal = j/r	
#		uR_bal = np.sqrt(2*cst.G*M/r - uph_bal**2 )	
#		return 

	def calc(self):
		print("start calc")
		Disk = 1
		self.res= {}
		T	=	10
#		cs	= 0.35e5 
		cs	= np.sqrt( cst.kB * T/cst.mn )
		print(f"cs is {cs/1e5:.2f} km/s")
		m0	= 0.975
	
		Omg = 1e-14 if par.Rotation else 0

		r_in_lim = cs*Omg**2* self.t**3
#		self.r_ax = self.r_ax[ self.r_ax > r_in_lim ]
		Mdot = cs**3*m0/cst.G
		if Ohashi2014:
			j0 = 1.882e20 # = 6.1e−4 km s−1 pc (Ohashi+2014)
			self.M = 0.3*cst.Msun
		else:
			j0=None
			self.M = Mdot * self.t 
		
		if use_solution_0th:
			funcs =  self.get_0th_order_function(self.r_ax) 
		else:
			funcs = None

		print("M is ",self.M/cst.Msun)

		for r in self.r_ax:
			print(r)
			R_ax  = r * np.sin(self.th_ax)
			z_ax  = r * np.cos(self.th_ax)
			mu	  = np.cos( self.th_ax )
			si	  = np.sqrt( 1 - mu**2 )	
			rCB = self.get_rCB(self.M, Omg, cs, self.t )

			if CM:	
#			if r < r_in_lim or CM:
				rho, ur, uth, uph, zeta, mu0  = self.get_Kinematics_CM( r , mu , Omg  , cs , self.t , self.M , Mdot ,j0=j0) 
				uR, uz = self.urth_to_uRz( ur , uth , self.th_ax )
				dic = {'R':R_ax, 'z': z_ax, 'den':rho, 'ur':ur,'uph':uph,'uth':uth,'uR':uR,'uz':uz , 'rr':np.full_like(self.th_ax,r) , 'tt': self.th_ax ,'zeta':zeta, 'mu0':mu0 }

			elif TSC:
#			else:
				#rho, ur, uth, uph = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t	, func=funcs)
				#zeta = ( Omg*(m0*cs*self.t/2)**2 )**2/(cst.G*self.M*r)
				#mu0 = np.zeros_like(mu)
				rho_TSC, ur_TSC, uth_TSC, uph_TSC = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t , func=funcs)	
				uR_TSC, uz_TSC = self.urth_to_uRz( ur_TSC , uth_TSC , self.th_ax )
				dic.update( {  'den_TSC':rho_TSC, 'ur_TSC':ur_TSC,'uph_TSC':uph_TSC,'uth_TSC':uth_TSC,'uR_TSC':uR_TSC,'uz_TSC':uz_TSC }  )

			if Disk:
				Md	= 0.1*self.M
				Rd	= get_rCB(self.M, Omg, cs, self.t ) # 50 * cst.au
				Td	= 30
				rho_disk	 = self.put_Disk_cyl( R , z_ax , self.M , Md , Rd , Td)	
#				shock_region = ( rho_disk / rho > (cst.gamma +1)/(cst.gamma -1)  )
				v_Kep		 = np.sqrt(cst.G * self.M/R )
				vphi_tot	 = uph * 1
				vphi_tot[rho_disk > rho] = v_Kep[rho_disk > rho]
				dic.update( {  'den_tot':rho + rho_disk, 'shock':shock_region, 'vphi_tot':vphi_tot	  }  )
	
			for key, vprof in dic.items():
				if not key in self.res:
					self.res[key] = []
				self.res[key].append( vprof )
		
		for k,v in self.res.items():
			self.res[k] = np.array(v)

		self.res.update( {'r_ax':self.r_ax, 'th_ax':self.th_ax, 't':self.t ,'M':self.M, 'r_in_lim':r_in_lim, 'r_CB_0': rCB} )
		pd.to_pickle( self.res , f"res_{self.t/cst.yr:.0e}.pkl")
		return 


#	exit()

##########################################################################################################################

###########
if __name__ == '__main__':
	main()
###########


