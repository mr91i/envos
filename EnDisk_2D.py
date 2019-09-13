#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cst as cst
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import os,copy
from col import *
import mytools as my
#from easyplot import EasyPlot
from scipy import integrate
from scipy import special
from scipy.interpolate import interp1d,interp2d,griddata
import calc_T2 as cT
import opacity as op
from scipy import optimize
import pandas as pd
import pickle
import myplot as mp
from sympy import *
from scipy.integrate import solve_ivp

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


def main():
	global Ms, Ls, alpha, f_dg, acc_model
		
	M	  =  1 * cst.Msun  # Final mass

	if high_res :
		r_ax  = np.logspace( 0	, 3 , 1001	)*cst.au
		th_ax = np.linspace( 0	 , np.pi/2 , 721	)
	else:
		r_ax  = np.logspace( 0	 , 3 , 601	 )*cst.au
		th_ax = np.linspace( 0	 , np.pi/2 , 73    )


#	th_ax = np.concatenate( [ np.linspace( 0	 , np.pi/2*75/90 ,	15	) , np.linspace( np.pi/2*75/90	 , np.pi/2 , 30   ) ] )
	
	if Calc_2D:
		for t in np.linspace( 1e5 , 1e6 , 10):## #year
#		for t in np.logspace( 4 , 6 , 5):## year
			if Ohashi2014:
				t=0
			Cm = Calc_2d_map( t , r_ax , th_ax ,  M )
			Cm.calc()
			Plots( Cm.res )
			if Ohashi2014:
				break
			#PVden_diagram( t , r_ax , th_ax , Cm.res )


def angle_slice_sca(r_ax,z_ax,v,th):
	func_v = interp1d(z_ax,v,fill_value='extrapolate')
	z_th = np.tan( th ) * r_ax
	v_slice = func_v(z_th)
	r_ax = r_ax / np.cos( th )
	return v_slice, r_ax 

def slice_at_midplane( tt,	*a_rths  ):
	print(tt,  a_rths)
	return [  a_rth[ tt==np.pi/2 ]	 for a_rth in a_rths	  ]	


def reproduce_map( v , x1_ax , x2_ax , X1_ax , X2_ax  ):
# Example : den = reproduce_map( re['den'] , re['r_ax'] , re['th_ax'] , rr , tt  )
	print(len(v),len(v[0]),len(x1_ax),len(x2_ax) , v.shape)
	f = interp2d( np.log10( x1_ax ) ,  x2_ax  , np.log10( v.T ) )
	fv = np.vectorize( f )
	v_new = 10**fv(  np.log10( X1_ax ) ,  X2_ax  )
	return v_new

#def Calc_2D_Envelope_Disk_Structure( t , M , mp ):
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
		mp.say( den0, uR0 , uph0 )

		uph_bal = np.sqrt( re['zeta']*cst.G*re['M']/re['r_ax']	 )
		uR_bal = np.sqrt(2*cst.G*re['M']/re['r_ax'] - uph_bal**2 ) 



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

#		uph_sl , r_ax = angle_slice_sca(R_ax, re["z"][0]/cst.au ,	re[ 'uph' ], np.pi/3 )


class Calc_2d_map:

	def __init__(self , t , r_ax , th_ax , M):
		self.t	  = t  * cst.yr
		self.r_ax = r_ax
		self.th_ax = th_ax
		self.M	  = M 
		self.P2 = lambda x: 0.5*(3*x**2-1)
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
					mp.say(root[i])
				else:
					mp.say(sol).exit()
#			root = np.array( [ np.real( np.roots( [ zeta, 0 , 1-zeta , - mu ] ) ) for mu in mu_ax ])
			mp.say(root)
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

	def get_0th_order_function(self):
		def f0(x, y):
			ret = np.zeros_like(y)
			V0	= y[0]
			al0 = y[1]
			ret[0] =	 (al0*(x - V0) - 2/x)*(x-V0)/((x-V0)**2 -1)
			ret[1] = al0*(al0 - 2/x*(x - V0))*(x-V0)/((x-V0)**2 -1)
			return ret
		t = 1e-2
		x = np.logspace( np.log10(1/t) , np.log10(t**2) , 10000 ) ## determines size of MC 

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
	
	

#		mu_to_mu0 = np.where( mu0 < SMALL1 , 1-zeta , mu/mu0 )
#		mu_to_mu0 = np.where( mu0 < SMALL1 , 0 , mu/mu0 )
		mu_to_mu0 = mu/mu0
		ur	= - vff * np.sqrt( 1 + mu_to_mu0 )
		uth =	vff * self.if_zero( sin<SMALL1 , ( mu0 - mu )/sin ) * np.sqrt( 1 + mu_to_mu0 ) 
		uph =	vff * self.if_one(	sin<SMALL1 , sin0/sin		 ) * np.sqrt( 1 - mu_to_mu0 )

#		mp.say( (ur/uph)**2 / zeta ).exit()
		
		rho = - Mdot / (4 * np.pi * r**2 * ur) / (1 + 2*zeta*self.P2(mu0))

		return rho, ur, uth, uph , zeta, mu0

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



	

	def put_Disk_sph(self, r , th , M , Md , Rd , Td):
		ind = -1
		R = r*np.sin(th)
		z = r*np.cos(th)
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
		return uR,uz
#		R = np.array( [ [  np.sin(th_ax) , np.cos(th_ax)  ] , [  np.cos(th_ax) , -np.sin(th_ax)  ] ] )
#		return np.dot( R , np.array( [ ur,up ] ).T )

#	def u_balistic( j , )
#		uph_bal = j/r	
#		uR_bal = np.sqrt(2*cst.G*M/r - uph_bal**2 )	
#		return 

	def calc(self):
		Disk = 0
		self.res= {}
		T = 10
		cs	= 0.35e5 
		cs	= np.sqrt( cst.kB * T/cst.mn )
		print(f"cs is {cs/1e5:.2f} km/s")
		m0	= 0.975
		Omg = 1e-14
		Md = 0.1 * cst.Msun
		Rd = 50 * cst.au
		Td	= 30
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
			funcs =  self.get_0th_order_function() 
		else:
			funcs = None

		print("M is ",self.M/cst.Msun)

		for r in self.r_ax:
			R_ax  = r * np.sin(self.th_ax)
			z_ax  = r * np.cos(self.th_ax)
			mu	  = np.cos( self.th_ax )
			si	  = np.sqrt( 1 - mu**2 )	
#			if CM:	
			if r < r_in_lim or CM:
				rho, ur, uth, uph, zeta, mu0  = self.get_Kinematics_CM( r , mu , Omg  , cs , self.t , self.M , Mdot ,j0=j0) 
#			elif TSC:
			else:
				rho, ur, uth, uph = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t	, func=funcs)
				zeta = ( Omg*(m0*cs*self.t/2)**2 )**2/(cst.G*self.M*r)
				mu0 = np.zeros_like(mu)
			rho_TSC, ur_TSC, uth_TSC, uph_TSC = self.get_Kinematics_TSC( r , mu , Omg , cs , self.t , func=funcs)	
	

			uR, uz = self.urth_to_uRz( ur , uth , self.th_ax )
			dic = {'R':R_ax, 'z': z_ax, 'den':rho, 'ur':ur,'uph':uph,'uth':uth,'uR':uR,'uz':uz , 'rr':np.full_like(self.th_ax,r) , 'tt': self.th_ax ,'zeta':zeta, 'mu0':mu0 }
			uR_TSC, uz_TSC = self.urth_to_uRz( ur_TSC , uth_TSC , self.th_ax )
			dic.update( {  'den':rho, 'ur':ur,'uph':uph,'uth':uth }  )
			if Disk:
				rho_disk = self.put_Disk_cyl( R , z_ax , self.M , Md , Rd , Td)	
				shock_region = ( rho_disk / rho > (cst.gamma +1)/(cst.gamma -1)  )
				v_Kep = np.sqrt(cst.G * self.M/R )
				vphi_tot = uph * 1
				vphi_tot[rho_disk > rho] = v_Kep[rho_disk > rho]
				dic.update( {  'den_tot':rho + rho_disk, 'shock':shock_region, 'vphi_tot':vphi_tot	  }  )
	
			for key, vprof in dic.items():
				if not key in self.res:
					self.res[key] = []
				self.res[key].append( vprof )
		
		for k,v in self.res.items():
			self.res[k] = np.array(v)

		self.res.update( {'r_ax':self.r_ax, 'th_ax':self.th_ax, 't':self.t ,'M':self.M, 'r_in_lim':r_in_lim} )
		pd.to_pickle( self.res , f"res_{self.t/cst.yr:.0e}.pkl")
		return 


def PVden_diagram( t , r_ax , th_ax , re , beam_size=0):
	direct=1
	interp=0
	n=1
	m=1
	
	den = re['den'][::n,::m]
	uR	= re['uR'][::n,::m]
	uph = re['uph'][::n,::m]
	RR	= re['R'][::n,::m]
	zz	= re['z'][::n,::m]
	rr	= re['rr'][::n,::m]
	tt	= re['tt'][::n,::m]
	ph_ax = np.linspace( 0 , np.pi, 180 )  # ph=0 <-> center of imaginary xy grid , ph goes counter clockwise
#	LS_ax = np.linspace( -100 , 100, 101 )*cst.au

	x_pRz = np.array([ RR*np.sin(ph) for ph in ph_ax	 ])
	y_pRz =  np.array([ RR*np.cos(ph) for ph in ph_ax	])
	z_pRz = np.array([ zz for ph in ph_ax	])
	uLS_pRz = np.array([ uR*np.cos(ph) - uph*np.sin(ph) for ph in ph_ax   ])
	den_pRz = np.array([ den for ph in ph_ax   ])
	for u in uLS_pRz:
		print(u)

	# pRz -> 

	if direct:
#		print( x_pRz , uLS_pRz )
		P_ax = np.linspace( 0, 500 , 301 )*cst.au
#		P_ax = np.logspace( -1, 3 , 201 )*cst.au
		V_ax = np.linspace( -3e5, 3e5 , 302 )
		
		uLS_pRz = uLS_pRz[ z_pRz < 100*cst.au ]
		x_pRz = x_pRz[ z_pRz  < 100*cst.au ]

		data = [ x_pRz , uLS_pRz ]
		pv = np.array([ [p,v] for p, v in zip( x_pRz.flatten() , uLS_pRz.flatten() )  ])
		H, edges = np.histogramdd( pv , bins = (P_ax,V_ax)	)
#		H, edges = np.histogramdd( pv , bins = (P_ax,V_ax)	, weights = den_pRz.flatten())
#		print( H ):wq

		xx, yy = np.meshgrid( edges[0][:-1], edges[1][:-1] , indexing='ij' )
#		print(xx.shape,yy.shape,H.shape)
#		mp.map( xx/cst.au , yy/cst.au , H , 'test', cblim=[1e-15,1e-11] , logcb=False )

		mp.map( xx/cst.au , yy/1e5 , H , f'test_{t/cst.yr:.0e}' , logcb=True )

		return



	if interp:
		xi_ax = np.linspace( 0 , 100, 16 )*cst.au # perpendicular to observer, horizontal
		yi_ax = np.linspace( -100 , 100, 31 )*cst.au # parallel to observaer
		zi_ax = np.linspace( 0 , 10, 4 )*cst.au # perpendicular to observer, up

		xi_gr , yi_gr , zi_gr = np.meshgrid( xi_ax , yi_ax , zi_ax,	indexing='xy')
		#beam_centers_xy = np.array( [ [ xi , 0	] for xi in xi_ax  ] )

		xyz = np.array([ [x,y,z] for x, y , z in zip( x_pRz.flatten() , y_pRz.flatten() , z_pRz.flatten() )  ])

		mth = 'linear'
		den_gr = griddata(xyz, den_pRz.flatten(), (xi_gr, yi_gr, zi_gr) ,method=mth)
		uLS_gr = griddata(xyz, uLS_pRz.flatten(), (xi_gr, yi_gr, zi_gr) ,method=mth)

	for d in den_gr:
		print(d)


	exit()

#	v = vel

#	z = integrate(	 )
	
	





def min_smooth(a,b):
	return (a**(-1)+b**(-1))**(-1)
def min_array( *arrays ):
	return np.array( [ min(vals) for vals  in zip(*arrays) ] )
def max_array( *arrays ):
	return np.array( [ max(vals) for vals  in zip(*arrays) ] )

k_BL94 = op.func_kR_BL94()
def kappa_R(T):	
	kp = 5 * (f_dg/0.01) * np.ones_like(T)	 
	return kp		

def kappa_R_z(T,rho):
	return np.array( [k_BL94(t,r)[0]  for t,r in zip(T,rho)] )

def Omega_K(r):
	return (cst.G* Ms *cst.Msun/r**3)**0.5
def Omega_K_z(r, z, H, q_rho = 0, q_T = 0 ):
	return Omega_K(r) * ( 1 - (H/r)**2 * ( q_rho + q_T + 0.5 * q_T * (z/H)**2  ))	
def Cs(T):
	return np.sqrt( cst.k_B * T /(2.34*cst.amu) )
def Hgas(r,T):
	return Cs(T)/Omega_K(r)

def get_Tacc(r,T,Mdot,alpha):
#	f_heat_dist = 0.5
	f_heat_dist = 1.0
	Mstar  = Ms*cst.Msun
	Omega  = Omega_K(r)
	Sigma  = Mdot / (3*cst.pi*alpha*Cs(T)**2 / Omega  )
	Tacc   = ( ( 3 * cst.G * Mstar * Mdot  ) / ( 8*cst.pi * cst.sigma_SB*r**3 ) )**0.25
	Tacc  *= (f_heat_dist*3/4*kappa_R(T)* Sigma*0.5 + np.sqrt(3)/4 )**0.25
	return Tacc

def get_new_Tacc(r,T,Mdot,alpha,eff):
	Omega  = Omega_K(r)
	Sigma  = Mdot / (3*cst.pi*alpha*Cs(T)**2 / Omega  )
	Tacc   = ( eff*( 3 * cst.G * Ms*cst.Msun * Mdot  ) / ( 8*cst.pi * cst.sigma_SB*r**3 ) )**0.25
	return Tacc

def get_Tirr(r,T,Mdot,alpha,Ls,h_ph=4):
	dlnh_dlnr  = -3/7*1/2 + 1.5
#	dlnh_dlnr  = -0.5*1/2 + 1.5
	f_shade  = 1/2
	#dlnh_dlnr = ( np.gradient( np.log(T),r)*r *(0.5) + 1.5 ).clip(0)
	#print(dlnh_dlnr)
	cs		   = Cs(T)
	Omega	   = Omega_K(r)
	kappa	   = kappa_R(T)
	h_surf	   = h_ph * cs/Omega
	Sigma	   = Mdot / (3*cst.pi*alpha*cs**2 / Omega  )
	tau		   = 0.5*kappa*Sigma
	mu0		   =  0.4 * Rs*cst.Rsun/r + h_surf/r * (dlnh_dlnr - 1)
	E0		   = f_shade * Ls * cst.Lsun /( 4*cst.pi * r**2 )
	T_irr	   = ( E0/4/cst.sigma_SB	)**0.25

#	T_irr	  *= np.array( [ ( ( mu0 + 0.5*np.exp(-tau/mu0)	)**0.25  if hp!=0 else (2*( mu0 + np.exp(-tau/mu0)	) )**0.25 )	for hp in h_ph	] )

#	if h_ph !=0:i

	if not scattering:
		T_irr	  *= ( mu0*(2 + 3*mu0) + (1-3*mu0**2)*np.exp(-tau/mu0)	)**0.25	
#		exit()

	if scattering:
		q = 1 # tau_s / tau_d
		k_abs = kappa_R(T)
		k_sca = k_abs*4
		sig_sca = k_sca / (k_abs + k_sca)
		alph = 1- sig_sca	
		beta = np.sqrt(3*alph)
		C1 = - 3*sig_sca*mu0**2/(1-beta**2*mu0**2)
		C2 = sig_sca*(2+3*mu0)/(beta*(1+2/3*beta) )/(1-beta**2*mu0**2)
		C1_ = (1+C1)*(2+3*mu0/q) + C2*(2+3/beta/q)
		C2_ = (1+C1)/mu0 * (q-3*mu0**2/q)
		C3_ = C2 * beta * (q-3/q/beta**2)
		T_irr *= (0.5*alph*mu0*( C1_ + C2_ * np.exp(-q*tau/mu0) + C3_*np.exp(-beta*q*tau) ) )**0.25
		#print(T_irr)



#	else:
#	T_irr	  *= (	mu0 + np.exp(-tau/mu0) )**0.25
#	T_irr	  *= (2)**0.25	if h_ph ==0 else 1 


#	T_irr	  *= ( mu0 + 0.5*np.exp(-tau/mu0)	)**0.25
#	T_irr *= ( mu0	)**0.25
#	T_irr	  *= (	0.5*np.exp(-tau/mu0)   )**0.25
#	T_irr	  = 110 * r**(-3/7) * Ls**(2/7)
	return T_irr

#def get_z_ph(z,rho,kappa):
def get_h_ph(r_list,H,Sigma,kappa):
## solve: tau_0 * erfc(z/sqrt(2)) = f
##	-->   z/sqrt(2) = erfcinv( f/tau )	
	shade_in_thin_region = 1
	if variable_photosphere:
		f=1-1/np.e
		#f = 1
#		print(	special.erfcinv( 2/(Sigma*kappa) ).clip(0)*np.sqrt(2)  )
#		return special.erfcinv( 2/(Sigma*kappa) ).clip(0)*np.sqrt(2)

		dlnh_dlnr = -3/7*1/2 + 1.5
		#dlnh_dlnr = np.gradient(np.log(H),r_list)*r_list
	#	print(dlnh_dlnr)
		Grazing_Angle = lambda x,r : 0.4 * Rs*cst.Rsun / r + x * H/r * (dlnh_dlnr - 1 )
		#func = lambda x,r : special.erfcinv( f*2/(Sigma*kappa) * Grazing_Angle(x,r) ).clip(0)*np.sqrt(2) - x
		func = lambda x,r : special.erfcinv( 2/(Sigma*kappa) *( - Grazing_Angle(x,r)*np.log(f) )).clip(0)*np.sqrt(2) - x
		h_ph = np.array([ optimize.fsolve( lambda x: func(x,r) , 4)[0] for r in r_list ] )
		#print(h_ph)
		if shade_in_thin_region:
			for i in range(len(h_ph)):
				if i!=0 and h_ph[i-1] * H[i-1] > h_ph[i] * H[i]:
					h_ph[i] = h_ph[i-1] * H[i-1]/H[i]
		
	else:
		h_ph = 4
	return h_ph


def calc_T(r_list,Mdot,alpha,Ls, mode="classic",i_sat=100,irr=True,acc=True,eff=0.1):
	r_au = r_list/cst.au
	Tvis_anal = cT.Tvis_anal(r_au, Mdot, Ms, kappa_R(200), alpha)
	Tacc_anal = cT.Tacc_anal(r_au, Mdot, Ms, eff)
	Tirr_anal = cT.Tirr_anal(r_au, Ls)

	if mode=='anal_acc':
		return Tacc_anal
	if mode=='anal_vis':
		return Tvis_anal
	if mode=='anal_irr':
		return Tirr_anal 
	if mode=='anal_new':
		Ta = ( Tacc_anal**4 + Tirr_anal**4 )**0.25 
		Ta[Tvis_anal>800]  = Tvis_anal[Tvis_anal>800]
		return Ta

	if mode=='classic' or mode=='new':
		T_0 = Tvis_anal 
		for i in range(i_sat):
			Tacc   = get_Tacc(r_list,T_0,Mdot,alpha)	if acc else 0 

			Sigma =  Mdot / (3*cst.pi*alpha*Cs(T_0)**2 / Omega_K(r_list)	)
#calc_Sigma_anal(r_list,Mdot,alpha,Ls,mode=None,eff=0.1)
			H = Cs(T_0)/Omega_K(r_list)
			hph = get_h_ph(r_list, H, Sigma  , kappa_R(T_0))

			Tirr   = get_Tirr(r_list,T_0,Mdot,alpha,Ls,h_ph=hph) if irr else 0 
			Told   = ( Tacc**4 + Tirr**4 )**0.25
			if np.all( np.abs(1 - Told/T_0) < 1e-4	):
				break
			T_0    = Told

		if mode=='classic':
			print(f"Mode: {mode} , Irr: {irr} , Acc: {acc} ; {i}")
			Convergence_Check(i)	
			return Told

	if	mode=='new':
		T_0 = Tacc_anal
		for j in range(i_sat):
			Tacc   = get_new_Tacc(r_list,T_0,Mdot,alpha,eff) if acc else 0
			Sigma =  Mdot / (3*cst.pi*alpha*Cs(T_0)**2 / Omega_K(r_list)	)
#calc_Sigma_anal(r_list,Mdot,alpha,Ls,mode=None,eff=0.1)
			H = Cs(T_0)/Omega_K(r_list)
			hph = get_h_ph(r_list, H, Sigma  , kappa_R(T_0))

			Tirr   = get_Tirr(r_list,T_0,Mdot,alpha,Ls,h_ph=hph) if irr else 0 
			Tnew   = ( Tacc**4 + Tirr**4 )**0.25
			if np.all( np.abs(1 - Tnew/T_0) < 1e-4	):
				break
			T_0    = Tnew
		else:
			pass
			exit()

		if Told is list or np.array:
			Tnew[ Told > 800 ]	= Told[ Told > 800 ]
		else:
			Tnew  = Told if  Told > 800  else Tnew

		print(f"Mode: {mode} , Irr: {irr} , Acc: {acc} ; {i} {j}")
		Convergence_Check(j)
		return Tnew

	print(f"I do not support this mode : {mode}")
	return


def Convergence_Check(i):
	if i>500:
		print("Not convergent")
#		exit()
	return

def calc_Sigma(r,T,Mdot,alpha):
	return Mdot / (3*cst.pi * alpha * Cs(T)**2 ) *Omega_K(r)

def get_cross_point(x,y,y0):
	return interp1d(y,x,fill_value='extrapolate')(y0)

## See Turner 2012

def make_empty_lists(n):
	return [ [] for i in range(n) ]


class Calc_time_evolution:
	def __init__(self, tlist, r, lum_mode=""	  ):
		self.tlist = tlist
		self.r	   = r
		self.SL = None
		self.Mdot_evol =None
		self.L_evol = None
		self.acc_mode = "Hartmann1988Anal"		

	def calc(self, T_args={}, acc_args={}):
		self.L_evol, self.Mdot_evol, SL = make_empty_lists(3)
		for t in self.tlist:## year
			Ls		  =  lumi_evol(t)
			md		  =  mdot_evol(t, mode= self.acc_mode, Lstar=Ls, **acc_args)
			Mdot	  =  md * cst.Msun/cst.yr
			T		  =  calc_T( self.r , Mdot , alpha, Ls, **T_args )
			self.L_evol.append( Ls )
			self.Mdot_evol.append( md )
			#print( T  )
			SL.append(func_a_snow(self.r,T))	
		return np.array(SL)/cst.au






#################################################################################################################


def set_physical_unit(disk):
	class Unit_Data():
		pass
	uni=Unit_Data()	
	uni.L	  = disk.H
	uni.V	  = disk.Cs
	uni.t	  = 1./disk.Omega
	uni.rho   = disk.rho_mid
	uni.Sigma = uni.rho * uni.L
	uni.P	  = uni.rho * uni.V*uni.V
	uni.nu	  = uni.L * uni.L /uni.t
	uni.eta   = uni.V**2 * uni.t
	uni.B	  = sqrt(4.0 * pi * uni.rho ) * uni.V
	uni.Q_H   = uni.eta / uni.B
	uni.Q_A   = uni.eta / uni.B / uni.B
	uni.flux  = uni.rho*uni.V**3
	return uni 

class Input_Data:
	def __init__(self,data):
		if isinstance(data, np.ndarray ):
			self.data = data
		elif isinstance(data, dict):
			self.dic = data
		else:
			print("Error in data type : ",type(data))
			exit()

	def inp(self,*nums):
#		print(nums)
		try:
			if len(nums)>=2:
				return [ self.data[:,i] for i in nums ]
			if len(nums)==1:
				return self.data[:,nums[0] ]
		except:
			print(f"failed to input data : index = {nums}")
			exit()
			return np.nan

	def get(self,name):
		try:
			return self.dic[name]
		except:
			print(f"failed to get data : name = {name}")
			return np.nan



## Set strings
#dn = f"/"
# hd_ss = "ss-"		# Header of Snapshot Figure
dn_ss_fig = "F_ss"
hd_hst = "hst-"  # Header of History Figure
ext = ".pdf"
fig_dn = f"{dn_ss_fig}"
os.makedirs(fig_dn, exist_ok=True)

class MyPlot:
	""" My plot functions """

	def __init__(self, x=None, xl=None, xlim=None, hd=None):
		self.opfn = ""
		self.dic = {}
		self.fig = plt.figure()
		self.vals_num = 0
		self.pm = False

		self.xlim = xlim
		self.xl   = xl
		self.x	  = x
		self.hd   = hd

		self.ls = None
		self.lw = None
		self.alp = None
		self.args_legend = {"loc": 1}
		self.pltfig = {}

	def tex_fmt(self, x, pos):
		a, b = '{:.0e}'.format(x).split('e')
		b = int(b)
		return r'${} \times 10^{{{}}}$'.format(a, b)

	def do_plot(self, value_cls_arr, opfn, c=[None], ls=[None], lw=[None], alp=[None],
				xl='', yl='', xlim=None, ylim=None, logx=False, logy=False, pm=False, leg=True, hl=[], vl=[], title='',fills=None,arrow=[],lbs=None,
				**args):
		if isinstance(value_cls_arr, list):
			if isinstance(value_cls_arr[0], np.ndarray): value_cls_arr = self.make_class_list_from_array_list(
				value_cls_arr)
		if isinstance(value_cls_arr, dict):					 value_cls_arr = self.make_class_list_from_dict(
			value_cls_arr)
		print("Plotting ", opfn)
		self.fig = plt.figure(**self.pltfig)

		self.ax = self.fig.add_subplot(111)
		num = len(value_cls_arr)
		my.exppand_array(num, c)
		my.exppand_array(num, ls)
		my.exppand_array(num, lw)
		my.exppand_array(num, alp)
		self.set_default(ls, self.ls)
		self.set_default(lw, self.lw)
		self.set_default(alp, self.alp)
		for i, V in enumerate(value_cls_arr):
			lb = lbs[i] if lbs else V.lab
			self.ax.plot(self.x, V.d, label=V.lab, c=c[i], ls=ls[i], lw=lw[i], alpha=alp[i], zorder=-i, **args)
			if pm:
				#				self.ax.plot( self.x , -V.d , label=V.lab , c=c_def_rgb_dark[i], ls=ls[i], lw=lw[i] , **args)
				self.ax.plot(self.x, -V.d, c=c_def_rgb_dark[i], ls=ls[i], lw=lw[i], **args)

			if fills!=None: 
				for fill in fills:
					if i == fill[0]:
						self.ax.fill_between(self.x,fill[1],fill[2],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
	
		for h in hl:
			plt.axhline(y=h)  
		for v in vl:
			plt.axvline(x=v)
		for i,ar in enumerate(arrow):
#			self.ax.arrow( *ar )
			print(lw[i],c[i])
			self.ax.annotate('', xy=(ar[0]+ar[2], ar[1]+ar[3]), xytext=(ar[0], ar[1]),
			arrowprops=dict(shrink=0, width=3, headwidth=8,headlength=10, connectionstyle='arc3', facecolor=c[i], edgecolor=c[i]),
			)			

		plt.title(title)
		if logx: plt.xscale("log")
		if logy: plt.yscale('log', nonposy='clip')
		if xlim is None: xlim = self.xlim
		plt.xlim(xlim)
		if ylim is None: ylim = plt.ylim()
		plt.ylim(ylim)
		# plt.yscale("log",nonposy="clip")
		plt.xlabel(self.xl)
		plt.ylabel(yl)
		if leg: plt.legend(**self.args_legend)

		plt.savefig(f"{fig_dn}/{opfn}{ext}", transparent=True, bbox_inches='tight')
		plt.close()


	def do_map(self, x , y , z , opfn, c=[None], ls=[None], lw=[None], alp=[None],
				xl='', yl='', cbl='',xlim=None, ylim=None, cblim=None, logx=False, logy=False, logcb=False, pm=False, leg=True, hl=[], vl=[], title='',fills=None,data="", 
				Vector=None,				
				**args):
		
		if data=="1+1D":
			yy, xx =  np.meshgrid(y[0] ,x)
		else:
			yy, xx =  np.meshgrid(y ,x)

		div = 10.0
		fig, ax = plt.subplots()

#		cmap=plt.get_cmap('cividis')
#		cmap=plt.get_cmap('magma')
		cmap=plt.get_cmap('inferno')
		if logcb:	
			log_cbmin = np.log10(cblim[0])
			log_cbmax = np.log10(cblim[1])	
			delta	 = ( log_cbmax	- log_cbmin  )/div
			interval = np.arange(log_cbmin, log_cbmax+delta, delta)
			vrange={ "vmin":cblim[0] , "vmax":cblim[1]	}
			img = ax.contourf( xx , yy, np.log10(np.abs(z)) , interval, vmin= log_cbmin, vmax= log_cbmax ,extend='both',cmap=cmap)
			ticks	 = np.arange( log_cbmin , log_cbmax +1, (log_cbmax - log_cbmin)/div )

		else:
			delta	 = (cblim[1] - cblim[0])/div
			interval = np.arange(cblim[0], cblim[1]+delta, delta)
			img = plt.contourf( xx , yy, z, interval, vmin=cblim[0], vmax=cblim[1], corner_mask=True, extend='both'  )
			ticks	 = np.arange( cblim[0] , cblim[1] +1,	(cblim[1] -  cblim[0])/div	   )

		if fills!=None:
			for fill in fills:
				i = fill[0]
				#ax.fill_between( x, y[0] ,fill[1],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
				ax.contourf( xx , yy, fill[1] , cmap=cmap , alpha=0.5)
		
		fig.colorbar(img, ax=ax,ticks=ticks , extend='both', label = cbl, pad=0.02)

		if Vector is not None:
			uu = Vector[0]
			vv = Vector[1]
			n = 18
			th_seed = np.linspace(0,np.pi/2, n	)
			rad  = x[-1] 
			seed_points = np.array( [ rad * np.sin(th_seed) , rad * np.cos(th_seed) ])
			plt.streamplot( x , y[0] , uu.T, vv.T , density = n/6 ,	linewidth=0.3*30/n, arrowsize=.3*30/n, color='w' , start_points=seed_points.T ) 

		plt.title(title)
		plt.gca().set_aspect('equal', adjustable='box')

		if logx: plt.xscale("log")
		if logy: plt.yscale('log', nonposy='clip')
		if xlim is None: xlim = self.xlim
		if ylim is None: ylim = fig.ylim()
		plt.xlim(xlim)
		plt.ylim(ylim)
#		plt.cblim(cblim)
		plt.xlabel(self.xl)
		plt.ylabel(yl)
		if leg: plt.legend(**self.args_legend)

		fig.savefig(f"{fig_dn}/{opfn}{ext}", transparent=True, bbox_inches='tight')
		plt.close()




	def make_class_list_from_array_list(self, arrs):
		cls_list = []
		for arr in arrs:
			cls_list.append(Value('', arr))
		return cls_list

	def make_class_list_from_dict(self, dic):
		cls_list = []
		for k , v in dic.items():
			cls_list.append(  Value(k,v)  )

		return cls_list

	def set_default(self,arg,v_def):
		for i,a in enumerate(arg):
			if a is None:
				arg[i] = v_def

def hline(value,like=None):
	return np.array([value]*len(like))


class Value:
	def __init__(self, lab, v_array):
		self.lab = lab
		self.d = v_array

	def return_array(self, n):
		return self.t[n]

	def dot(self, X, Y):
		return

	#	def __add__(self, lab, other ):
	#		return Value( lab , array( map( lambda x, y: x + y	,	self.t , other.t  )))

	def __mul__(self, other):
		return np.vectorize(lambda x, y: x * y)(self.d, other.d)

	def set(self, lab, array):
		return Value(lab, array)
##########################################################################################################################

###########
if __name__ == '__main__':
	main()
###########


