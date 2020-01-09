import numpy as np
import cst
from scipy import integrate
from scipy.integrate import ode,odeint,solve_ivp
import matplotlib.pyplot as plt
from copy import copy,deepcopy
from scipy.interpolate import interp1d
from scipy import optimize

plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = 'black'
plt.rcParams["legend.handlelength"] = 0.5


## Initial Setting for model
mode = 'TSC'


## make velocity and density structure 

# Isothermal sphere 

## unpertubed state


a = 0.35e5 # cm/s : sound speed
rho_0 = lambda r : a**2 / (2 * pi * cst.G * r**2)

R = 5.4e16 #--> 1Msun
M = 2 * a**2 * R / cst.G


# Rigid rotation
Omega = 1e-14  ## constant

f_rho0 = None


def main():
	Solve_Spherically_Symmetric_Solution()
	Solve_Pertubed_Solution()





def Solve_Spherically_Symmetric_Solution():
# Solve Spherically symmetric solutions
# unpertubed rotating equilibria
	global f_rho0

	def f(x, y):
		ret = np.zeros_like(y)
		ret[0] = y[1]
		ret[1] = - 2*y[1]/x - 2*np.exp(y[0])/x**2 + 2 * (1+x**2)/x**2
		return ret

	ksi = np.logspace( -2 , 2, 200	)
	y0 = [ 0 , 0 ]
	sol = solve_ivp( f , (ksi[0],ksi[-1]) , y0 , t_eval=ksi )
	Phi = sol.y[0]
	pl = Plot() 
	M = [ ksi[0] + integrate.simps(		np.exp( Phi )[:i] , ksi[:i]	) for i in	range(1,1+len(ksi))  ] 
	pl.line(ksi,  [ np.exp( Phi )/ksi**2 , M ] , 'fig1.pdf' , xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',logx=True,logy=True)
	f_rho0 = fnize( ksi, np.exp( Phi )/ksi**2 , logy=True ) 


def Solve_Pertubed_Solution():
	
	def f0(x, y):
		ret = np.zeros_like(y)
		V0	= y[0]
		al0 = y[1]
		ret[0] =	 (al0*(x - V0) - 2/x)*(x-V0)/((x-V0)**2 -1)
		ret[1] = al0*(al0 - 2/x*(x - V0))*(x-V0)/((x-V0)**2 -1)
		return ret

	def f0_i(x, y):
		ret = np.zeros_like(y)
		al0 = y[0]
		V0	= y[1]
		ret[0] = al0*(al0 - 2/x*(x - V0))*(x-V0)/((x-V0)**2 -1)
		ret[1] =	 (al0*(x - V0) - 2/x)*(x-V0)/((x-V0)**2 -1)
		return ret

	def fQ( x , y , K=0 , fun_al0 = None , fun_V0=None ,fun_dal0dx = None , fun_dV0dx =None ): ## for ode
		ret = np.zeros_like(y)
		al0    = 2/x**2	  if ( fun_al0	is None	 ) else fun_al0( x )
		V0	   =	0	  if ( fun_V0		is None )	  else fun_V0( x )
		dal0dx = -4/x**3  if ( fun_dal0dx is None ) else fun_dal0dx( x )
		dV0dx  = 0		  if ( fun_dV0dx	is None ) else fun_dV0dx( x )
		al	   = y[0]
		V	   = y[1]
		W	   = y[2]
		Q	   = y[3] 
		P	   = y[4]
		Psi    = -x**2/3 - 1/x**3 * Q - x**2 * P
		dPsidx =  -2/3*x + 3*Q/x**4 - 2*x*P
		m = x**2 * al0 * (x - V0)
		A = al/x**2 * (2*x*V0 + x**2*dV0dx ) + V/x**2 * (x**2*dal0dx + 2*x*al0) - 6*al0*W/x
		B = -al/al0**2*dal0dx  + ( 2 + dV0dx )*V + ( dPsidx + 2/3*(m/2)**4/x**3 )
#		B = -al/al0**2*dal0dx  + ( 2 + dV0dx )*V + ( dPsidx + 2/3*(m/2)**4/x**3 )
			
#		print("A:", al/x**2 * (2*x*V0 + x**2*dV0dx ) , V/x**2 * (x**2*dal0dx + 2*x*al0) , - 6/x*al0*W)
#		print("B:",-al/al0**2*dal0dx  , ( 2 + dV0dx )*V , ( dPsidx + 2/3*(m/2)**4/x**3 ))
		ret[0] = 1/( (x-V0)**2 -1  ) * ((x-V0)*A + al0*B  )	
#		ret[1] = 1/( (x-V0)**2 -1	) * ((x-V0)*B + 1/al0*A)
		ret[1] = 1/( (x-V0)**2 -1	) * ((x-V0)*B + 1/al0/2*A)
		ret[2] = 1/x/(x-V0) * ( ( 2*x+V0   )*W + al/al0 + (Psi+(m/2)**4/(3*x**2)) )
		ret[3] =  0.2*x**4*al
		ret[4] = -0.2*al/x
#		print( x, Q , y[3] , 0.2*x**4*al)
		return ret

	def fQ_out(y, x):
		ret = np.zeros_like(y)
		al = y[0]
		V  = y[1]
		W  = y[2]
		Q  = y[3]
		P  = y[4]		
		ret[0] = 1/( x**2 -1  ) * ( -12/x**2*W + 2/x**2*( x*al + 2*V + 3*Q/x**4 - 2*x*P	)	)
		print( "fQout:",-12/x**2*W ,  2/x**2* x*al ,  2/x**2* 2*V , 2/x**2* 3*Q/x**4 , -2/x**2* 2*x*P )    

		ret[1] = 1/( x**2 -1  ) * ( x*(x*al + 2*V + 3*Q/x**4 - 2*x*P) -6/x*W )
		ret[2] = 2/x*W + 0.5*al - Q/x**5 - P
		ret[3] = 0.2*x**4*al
		ret[4] = -0.2*al/x
		return ret

	t = 1e-2
	x = np.logspace( np.log10(1/t) , np.log10(t**2) , 10000	) ## determines size of MC 

	y0 = [ 0 , 2/x[0]**2 ]
	sol = solve_ivp( f0 , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True, method='BDF',rtol = 1e-12,	atol = [1e-8,1e-15], vectorized=True)
#	sol = solve_ivp( f0 , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True, method='LSODA',rtol = 1e-12,  atol = 1e-14, vectorized=True)

	#sol = solve_ivp( f0 , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True, method='RK45',	rtol = 1e-13,  atol = 1e-10, vectorized=True)

	V0	= sol.y[0]
	al0 = sol.y[1]
	dV0dx = np.gradient(V0, x)

	dal0dx = np.gradient(al0, x)

	pl = Plot()
	s = { "logx" : True, "logy" :True }
	pl.line( sol.t ,  np.abs(sol.y) , 'test_f0_out.pdf' , xl=r'log y', yl=r'log $\rho$[$\Omega^2/2\pi G$]',lbs=[r"$V_{0}$", r"$\alpha_{0}$"],xlim=[1e-4,1e3],ylim=[1e-12,1e12],**s)
	pl.line(x, 0.25*( x**2*al0*(x-V0) )**2 , 'fig2_specificAM.pdf' , xl=r'log y', yl=r'$(m_0/2)^2$',**s)
	m0 = 0.975
	rho = (m0/2/x**3)**0.5/2/t**2
	pl.line(x, [ al0/2/t**2 , f_rho0( x*t )*2 , rho ] , 'rho.pdf' , xl=r'log y', yl=r'$rho$',lbs=["Shu's 0th order sol.","Unpertubed state","Inner expansion-wave sol."],**s)	

#	f_V0  = lambda x : sol.sol(x)[0]			
#	f_al0 = lambda x : sol.sol(x)[1].clip(0)
#	f_dal0dx = fnize( x, dal0dx )			   
#	f_dV0dx  = fnize( x, -dV0dx )		
	f_al0 = fnize( x, al0 , logy=True ) 
#	f_V0 = lambda x : sol.sol(x)[0]  #fnize( x, -V0 , neg=True)	 
	f_V0 = fnize( x, V0 )  
	f_dal0dx = fnize( x, dal0dx  ) 
	f_dV0dx  = fnize( x, dV0dx	 )

	print(f_al0(x),f_V0(x), f_dal0dx(x), f_dV0dx(x) )

#	x = np.logspace( 3 , -1, 2000	)
	#x = np.logspace( np.log10(1/t) , np.log10(t**2) , 1000	) ## determines size of MC 

#	K  = -11244.125560319997
#	K0 = 
#	Qin0 = -3 
#	for i in range(100):
#		print(K)	
#		K = 5e5
#	x_inf = x[0]
#	y0 = np.array([ -2/7*x_inf**(-7) , -1/2*x_inf**(-4) , 1/6*x_inf**(-4), 1 + 1/35*x_inf**(-2) , -2/245*x_inf**(-7) ])*K
	i=0
	def procedure_for_K( K ):
		nonlocal i
		i += 1	
		y0 = [ 0, 0, 0, K , 0]
		sol2 = solve_ivp( lambda y,x: fQ(y,x,K=K) , (x[0],x[-1]) , y0 , t_eval=x , dense_output=True)
#		sol2 = solve_ivp( fQ , (x[0],1) , y0  , method='RK45',rtol = 1e-3,	atol = 1e-6, vectorized=True)
		pl = Plot()
		pl.line( sol2.t , np.abs(sol2.y) , f'test_fQ_ewol_{i}.pdf' ,
			xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-6,1e3],ylim=[1e-20,1e15],lbs=["al","V","W","Q","P"]*2 ,
			lss = [":"]*5+["-"]*5, **s)


		sol3 = solve_ivp( lambda yy , xx: fQ( yy , xx , K=K, fun_al0 =f_al0 , fun_V0=f_V0 ,fun_dal0dx =f_dal0dx , fun_dV0dx =f_dV0dx ) , (x[0],x[-1]) , y0 , t_eval=x ,
			method='BDF',rtol = 1e-8,	atol = 1e-12, vectorized=True)

		l = max( len(sol2.t) , len(sol3.t) ) 
		print( np.abs(sol2.y)[:l].shape , np.abs(sol3.y)[:l].shape )

		pl.line( sol3.t[:l] , np.concatenate(  [ np.abs(sol2.y)[:l]	, np.abs(sol3.y)[:l] ] ) , f'test_fQ_{i}.pdf' ,
			xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-6,1e3],ylim=[1e-20,1e15],lbs=["al","V","W","Q","P"]*2 ,lss = [":"]*5+["-"]*5, **s)
		Qin = sol3.y[3][-1]
		print( f"{i}: {K} --> {Qin}")	
		return Qin

#	K_sol = optimize.newton( procedure_for_K , -0.001596049224825923 , tol = 1e-10 )
#	print( optimize.newton( procedure_for_K ,  -0.0012002377673691503 , tol = 1e-10 ) )
#	print( optimize.bisect( procedure_for_K ,  -1e3  ,	0 ) )

	K_sol = -0.001596049224825923
	y0 = [ 0, 0, 0, K_sol , 0]
	sol3 = solve_ivp( lambda yy , xx: fQ( yy , xx , K=K_sol, fun_al0 =f_al0 , fun_V0=f_V0 ,fun_dal0dx =f_dal0dx , fun_dV0dx =f_dV0dx ) , (x[0],x[-1]) , y0 , t_eval=x ,
			method='RK45',rtol = 1e-6,	atol = 1e-8, vectorized=True)

	pl = Plot()
	al, V, W, Q, P = sol3.y
	pl.line( sol3.t , (al, V, -W, -Q, -P)  , f'test_fQ_res.pdf' ,
			xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-6,1e3],ylim=[1e-20,1e8],lbs=["al","V","-W","-Q","P"]*2 ,**s)

	exit()
# -11299.74178082879 --> -21.22070249201574
# -11300.404561744872 
#		if i!=0 and Qin != Qin0:
#		print("Before",K)
#		if i!=0:
#			print( "a", K ,  )
#			K	+=	- Qin * ( K - K0)/(Qin - Qin0 )
#			print(	 "b",K, Qin * (K - K0)/(Qin - Qin0 )  )
#		else:
#			K -= 1
#		print("After",K)

#		K0 = deepcopy(K)
#		Qin0 = deepcopy(Qin)

#		if sol2.y[3][-1] > 0:
#			K -= 1
#		else:
#			K += 0.01

   #	 K0 = K
  #		 Qin0 = Qin
	
#	print(	 np.concatenate(  [ np.abs(sol.y.T)[:l]   , np.abs(sol2.y.T)[:l] ] )	 )

#	pl.line( sol2.t[:l] , ( np.abs(sol.y.T)[:l]	, np.abs(sol2.y.T)[:l] ) , 'test_fQ.pdf' , xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-1,1e3],ylim=[1e-25,1e8],lbs=[ '+EWS'  ,'Q-Flow'],**s)
#	pl.line( (sol.t, np.abs(sol.y.T) ) , ( sol2.t	 , np.abs(sol2.y.T) ) , 'test_fQ.pdf' , xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-1,1e3],ylim=[1e-25,1e8],lbs=["ivp",'ivp2'],**s#)

#	pl.line(x[:len(y)], alQ + 2*VQ - 6*WQ + 3*Q_ - 2*P_ , 'test_fQ_2.pdf' , xl=r'log r[a/$\Omega$]', yl=r'',**s)

	exit()



	##
#	x = np.logspace( 2 , 0, 200  )
#	x_inf = x[0]
	y0 = np.array([ -2/7*x_inf**(-7) , -1/2*x_inf**(-4) , 1/6*x_inf**(-4), 1 + 1/35*x_inf**(-2) , -2/245*x_inf**(-7) ])*K
	y = odeint( fQ_out, y0, x )
	pl = Plot()
	pl.line(x, np.abs(y) , 'test_fQ_out.pdf' , xl=r'log r[a/$\Omega$]', yl=r'log $\rho$[$\Omega^2/2\pi G$]',xlim=[1e-1,1e3],ylim=[1e-20,1e3],**s)








def cum_int( y , x , ini=0	):
	return np.array(	[ ini + integrate.simps( y[:i] , x[:i] ) for i in  range(1,1+len(x)) ] )

def cum_int_rev( y , x , ini=0	):
	return np.array(	[ ini + integrate.simps( y[:i] , x[:i] ) for i in  range( 1+len(x), 1  , -1) ] )

def fnize( x , y , logy=False):
	if logy :
		log_f = interp1d( np.log( x ), np.log( y ) , fill_value='extrapolate')
		return lambda xx : np.exp( log_f( np.log(xx) ) ) 
	else:
		f = interp1d( np.log( x ) ,  y	, fill_value='extrapolate')
		return lambda xx : f( np.log(xx) ) 

class Plot:
	logx0 = False
	def __init__(self,xl=None,yl=None,logx=False,logy=False):
		global logx0
		self.xl00 = xl
		self.yl0 = yl
		self.logx0 = logx
		self.logy0 = logy
		logx0 = True	

#	locals().items()

	def ini(self, *args ):
		print(locals())
		for a in args:
			if a is None:
				a = self.a


	def line(self, x , ys , fname , \
				xl=None, yl=None, logx=None, logy=None,xlim=None,ylim=None,lbs=None,lss=None):
		plt.figure()
		if len(x) == len(ys):
			plt.plot( x, ys)
		else:
			for i,y in enumerate(ys) :
				lb = lbs[i] if lbs else None
				ls = lss[i] if lss else None
				plt.plot(x,y,label=lb,linestyle=ls)
		if lbs: plt.legend()
		if xl: plt.xlabel(xl)
		if yl: plt.ylabel(yl)
		if xlim: plt.xlim(xlim)
		if ylim: plt.ylim(ylim)
		if logx: plt.xscale('log') 
		if logy: plt.yscale('log')
		plt.savefig(fname)
		


main()




