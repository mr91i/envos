#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d,griddata
pyver = sys.version_info[0] + 0.1*sys.version_info[1]
#print("Message from %s"% os.path.dirname(os.path.abspath(__file__)))
def msg(s):
	print("[plotter.py] %s"%s)
msg("%s is used."% os.path.abspath(__file__))
msg("This is python %s"%pyver)
dn_this_file = os.path.dirname(os.path.abspath(__file__))
plt.switch_backend('agg')

dbg = 1

class say:
	def __init__( self, *something ):
		if dbg:
			print( something )
			pass
	def exit(self):
		exit()
class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

class Default_params:
	def __init__(self):
		# For saving figures
		self.hd	   = ""
		self.ext	= ".pdf"
		self.fig_dn = os.path.abspath( dn_this_file + "/../fig" )
		if pyver >= 3.2:
			os.makedirs(self.fig_dn, exist_ok=True )
		else:
			if not os.path.exists(self.fig_dn):
				os.makedirs(self.fig_dn)

		# Default values
		self.x	  = None
		self.xlim  = None
		self.xl	  = None
		self.c	= None
		self.ls	= None
		self.lw	= None
		self.alp = None
		self.pm	= False
		self.args_leg = {"loc": 'best'}
		self.args_fig = {}

defs = Default_params()

def set( **kws ):
	global defs
	defs.__dict__.update( kws )
	
def Update( **kws ):
	inp = {}
	for kw in kws.keys():
		inp["g_"+kw]=kws.pop(kw)
	globals().update( inp )

def tex_fmt( x, pos):
	a, b = '{:.0e}'.format(x).split('e')
	b = int(b)
	return r'${} \times 10^{{{}}}$'.format(a, b)

#
# For Preprocessing
#
def give_def_vals( vals ,  defvals ):
	for i, v in enumerate(vals):

		if ( not isinstance(v,(list,np.ndarray)) ) and (v == None or v== '' ) :
			vals[i] = defvals[i]
	return vals

def set_options( n , opts , defopts ):
	for i , opt in enumerate(opts):
		if isinstance( opt , list ):
			opts[i] = expand_list( n , opt )
			opts[i] = set_default_to_None( opt , defopts[i] )
		elif isinstance( opt , (float,str) ):
			opts[i] = [ deopts[i] ]*n

def expand_list( n , l ):
	if len( l ) < n :
		l += [None] * (n - len( l ))
	return l

def set_default_to_None( opt , defopt ):
	if isinstance( opt , list ):
		return [ ( defopt if a == None else a ) for a in opt ]

def conv_list_to_dict( l ):
	if isinstance( l[0] , (int,float) ):
		return { '' : l }
	elif isinstance( l[0] , (list,tuple) ):	
		return { i: v for i,v in enumerate(l) }
	else:
		exit()

def reproduce_map( v , x1_ax , x2_ax , X1_ax , X2_ax  ):
# Example : den = reproduce_map( re['den'] , re['r_ax'] , re['th_ax'] , rr , tt  )
	f = interp2d( x1_ax  ,	x2_ax  ,  v ,  )
	fv = np.vectorize( f )
	v_new = 10**fv(  X1_ax ,  X2_ax  )
	return v_new


def reproduce_map2( v , x1_mg , x2_mg , X1_ax , X2_ax  ):
# Example : den = reproduce_map( re['den'] , re['r_ax'] , re['th_ax'] , rr , tt  )
	f = interp2d( x1_mg  ,	x2_mg ,  v , fill_value = np.nan )
#	f = interp2d( x1_mg.flatten()  ,  x2_mg.flatten()  ,  v.flatten() , fill_value = np.nan )
	fv = np.vectorize( f )
	v_new = 10**fv(  X1_ax ,  X2_ax  )
	return v_new
#
# Plot function : draw lines 
#
def plot( y_dic , opfn, x = None , c=[None], ls=[], lw=[], alp=[], leg=True, frg_leg=0,  title='', 
			xl='', yl='', xlim=None, ylim=None, logx=False, logy=False, loglog=False,
			pm=False, hl=[], vl=[], fills=None, arrow=[], lbs=None,
			*args, **kwargs):

	settings=locals()

	# Start Plotting
	msg("Plotting %s"%opfn)
	fig = plt.figure( **defs.args_fig )
	ax	= fig.add_subplot(111)

	# Preprocesing
	if isinstance( y_dic , (list,tuple) ):
		y_dic = conv_list_to_dict( y_dic )
	say( 'Inputs to give_def_vals : ' ,  [x,xlim,xl] ,	[defs.x,defs.xlim,defs.xl]	)
	x,xlim,xl = give_def_vals( [x,xlim,xl] ,  [defs.x,defs.xlim,defs.xl] )
	say( 'Inputs to give_def_vals : ' ,  [x,xlim,xl] )

	set_options( len(y_dic) , [c,ls,lw,alp] ,  [defs.c,defs.ls,defs.lw,defs.alp]  )	


	# Main plot
	for i, (k, y) in enumerate( y_dic.items() ):
		# Preprocessing for each plot
		lb	   = lbs[i]  if lbs else k
		pmplot = (x, -y) if pm	else ()	
		xy = (x , y) if (x is not None) else (y,)
	
		# Plot 
		say( 'Inputs to plot',xy ) 

		ax.plot( *xy , label=lb , c=c[i], ls=ls[i], lw=lw[i], alpha=alp[i], zorder=-i, **kwargs)
		if pm:
			ax.plot( *pmplot , label=lb , c=c[i], ls=ls[i], lw=lw[i], alpha=alp[i], zorder=-i, **kwargs)


		# Enroll flags
		if lb and frg_leg==0: 
			frg_leg = 1

	# Fill
	if fills: 
		for fil in fills:
			i = fil[0]
			ax.fill_between(x,fil[1],fil[2], facecolor=c_def_rgb[i] , alpha=0.2 , zorder=-i-0.5 )

	# Horizontal lines
	for h in hl:
		plt.axhline(y=h)  

	# Vertical lines
	for v in vl:
		plt.axvline(x=v,alpha=0.5)

	# Arrow
	for i, ar in enumerate(arrow):
		ax.annotate('', xytext=(ar[0], ar[1]) , xy=( ar[2] , ar[3] ), xycoords='data',annotation_clip =False ,
					arrowprops=dict(shrink=0, width=3, headwidth=8,headlength=8, 
									connectionstyle='arc3', facecolor=c[i], edgecolor=c[i])
					)			

	# Postprocessing
	plt.title(title)
	
	if leg and frg_leg==1:	plt.legend(**defs.args_leg)
	if logx or loglog: plt.xscale("log")
	if logy or loglog: plt.yscale('log', nonposy='clip')
	if xlim: plt.xlim(xlim)
	if ylim: plt.ylim(ylim)
	if xl:	 plt.xlabel(xl)
	if yl:	 plt.ylabel(yl)

	# Saving
	savefile = "%s/%s%s"%(defs.fig_dn,opfn,defs.ext)
	fig.savefig(savefile, transparent=True, bbox_inches='tight')
	msg("	Saved %s to %s"%(opfn,savefile))
	plt.close()

	return Struct(**settings)


#
# Plot function : draw a map
#
#	cmap=plt.get_cmap('cividis')
#	cmap=plt.get_cmap('magma')
def map( x , y , z , opfn, c=[None], ls=[None], lw=[None], alp=[None],
			xl='', yl='', cbl='',xlim=None, ylim=None, cblim=None, logx=False, logy=False, logcb=False, loglog=False,
			pm=False, leg=False, hl=[], vl=[], title='',fills=None,data="", Vector=None, div=10.0, n_sline=18,hist=False,		
			square=True,
			**args):

	# Start Plotting
	msg("Plotting %s"%opfn)
	fig = plt.figure( **defs.args_fig )
	ax	= fig.add_subplot(111)
	cmap=plt.get_cmap('inferno')

	y = y[0] if data=="1+1D" else y
	if not isinstance(x[0],(list,np.ndarray)): 
		xx , yy =  np.meshgrid( x, y ,	indexing='ij' )
	else:
		xx = x
		yy = y

	if xlim==None:
		xlim = [ np.min(x) , np.max(x) ]
	if ylim==None:
		ylim = [ np.min(y) , np.max(y) ]
	if cblim==None:
		cblim = [ np.min(z[z>0]) , np.max(z) ]
		say(z, cblim)

	if logcb:
		cbmin , cbmax = np.log10(cblim)
	else:
		cbmin , cbmax = cblim

	delta = ( cbmax  - cbmin  )/div
	z	= np.log10(np.abs(z)) if logcb else z 

	if (cbmax+delta - cbmin)/delta > 100:
		print("Wrong!")
		exit()
	interval = np.arange( cbmin , cbmax+delta , delta )

#	print(xx.shape,yy.shape,z.shape,interval)
#	img = ax.contourf( z )	
#	img = ax.contourf( z , interval , vmin=cbmin, vmax=cbmax , extend='both' , cmap=cmap )
	img = ax.contourf( xx , yy , z , interval , vmin=cbmin, vmax=cbmax , extend='both' , cmap=cmap )
	say("OK!")
#	print( cbmin , cbmax +1 , delta  )
	ticks	 = np.arange( cbmin , cbmax+delta , delta ) # not +1 to cb_max
	fig.colorbar(img, ax=ax,ticks=ticks , extend='both', label = cbl, pad=0.02)
	say("OK!!")

	if fills is not None:
		for fill in fills:
			i = fill[0]
			#ax.fill_between( x, y[0] ,fill[1],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
			ax.contourf( xx , yy, fill[1] , cmap=cmap , alpha=0.5)
	
	if Vector is not None:
		uu = Vector[0]
		vv = Vector[1]
		th_seed = np.linspace(0 , np.pi/2 , n_sline	)
		rad  = np.max(xlim)
	
		seed_points = np.array( [ rad * np.sin(th_seed) , rad * np.cos(th_seed) ]).T
#		seed_points = None
	
		quiver=0
		if quiver:
			l = np.sqrt(uu**2+vv**2)
			n = 8
			m = 8
			plt.quiver( xx[::n,::m], yy[::n,::m] , (uu/l)[::n,::m], (vv/l)[::n,::m], angles='xy', color='red' )

		
		x_ax = np.linspace( xlim[0] , xlim[1] , 200	)
		y_ax = np.linspace( ylim[0] , ylim[1] , 200	)

#		x_ax = np.linspace( np.min(x) , np.max(x) , 400  )
#		y_ax = np.linspace( np.min(x) , np.max(x) , 400  )

		xx_gr , yy_gr = np.meshgrid( x_ax , y_ax , indexing='xy')
		xy = np.array([ [x,y] for x , y in zip(xx.flatten(),yy.flatten())  ])
		mth = 'linear'
		uu_gr = griddata(xy, uu.flatten(), (xx_gr, yy_gr) ,method=mth)
		vv_gr = griddata(xy, vv.flatten(), (xx_gr, yy_gr) ,method=mth)

		plt.streamplot( xx_gr , yy_gr , uu_gr , vv_gr , density = n_sline/4 , linewidth=0.3*30/n_sline, arrowsize=.3*30/n_sline, color='w'	, start_points=seed_points)
		
		
		



	plt.title(title)
	if square:
		plt.gca().set_aspect('equal', adjustable='box')

	if logx or loglog: plt.xscale("log")
	if logy or loglog: plt.yscale('log', nonposy='clip')
	if xlim: plt.xlim(xlim)
	if ylim: plt.ylim(ylim)
	if xl : plt.xlabel(xl)
	if yl: plt.ylabel(yl)
	if leg: plt.legend(**defs.args_leg)

	savefile = "%s/%s%s"%(defs.fig_dn,opfn,defs.ext)
	fig.savefig(savefile, transparent=True, bbox_inches='tight')
	msg("	Saved %s to %s"%(opfn,savefile))
	plt.close()

