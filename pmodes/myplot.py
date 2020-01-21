#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.interpolate import interp2d, griddata
pyver = sys.version_info[0] + 0.1*sys.version_info[1]
#print("Message from %s"% os.path.dirname(os.path.abspath(__file__)))

debug = 0
def msg(*s, **args):
	if "debug" in args:
		if args["debug"]==True:
			print("[debug] ", end='')
		else:
			return
	else:
		print("[plotter.py] ", end='')
	print(*s)
	if ("exit" in args) and (args["exit"]==True):
		exit()



msg("%s is used."% os.path.abspath(__file__))
msg("This is python %s"%pyver)
dn_this_file = os.path.dirname(os.path.abspath(__file__))

class Struct:
	def __init__(self, **entries):
		self.__dict__.update(entries)

class Default_params:
	def __init__(self):
		# For saving figures
		self.hd	   = ""
		self.ext	= ".pdf"
		self.fig_dn = os.path.abspath(dn_this_file) #os.path.abspath( dn_this_file + "/../fig" )
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

#defs = Default_params()

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

class Plotter:
#	def_logx = False
#	def_logy = False
	def __init__(self, fig_dir_path, x=None, xlim=None, xl='',
				 c=[], ls=[], lw=[], alp=[], pm=False,
				 logx=False, logy=False, leg=False,
				 args_leg={"loc":'best'}, args_fig={}):

		# For saving figures
		self.hd = ""
		self.ext = ".pdf"
		self.fig_dn = os.path.abspath(fig_dir_path) #os.path.abspath( dn_this_file + "/../fig" )
		self._mkdir(self.fig_dn)

		# Default values
		self.x = x
		self.xlim = xlim
		self.xl = xl
		self.c = c
		self.ls = ls
		self.lw = lw
		self.alp = alp
		self.pm = pm
		self.leg = leg
		self.args_leg = args_leg
		self.args_fig = args_fig
		self.fig = None
		self.logx = logx
		self.logy = logy
#		set_default(logx=logx)
		
#	def set_default(**args):
#		for k,v in args.items():		
#			setattr(self, k, v)

	@staticmethod
	def _mkdir(dpath):
		if pyver >= 3.2:
			os.makedirs(dpath, exist_ok=True)
		else:
			if not os.path.exists(dpath):
				os.makedirs(dpath)

	def reform_ylist(self, l):

		def checktype(a, types):
			if len(a) == len(types):
				return all( [ isinstance(x, t) for x, t in zip(a, types)] )
			else:
				return False

		def isvector(x):
			return not np.isscalar(x)

		def check_vector_type(vec):
			## [k, x, y]
			if len(vec) == 3 \
			   and isinstance(vec[0], str) \
			   and isvector(vec[1]) \
			   and isvector(vec[2]) \
			   and len(vec[1]) == len(vec[2]):
				return 1

			## [k, y]
			elif len(vec) == 2 \
			   and isinstance(vec[0], str) \
			   and isvector(vec[1]) \
			   and (len(self.x) == len(vec[1])):
				return 2

			## [x, y]
			elif len(vec) == 2 \
			   and isvector(vec[0]) \
			   and isvector(vec[1]) \
			   and (len(vec[0]) == len(vec[1])):
				return 3

			## y
			elif len(vec) == len(self.x) \
			   and isvector(vec):
				return 4

			else:
				return -1

		if isvector(l):
			itype = check_vector_type(l)
			print(itype)
			if itype == 1:
				return [l]

			elif itype == 2:
				return [[l[0], self.x, l[1]]]

			elif itype == 3:
				return [[0, l[0], l[1]]]
	
			elif itype == 4:
				return [[0, self.x, l]]
			else:

				jtype = check_vector_type(l[0])
				print(jtype)
				if jtype == 1:
					return l
	
				elif jtype == 2:
					return [[k, self.x, y] for k, y in l]
	
				elif jtype == 3:
					return [[i, x, y] for i, x, y in enumerate(l)]
	
				elif jtype == 4:
					return [[0, self.x, l]]

			raise Exception("Unknown type")
	
#		elif isinstance(l, dict):
#			return [[k, v] for k, v in l.items#()]

#		elif isinstance(l[0], (int,float)):
#			return ['', l]
	
#		else:
#			print("Unknown type:", l)
#			exit()
	#
	# Plot function : draw lines 
	#
	# usage: 
	# plot(y, "name")
	# plot([y1, y2, ..., yN], "test", x=x)
	# plot([leg1, y1], [leg2, y2], ..., [legN, yN] ], "test")
	# plot([leg1, x1, y1], [leg2, x2, y2], ..., [legN, xN, yN] ], "test")
	#
	def plot(self, y_list, opfn, x=None, c=[None], 
			 ls=[], lw=[], alp=[], leg=True, frg_leg=0, title='', 
			 xl='', yl='', xlim=None, ylim=None, 
			 logx=False, logy=False, loglog=False, pm=False, 
			 hl=[], vl=[], fills=None, arrow=[], lbs=None,
			 datatype="",
			 *args, **kwargs):

		settings=locals()
	
		# Start Plotting
		msg("Plotting %s"%opfn)
		fig = plt.figure( **self.args_fig )
		ax = fig.add_subplot(111)
	
		# Preprocesing
		data = self.reform_ylist(y_list)

#		set_options( len(y_list) , [c,ls,lw,alp] ,	[defs.c,defs.ls,defs.lw,defs.alp]  )	
	
		# Main plot
		for i, (k, x, y) in enumerate( data ):
			# Preprocessing for each plot
			lb = lbs[i] if lbs else k
			xy = (x, y) if (x is not None) else (y,)
			c = c[i] if len(c) >= i+1 else None
			ls = ls[i] if len(ls) >= i+1 else None	
			lw = lw[i] if len(lw) >= i+1 else None			
			alp = alp[i] if len(alp) >= i+1 else None

			# Plot 
			msg('Inputs to plot', xy, debug=1) 
			ax.plot( *xy, label=lbs, c=c[i], ls=ls[i], lw=lw[i], alpha=alp[i], zorder=-i, **kwargs)
			if pm:
				ax.plot( x, -y, label=lb , c=c[i], ls=ls[i], lw=lw[i], alpha=alp[i], zorder=-i, **kwargs)
	
			# Enroll flags
			if lb and frg_leg==0: 
				frg_leg = 1
	
		# Fill
		if fills: 
			for fil in fills:
				i = fil[0]
				ax.fill_between(x, fil[1], fil[2], facecolor=c_def_rgb[i], alpha=0.2, zorder=-i-0.5)
	
		# Horizontal lines
		for h in hl:
			plt.axhline(y=h)  
	
		# Vertical lines
		for v in vl:
			plt.axvline(x=v, alpha=0.5)
	
		# Arrow
		for i, ar in enumerate(arrow):
			ax.annotate('', xytext=(ar[0], ar[1]), xy=( ar[2] , ar[3] ), xycoords='data', annotation_clip=False,
						arrowprops=dict(shrink=0, width=3, headwidth=8, headlength=8,
										connectionstyle='arc3', facecolor=c[i], edgecolor=c[i])
						)			
	
		# Postprocessing
		plt.title(title)
		
		if leg and frg_leg==1:
			plt.legend(**self.args_leg)
	
		if logx or loglog:
			plt.xscale("log")
	
		if logy or loglog:
			plt.yscale('log', nonposy='clip')
	
		if xlim is None:
			xlim = self.xlim
		if xlim:
			plt.xlim(xlim or self.xlim)
	
		if ylim:
			plt.ylim(ylim)
	
		if xl is None:
			xl = self.xl
		if xl:
			plt.xlabel(xl)
	
		if yl:
			plt.ylabel(yl)
	
		# Saving
		savefile = "%s/%s%s"%(self.fig_dn,opfn,self.ext)
		fig.savefig(savefile, transparent=True, bbox_inches='tight')
		msg("	Saved %s to %s"%(opfn,savefile))
		plt.close()
	
		return Struct(**suettings)
	
	
	#
	# Plot function : draw a map
	#
	#	cmap=plt.get_cmap('cividis')
	#	cmap=plt.get_cmap('magma')
	def map(self, x, y, z, opfn, mode='contourf', 
			c=[None], ls=[None], lw=[None], alp=[None], 
			cmap=plt.get_cmap('inferno'),
			xl='', yl='', cbl='', 
			xlim=None, ylim=None, cblim=None, 
			logx=False, logy=False, logcb=False, loglog=False,
			pm=False, leg=False, hl=[], vl=[], title='', 
			fills=None, data="", Vector=None, 
			div=10.0, n_sline=18, hist=False,		
			square=True, seeds_angle=[0,np.pi/2], 
			**args):
	
		# Start Plotting
		msg("Plotting %s"%opfn)
		fig = plt.figure( **self.args_fig )
		ax = fig.add_subplot(111)
	
		y = y[0] if data=="1+1D" else y
		if not isinstance(x[0],(list, np.ndarray)):
			xx , yy =  np.meshgrid(x, y, indexing='xy')
		else:
			xx = x
			yy = y
	
		if xlim==None:
			xlim = [ np.min(x) , np.max(x) ]
		if ylim==None:
			ylim = [ np.min(y) , np.max(y) ]
		if cblim==None:
			cblim = [ np.min(z[z>0]) , np.max(z) ]
			msg(z, cblim)
	
		if logcb:
			cbmin , cbmax = np.log10(cblim)
		else:
			cbmin , cbmax = cblim
	
		delta = ( cbmax  - cbmin  )/float(div)
		z	= np.log10( np.where( z!=0, abs(z), np.nan ) ) if logcb else z 
	
		if (cbmax+delta - cbmin)/delta > 100:
			print("Wrong!")
			exit()
		interval = np.arange( cbmin , cbmax+delta , delta )
	
	#levels = np.linspace(cbmin, cbmax, n_lv+1)
	#norm = BoundaryNorm(levels, ncolors=cmap.N)

	#	print(xx.shape,yy.shape,z.shape,interval)
	#	img = ax.contourf( z )	
	#	img = ax.contourf( z , interval , vmin=cbmin, vmax=cbmax , extend='both' , cmap=cmap )
		img = ax.contourf( xx , yy , z , interval , vmin=cbmin, vmax=cbmax , extend='both' , cmap=cmap )
	
		if mode=="grid":
			xxi = np.hstack(( xx-dx/2. , (xx[:,-1]+dx/2.).reshape(-1,1) ))
			xxi = np.vstack(( xxi , xxi[-1] ))
			yyi = np.vstack(( yy-dy/2. , (yy[-1,:]+dy/2.).reshape(1,-1) ))
			yyi = np.hstack(( yyi , yyi[:,-1].reshape(-1,1)  ))
			img = plt.pcolormesh(xxi, yyi, z , norm=norm, rasterized=True, cmap=cmap)
	#plt.pcolormesh(xxi, yyi, z , cmap=cmap, norm=norm, rasterized=True)
	
		if mode=="contourf":
			img = ax.contourf( xx , yy , z , interval , vmin=cbmin, vmax=cbmax , extend='both' , cmap=cmap)
	#plt.contourf( xx , yy , z , n_lv , cmap=cmap)
	
		if mode=="contour":
			jM, iM = np.unravel_index(np.argmax(z), z.shape)
			plt.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
			img = plt.contour(xx, yy, z, cmap=cmap, levels=levels)
	
		if mode=="scatter":
			img = plt.scatter(xx, yy, vmin=cbmin, vmax=cbmax, c=z, s=1, cmap=cmap)
	
		ticks = np.arange( cbmin , cbmax+delta , delta ) # not +1 to cb_max
		fig.colorbar(img, ax=ax, ticks=ticks, extend='both', label=cbl, pad=0.02)
	
		if fills is not None:
			for fill in fills:
				i = fill[0]
				#ax.fill_between( x, y[0] ,fill[1],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
				ax.contourf( xx , yy, fill[1] , cmap=cmap , alpha=0.5)
		
		if Vector is not None:
			uu, vv = Vector
			th_seed = np.linspace(seeds_angle[0], seeds_angle[1], n_sline)
			rad = np.max(xlim)
			seed_points = np.array( [ rad * np.sin(th_seed) , rad * np.cos(th_seed) ]).T
			
			x_ax = np.linspace( xlim[0] , xlim[1] , 200	)
			y_ax = np.linspace( ylim[0] , ylim[1] , 200	)
	
			xx_gr , yy_gr = np.meshgrid( x_ax , y_ax , indexing='xy')
			xy = np.array([ [x,y] for x , y in zip(xx.flatten(),yy.flatten())  ])
			mth = 'linear'
			uu_gr = griddata(xy, uu.flatten(), (xx_gr, yy_gr) ,method=mth)
			vv_gr = griddata(xy, vv.flatten(), (xx_gr, yy_gr) ,method=mth)	
			plt.streamplot(xx_gr, yy_gr, uu_gr, vv_gr, 
						   density=n_sline/4, linewidth=0.3*30/n_sline, arrowsize=.3*30/n_sline, 
						   color='w', start_points=seed_points)
			
		if title:
			plt.title(title)

		if square:
			plt.gca().set_aspect('equal', adjustable='box')

		if (logx or self.logx) or loglog:
			plt.xscale("log")

		if (logx or self.logx) or loglog:
			plt.yscale('log', nonposy='clip')

		if xlim is None:
			xlim = self.xlim
		if xlim:
			plt.xlim(xlim)

		if ylim is None:
			ylim = self.ylim
		if ylim:
			plt.ylim(ylim)

		if xl: 
			plt.xlabel(xl)

		if yl:
			plt.ylabel(yl)

		if leg or self.leg:
			plt.legend(**self.args_leg)

		self.fig = fig
		return fig	

	def save(self, opfn):
		savefile = "%s/%s%s"%(self.fig_dn, opfn, self.ext)
		print(self.fig_dn)
		self.fig.savefig(savefile, transparent=True, bbox_inches='tight')
		msg("	Saved %s to %s"%(opfn, savefile))
		plt.close()

def _contour_plot(xx, yy, z, n_lv=20, cbmin=None, cbmax=None, 
				 mode="default", cmap=plt.get_cmap('viridis'), 
				 smooth=False):

	dx = xx[0,1] - xx[0,0]
	if len(yy) > 1:
		dy = yy[1,0] - yy[0,0]
	else:
		dy = dx*10

	if cbmin == None:
		cbmin = z.min()

	if cbmax == None:
		cbmax = z.max()

	levels = np.linspace(cbmin, cbmax, n_lv+1)
	norm = BoundaryNorm(levels, ncolors=cmap.N)
#	print("levels is ", levels, ' norm is', vars(norm) )

	if mode=="grid":
#		n_lv = 6	
#		std = np.std(z)
#		tick_min = 3*std
#		tick_max = 15*std		
		## Make interface coordinate
		xxi = np.hstack(( xx-dx/2. , (xx[:,-1]+dx/2.).reshape(-1,1) ))
		xxi = np.vstack(( xxi , xxi[-1] ))
		yyi = np.vstack(( yy-dy/2. , (yy[-1,:]+dy/2.).reshape(1,-1) ))
		yyi = np.hstack(( yyi , yyi[:,-1].reshape(-1,1)  ))
		return plt.pcolormesh(xxi, yyi, z , cmap=cmap, norm=norm, rasterized=True)

	if mode=="contourf":
		return plt.contourf( xx , yy , z , n_lv , cmap=cmap)

	if mode=="contour":
		jM, iM = np.unravel_index(np.argmax(z), z.shape)
		plt.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
		return plt.contour(xx, yy, z, cmap=cmap, levels=levels)

#		return plt.contour( xx , yy , z , n_lv , cmap=cmap, vmin=cbmin, vmax=cbmax ,levels=levels)
	if mode=="scatter":
		return plt.scatter(xx, yy, vmin=cbmin, vmax=cbmax, c=z, s=1, cmap=cmap)
