#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, griddata
plt.switch_backend('agg')
pyver = sys.version_info[0] + 0.1*sys.version_info[1]
#print("Message from %s"% os.path.dirname(os.path.abspath(__file__)))
debug_mode = 0
import mytools
from matplotlib.colors import BoundaryNorm, Normalize


msg = mytools.Message(__file__, debug=False)
#######################

msg("%s is used." % os.path.abspath(__file__), debug=1)
msg("This is python %s" % pyver, debug=1)
dn_this_file = os.path.dirname(os.path.abspath(__file__))


class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)

class Plotter:
    # input --> Plotter class --> default value 
    #                                  v 
    # input --> plot function -------------------> used parameter

    def __init__(self, fig_dir_path, 
                 figext="pdf", 
                 x=None, y=None, xlim=None, ylim=None, cblim=None, xl=None, yl=None, cbl=None,
                 c=[], ls=[], lw=[], alp=[], pm=False,
                 logx=False, logy=False, logxy=False, logcb=False, leg=False, 
                 figsize=None, square=False, 
                 fn_wrapper=lambda s:s, 
                 decorator=lambda y:y,
                 args_leg={"loc": 'best'}, args_fig={}):

        for k, v in locals().items():
            if k is not "self":
                setattr(self, k, v)

        if self.figsize is not None:
            self.args_fig["figsize"] = self.figsize

        # For saving figures
        self.fig_dn = os.path.abspath(fig_dir_path)
        self._mkdir(self.fig_dn)
        self.fig = None

    @staticmethod
    def _mkdir(dpath):
        if pyver >= 3.2:
            os.makedirs(dpath, exist_ok=True)
        else:
            if not os.path.exists(dpath):
                os.makedirs(dpath)

    @staticmethod
    def notNone(*args):
        for a in args:
            if a is not None:
                return a
        else:
            return None

    def reform_ylist(self, x, l):
        x0 = x if x is not None else self.x

        def checktype(a, types):
            if len(a) == len(types):
                return all([isinstance(aa, t) for aa, t in zip(a, types)])
            else:
                return False
#       def analyze_vector(vec):
#           for vv in vec:
#               if isvector(v):
#                   return "V"
#               elif is:

        def isvector(vec):
            return not np.isscalar(vec)

#       def isdata(d):

        def check_list_type(vec):
            vector = (list, np.ndarray)

            ## [k, x, y]
            if checktype(vec, (str, vector, vector)) \
               and len(vec[1]) == len(vec[2]):
                return 1

            ## [k, y]
            elif checktype(vec, (str,  vector))\
                    and (len(x0) == len(vec[1])):
                return 2

            ## [x, y]
            elif checktype(vec, (vector,  vector))\
                    and (len(vec[0]) == len(vec[1])) \
                    and (not isinstance(vec[0][0], str)):
                return 3

            # y
            elif len(vec) == len(x0) \
                    and isvector(vec):
                return 4

            else:
                return -1

        if isvector(l):
            itype = check_list_type(l)
            msg("itype = ", itype, debug=1)
            if itype == 1:
                return [l]

            elif itype == 2:
                return [[l[0], x0, l[1]]]

            elif itype == 3:
                return [[0, l[0], l[1]]]

            elif itype == 4:
                return [[0, x0, l]]
            else:

                jtype = check_list_type(l[0])
                msg("jtype = ", jtype, debug=1)
                if jtype == 1:
                    return l

                elif jtype == 2:
                    return [[k, x0, y] for k, y in l]

                elif jtype == 3:
                    return [[i, x, y] for i, x, y in enumerate(l)]

                elif jtype == 4:
                    return [[0, x0, l]]

            raise Exception("Unknown type")

    #
    # Plot function : draw lines
    #
    # usage:
    # plot(y, "name")
    # plot([y1, y2, ..., yN], "test", x=x)
    # plot([leg1, y1], [leg2, y2], ..., [legN, yN] ], "test")
    # plot([leg1, x1, y1], [leg2, x2, y2], ..., [legN, xN, yN] ], "test")
    #

    def plot(self, y_list, out=None, x=None, c=[None],
             ls=[], lw=[], alp=[], leg=None, frg_leg=0, title='',
             xl=None, yl=None, xlim=None, ylim=None,
             logx=False, logy=False, logxy=False, pm=False,
             hl=[], vl=[], fills=None, arrow=[], lbs=None,
             datatype="", square=None, show=False, save=True, result="fig",
             *args, **kwargs):

        input_settings = locals()  
        for k, v in input_settings.items():
            setattr(self, "inp_"+k, v)

        # Start Plotting
        msg("Plotting %s" % out)
        fig = plt.figure(**self.args_fig)
        ax = fig.add_subplot(111)

        # Preprocesing
        data = self.reform_ylist(x, y_list)

        # Main plot
        for i, (k, x, y) in enumerate(data):
            # Preprocessing for each plot
            lb_i = lbs[i] if lbs else k
            xy_i = (x, self.decorator(y)) if (x is not None) else (self.decorator(y),)
            c_i = c[i] if len(c) >= i+1 else None
            ls_i = ls[i] if len(ls) >= i+1 else None
            lw_i = lw[i] if len(lw) >= i+1 else None
            alp_i = alp[i] if len(alp) >= i+1 else None

            # Plot
            msg('Inputs to plot', xy_i, debug=1)
            ax.plot(*xy_i, label=lb_i, c=c_i, ls=ls_i,
                    lw=lw_i, alpha=alp_i, zorder=-i, **kwargs)
            if pm:
                ax.plot(x, -y, label=lb_i, c=c_i, ls=ls_i,
                        lw=lw_i, alpha=alp_i, zorder=-i, **kwargs)

            # Enroll flags
            if lb_i and frg_leg == 0:
                frg_leg = 1

        # Fill
        if fills:
            for fil in fills:
                i = fil[0]
                ax.fill_between(
                    x, fil[1], fil[2], facecolor=c_def_rgb[i], alpha=0.2, zorder=-i-0.5)

        # Horizontal lines
        for h in hl:
            plt.axhline(y=h)

        # Vertical lines
        for v in vl:
            plt.axvline(x=v, alpha=0.5)

        # Arrow
        for i, ar in enumerate(arrow):

            if True:
                c_i = c[i] if len(c) >= i+1 else None
                print(ar)
                ax.annotate('', xytext=(ar[0], ar[1]), xy=(ar[2], ar[3]), 
                            xycoords='data', annotation_clip=False,size=30,
                            arrowprops=dict(shrink=0, width=3, headwidth=8, headlength=8, lw=0,
                                            connectionstyle='arc3', fc=c_i, ec=c_i)
                            )
            else:
                ar_x = ar[0]
                ar_y = ar[1]
                ar_dx = ar[2] - ar[0]
                ar_dy = ar[3] - ar[1]
                ar_len = np.sqrt(ar_dx**2 + ar_dy**2)
                linewidth = mpl.rcParams['lines.linewidth']
                import matplotlib.patches as pat
                curve = pat.ArrowStyle.Curve()
                #curve.set_capstyle("batt")
                curve.capstyle = "batt"
                curve._capstyle = "batt"
                c_i = c[i] if len(c) >= i+1 else None
                ls_i = ls[i] if len(ls) >= i+1 else None
                lw_i = lw[i] if len(lw) >= i+1 else linewidth
                ax.annotate('', xytext=(ar[0], ar[1]), xy=(ar[2], ar[3]), 
                            xycoords='data', annotation_clip=False,
                            arrowprops=dict(arrowstyle=curve,lw=lw_i*0.5,
                                            connectionstyle='arc3', facecolor=c_i, edgecolor=c_i, linestyle=ls_i)
                            )
    #            plt.plot((ar[0], ar[2]), (0.01*ar[0]+0.99*ar[1], 0.01*ar[2]+0.99*ar[3]), c=c_i, ls=ls_i, lw=lw_i)
    #            plt.plot((ar[0], ar[2]), (0.01*ar[0]+0.99*ar[1], 0.01*ar[2]+0.99*ar[3]), c=c_i, ls=ls_i, lw=lw_i)
                ax.annotate('', xytext=(0.01*ar[0]+0.99*ar[2], 0.01*ar[1]+0.99*ar[3]), xy=(ar[2], ar[3]), 
                            xycoords='data', annotation_clip=False,
                            arrowprops=dict(lw=0,headwidth=lw_i*3, headlength=lw_i*3,
                                            connectionstyle='arc3', facecolor=c_i, edgecolor=c_i, linestyle=ls_i)
                            )

        # Postprocessing
        plt.title(title)

        if (leg or self.leg) and frg_leg == 1:
            plt.legend(**self.args_leg)

        if (logx or self.logx) or (logxy or self.logxy):
            plt.xscale("log", nonposx='clip')

        if (logy or self.logy) or (logxy or self.logxy):
            plt.yscale('log', nonposy='clip')

        plt.xlim(self.notNone(xlim, self.xlim, [min(x), max(x)]))
        plt.ylim(self.notNone(ylim, self.ylim, [min(y), max(y)]))
        plt.xlabel(self.notNone(xl, self.xl, ''))
        plt.ylabel(self.notNone(yl, self.yl, ''))

        
        if self.notNone(square, self.square):
            plt.gca().set_aspect('equal', adjustable='box')

        self.fig = fig

        if show:
            self.save(show)

        # Saving
        if save:
            self.save(out)

        if result=="fig":
            return fig
        elif result=="class":
            return Struct(input_settings)
    #
    # Plot function : draw a map
    #
    #   cmap=plt.get_cmap('cividis')
    #   cmap=plt.get_cmap('magma')

    def map(self, z=None, out=None, x=None, y=None, mode='contourf',
            c=[None], ls=[None], lw=[None], alp=[None],
            cmap=plt.get_cmap('cividis'),
            xl=None, yl=None, cbl=None,
            xlim=None, ylim=None, cblim=None,
            logx=None, logy=None, logcb=None, logxy=False,
            pm=False, leg=False, hl=[], vl=[], title=None,
            fills=None, data="", Vector=None,
            div=10.0, n_sline=18, hist=False,
            square=None, seeds_angle=[0, np.pi/2],
            save=True, show=False, result="fig",
            twoaxis=False, 
            **args):

        for k, v in locals().items():
            setattr(self, "inp_"+k, v)

        # Start Plotting
        msg("Plotting %s" % out)
        self.fig = plt.figure(**self.args_fig)
        self.ax = self.fig.add_subplot(111)
        #if twoaxis:
        #    ax2 = ax.twinx()


        x = self.notNone(x, self.x)
        y = self.notNone(y, self.y)

        y = y[0] if data == "1+1D" else y
        if not isinstance(x[0], (list, np.ndarray)):
            xx, yy = np.meshgrid(x, y, indexing='xy')
        else:
            xx = x
            yy = y

        z = self.decorator(z)
   
        self.ax.set_xlim(self.notNone(xlim, self.xlim, [x[0], x[-1]]))
        self.ax.set_ylim(self.notNone(ylim, self.ylim, [y[0], y[-1]]))
        self.ax.set_xlabel(self.notNone(xl, self.xl, ""))
        self.ax.set_ylabel(self.notNone(yl, self.yl, ""))

        cblim = self.notNone(cblim, self.cblim, [np.min(z), np.max(z)] )
        cblim[1] *=  0.999
        if self.notNone(logcb, self.logcb):  # (logcb or self.logcb):
            cblim = np.log10(cblim)
            #print(z, np.max(z), np.min(z) )            
            z = np.log10(np.where(z != 0, abs(z), np.nan))

        if cblim is not None:
            delta = (cblim[1] - cblim[0])/float(div)

            if (cblim[1]+delta - cblim[0])/delta > 100:
                raise Exception("Wrong cblim, probably...", cblim)

        interval = np.arange(cblim[0], cblim[1]+delta, delta)


        levels = np.linspace(cblim[0], cblim[1], int(div)+1)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        if mode == "grid":
            # Note: 
            # this guess of xi, yi relyies on the assumption where
            # grid space is equidistant.

#            print( len(x[:,0]-0.5*(x[:,1]-x[:,0])), 
#                              (0.5*(x[:,0:-1]+x[:,1:]) ).shape  )    
#np.stack(( [x[:,0]-0.5*(x[:,1]-x[:,0])], 
#                              0.5*(x[:,0:-1]+x[:,1:]), 
#                             [x[:,-1]+0.5*(x[:,-1]-x[:,-2])] ), axis=-1)
#            yi = np.hstack([[y[0]-0.5*(y[1]-y[0])], 0.5*( y[0:-1]+y[1:] ), [y[-1] + 0.5*(y[-1]-y[-2])]])
#            xxi, yyi = np.meshgrid(xi, yi, indexing='xy')  

#            dx = x[1] - x[0] if len(x) != 1 else y[1] - y[0]
#            dy = y[1] - y[0] if len(y) != 1 else x[1] - x[0]
#            xxi = np.hstack((xx-dx/2., (xx[:, -1]+dx/2.).reshape(-1, 1)))
#            xxi = np.vstack((xxi, xxi[-1]))
#            yyi = np.vstack((yy-dy/2., (yy[-1, :]+dy/2.).reshape(1, -1)))
#            yyi = np.hstack((yyi, yyi[:, -1].reshape(-1, 1)))

  #          print(x.shape, y.shape)
            
            xxi, yyi = mytools.make_meshgrid_interface(xx, yy)
            #print(xx.shape,"\n", xxi.shape, '\n')
            #print(yy,"\n", yyi, "\n") 
            #print(vars(norm), z, xxi, yyi )
 #           print(xxi.shape, yyi.shape, z.shape, x.shape, y.shape)
            #print( (z-np.min(z))/(np.max(z)-np.min(z)) ) 
            #img = ax.pcolormesh(xxi.T, yyi.T, (z.T-np.min(z))/(np.max(z)-np.min(z)), 
            #                     cmap=cmap)
            #img = ax.pcolormesh(xxi.T, yyi.T, z.T, norm=norm, vmin=cblim[0],  vmax=cblim[1],
            #                     cmap=cmap,shading='flat')

#            print(np.max(xx), np.max(xxi))

#            exit()
#            for x0, xx0, xxi0 in zip(x, xx, xxi):                
#                print(x0, xxi0)

            #print(xxi, yyi)
            img = self.ax.pcolormesh(xxi.T, yyi.T, z.T, norm=norm, vmin=cblim[0],  vmax=cblim[1],
                                 cmap=cmap, rasterized=True)

            #img = ax.pcolormesh(x.T, y.T, z.T, norm=norm, vmin=cblim[0],  vmax=cblim[1],
            #                     cmap=cmap, shading='gouraud', rasterized=True)

        elif mode == "contourf":
            # Note:
            # if len(x) or len(y) is 1, contourf returns an eroor.
            # Instead of this, use "grid" method.
            img = self.ax.contourf(xx, yy, z, interval, vmin=cblim[0],
                              vmax=cblim[1], extend='both', cmap=cmap)
            
            #ax.contourf( xx , yy , z , n_lv , cmap=cmap)

        elif mode == "contour":
            #jM, iM = np.unravel_index(np.argmax(z), z.shape)
            #self.ax.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
            #print(levels)
            img = self.ax.contour(xx, yy, z, cmap=cmap, levels=levels, linewidths=lw, extend='both', corner_mask =True)
            #img = plt.contour(xx, yy, z,levels=levels )
#            print(type(img))

        elif mode == "contourfill":
            #img = self.ax.contourf(xx, yy, z, interval, vmin=cblim[0],
            #                  vmax=cblim[1], extend='both', cmap=cmap)
            #extent = [self.ax.get_xlim()[0]-1,self.ax.get_xlim()[1]+1, self.ax.get_ylim()[0],self.ax.get_ylim()[1]]
            #print(extent)
            #img = self.ax.imshow(z, cmap=cmap, origin='lower', interpolation='nearest', extent=extent)
            xxi, yyi = mytools.make_meshgrid_interface(xx, yy)
            img = self.ax.pcolormesh(xxi.T, yyi.T, z.T,  vmin=cblim[0],  vmax=cblim[1],
                                 cmap=cmap, rasterized=True)
            lines =self.ax.contour(xx, yy, z, levels=levels, colors='w', linewidths=lw)
            #s = plt.contour(xx, yy, z, levels=levels)
            #self.ax.clabel(cs, levels[1::2])

        elif mode == "scatter":
            img = self.ax.scatter(xx, yy, vmin=cblim[0], vmax=cblim[1],
                              c=z, s=1, cmap=cmap)
        else:
            raise Exception("No such a mode for mapping: ", mode)

        ticks = np.arange(cblim[0], cblim[1]+delta, delta)  # not +1 to cb_max
        img.set_clim(*cblim)
        print(img.get_cmap(), img.get_array() , img.get_clim()) 
        #self.fig.colorbar(img, ax=self.ax, ticks=ticks,
        #             extend='both', label=self.notNone(cbl, self.cbl), pad=0.02)
        

        #smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        #smap = mpl.contour.QuadContourSet(self.ax, levels)
        #smap.set_array([])
        #smap = img
        #smap.levels = levels
        cbar = self.fig.colorbar(img, ax=self.ax, ticks=ticks, extend='both', label=self.notNone(cbl, self.cbl), pad=0.02, format="%.2g")
        #img.levels = levels
        #cbar = self.fig.colorbar(img)
        if mode == "contourfill":
            cbar.add_lines(lines)

        if fills is not None:
            for fill in fills:
                i = fill[0]
                #ax.fill_between( x, y[0] ,fill[1],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
                self.ax.contourf(xx, yy, fill[1], cmap=cmap, alpha=0.5)

        if Vector is not None:
            uu, vv = Vector# self.notNone(Vector,self.Vector)
            #seeds_angle = #self.notNone(seeds_angle, self.seeds_angle)
            th_seed = np.linspace(seeds_angle[0], seeds_angle[1], n_sline)
            rad = self.ax.get_xlim()[1]
            seed_points = np.array(
                [rad * np.sin(th_seed), rad * np.cos(th_seed)]).T
            x_ax = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], 200)
            y_ax = np.linspace(self.ax.get_ylim()[0], self.ax.get_ylim()[1], 200)
            xx_gr, yy_gr = np.meshgrid(x_ax, y_ax, indexing='xy')
            xy = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())])
            mth = 'linear'
            uu_gr = griddata(xy, uu.flatten(), (xx_gr, yy_gr), method=mth)
            vv_gr = griddata(xy, vv.flatten(), (xx_gr, yy_gr), method=mth)
            self.ax.streamplot(xx_gr, yy_gr, uu_gr, vv_gr,
                           density=n_sline/4, linewidth=0.3*30/n_sline, arrowsize=.3*30/n_sline,
                           color='w', start_points=seed_points)

        # Horizontal lines
        for h in hl:
            plt.axhline(y=h)

        # Vertical lines
        for v in vl:
            plt.axvline(x=v, alpha=0.5)

        if title:
            self.ax.set_title(title)


        if self.notNone(square, self.square):
            self.ax.set_aspect('equal', adjustable='box')

        if (logx or self.logx) or (logxy or self.logxy):
            self.ax.set_xscale("log", nonposx='clip')

        if (logx or self.logx) or (logxy or self.logxy):
            self.ax.set_yscale('log', nonposy='clip')

        if leg or self.leg:
            self.ax.set_legend(**self.args_leg)

#        self.fig = fig

        if show:
            self.save(show)

        # Saving
        if save:
            self.save(out)


        if result=="fig":
            return self.fig

        elif result=="class":
            return self #Struct(self) 



    def save(self, out):
        savefile = "%s/%s.%s" % (self.fig_dn, self.fn_wrapper(out), self.figext)
        self.fig.savefig(savefile, bbox_inches='tight')
        msg("   Saved %s to %s" % (out, savefile))
        plt.close()
        plt.clf()

    def show(self, out):
        plt.show()

import itertools 
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def generate_cmap_rgb(cmap):
    return [(int(c[1:3], 16)/255., int(c[3:5], 16)/255., int(c[5:7], 16)/255.) for c in cmap]

class cycle_list:
    def __init__(self, cylist):
        self.cylist = cylist
        self.len = len(cylist)
    def __getitem__(self, i):
        return self.cylist[i%self.len]

c_def = ["#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#34495e",
         "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50"]
myls = cycle_list(["-","--",":","-."])
myls_txt = cycle_list(["solid","dashed","dotted","dashdot"])
myc = cycle_list(c_def)


c_def_rgb = generate_cmap_rgb(c_def)
c_cycle = cycler(color=c_def)
mpl.rc('font', weight='bold', size=17)
mpl.rc('lines', linewidth=4, color="#2c3e50", markeredgewidth=0.9, #solid_capstyle="projecting", dash_capstyle='butt',
       marker=None, markersize=4)
mpl.rc('patch', linewidth=2, edgecolor='k')
mpl.rc('text', color='#2c3e50')
mpl.rc('axes', linewidth=2, facecolor='none', titlesize=30, labelsize=20,
       labelweight='bold', prop_cycle=c_cycle, grid=False)
mpl.rc('xtick.major', size=6, width=2)
mpl.rc('ytick.major', size=6, width=2)
mpl.rc('xtick.minor', size=3, width=2)
mpl.rc('ytick.minor', size=3, width=2)
mpl.rc('xtick', direction="in", bottom=True, top=True)
mpl.rc('ytick', direction="in", left=True, right=True)
mpl.rc('grid', color='#c0392b', alpha=0.3, linewidth=1)
mpl.rc('legend', loc='upper right', numpoints=1, fontsize=12, borderpad=0.5,
       markerscale=1, labelspacing=0.2, frameon=True, facecolor='w',
       framealpha=0.9, edgecolor='#303030', handlelength=1, handleheight=0.5,
       fancybox=False)
golden_ratio = np.array([1, (1+np.sqrt(5))*0.5])
golden_ratio_rev = np.array([(1+np.sqrt(5))*0.5, 1])
mpl.rc('figure', figsize=golden_ratio_rev*5, dpi=160, edgecolor="k")
mpl.rc('savefig', dpi=200, facecolor='none', edgecolor='none')
mpl.rc('path', simplify=True, simplify_threshold=1)
mpl.rc('pdf', compression=9, fonttype=3)
