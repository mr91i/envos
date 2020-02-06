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


msg = mytools.Message(__file__)
#######################

msg("%s is used." % os.path.abspath(__file__))
msg("This is python %s" % pyver)
dn_this_file = os.path.dirname(os.path.abspath(__file__))


class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)

class Plotter:
    #   def_logx = False
    #   def_logy = False
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
            self.args_fig["figsize"] = sel.figsize

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
             ls=[], lw=[], alp=[], leg=True, frg_leg=0, title='',
             xl=None, yl=None, xlim=None, ylim=None,
             logx=False, logy=False, logxy=False, pm=False,
             hl=[], vl=[], fills=None, arrow=[], lbs=None,
             datatype="", square=None, save=True, result="fig",
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
            ax.annotate('', xytext=(ar[0], ar[1]), xy=(ar[2], ar[3]), xycoords='data', annotation_clip=False,
                        arrowprops=dict(shrink=0, width=3, headwidth=8, headlength=8,
                                        connectionstyle='arc3', facecolor=c[i], edgecolor=c[i])
                        )

        # Postprocessing
        plt.title(title)

        if leg and frg_leg == 1:
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
            cmap=plt.get_cmap('inferno'),
            xl=None, yl=None, cbl=None,
            xlim=None, ylim=None, cblim=None,
            logx=None, logy=None, logcb=None, logxy=False,
            pm=False, leg=False, hl=[], vl=[], title=None,
            fills=None, data="", Vector=None,
            div=10.0, n_sline=18, hist=False,
            square=None, seeds_angle=[0, np.pi/2],
            save=True, result="fig",
            **args):

        for k, v in locals().items():
            setattr(self, "inp_"+k, v)

        # Start Plotting
        msg("Plotting %s" % out)
        fig = plt.figure(**self.args_fig)
        ax = fig.add_subplot(111)

        x = self.notNone(x, self.x)
        y = self.notNone(y, self.y)

        y = y[0] if data == "1+1D" else y
        if not isinstance(x[0], (list, np.ndarray)):
            xx, yy = np.meshgrid(x, y, indexing='xy')
        else:
            xx = x
            yy = y

        z = self.decorator(z)
   
        plt.xlim(self.notNone(xlim, self.xlim, [x[0], x[-1]]))
        plt.ylim(self.notNone(ylim, self.ylim, [y[0], y[-1]]))
        plt.xlabel(self.notNone(xl, self.xl, ""))
        plt.ylabel(self.notNone(yl, self.yl, ""))

        cblim = self.notNone(cblim, self.cblim, [np.min(z), np.max(z)] )
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
        norm = BoundaryNorm(levels, ncolors=cmap.N)
        if mode == "grid":
            # Note: 
            # this guess of xi, yi relyies on the assumption where
            # grid space is equidistant.
            
            dx = x[1] - x[0] if len(x) != 1 else y[1] - y[0]
            dy = y[1] - y[0] if len(y) != 1 else x[1] - x[0]
            xxi = np.hstack((xx-dx/2., (xx[:, -1]+dx/2.).reshape(-1, 1)))
            xxi = np.vstack((xxi, xxi[-1]))
            yyi = np.vstack((yy-dy/2., (yy[-1, :]+dy/2.).reshape(1, -1)))
            yyi = np.hstack((yyi, yyi[:, -1].reshape(-1, 1)))

            img = plt.pcolormesh(xxi, yyi, z, norm=norm,
                                 rasterized=True, cmap=cmap)

        if mode == "contourf":
            # Note:
            # if len(x) or len(y) is 1, contourf returns an eroor.
            # Instead of this, use "grid" method.
            img = ax.contourf(xx, yy, z, interval, vmin=cblim[0],
                              vmax=cblim[1], extend='both', cmap=cmap)
            #plt.contourf( xx , yy , z , n_lv , cmap=cmap)

        if mode == "contour":
            jM, iM = np.unravel_index(np.argmax(z), z.shape)
            plt.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
            img = plt.contour(xx, yy, z, cmap=cmap, levels=levels)

        if mode == "scatter":
            img = plt.scatter(xx, yy, vmin=cblim[0], vmax=cblim[1],
                              c=z, s=1, cmap=cmap)

        ticks = np.arange(cblim[0], cblim[1]+delta, delta)  # not +1 to cb_max
        fig.colorbar(img, ax=ax, ticks=ticks,
                     extend='both', label=self.notNone(cbl, self.cbl), pad=0.02)

        if fills is not None:
            for fill in fills:
                i = fill[0]
                #ax.fill_between( x, y[0] ,fill[1],facecolor=c_def_rgb[i],alpha=0.3,zorder=-i-0.5)
                ax.contourf(xx, yy, fill[1], cmap=cmap, alpha=0.5)

        if Vector is not None:
            uu, vv = Vector# self.notNone(Vector,self.Vector)
            #seeds_angle = #self.notNone(seeds_angle, self.seeds_angle)
            th_seed = np.linspace(seeds_angle[0], seeds_angle[1], n_sline)
            rad = ax.get_xlim()[1]
            seed_points = np.array(
                [rad * np.sin(th_seed), rad * np.cos(th_seed)]).T
            x_ax = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
            y_ax = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
            xx_gr, yy_gr = np.meshgrid(x_ax, y_ax, indexing='xy')
            xy = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())])
            mth = 'linear'
            uu_gr = griddata(xy, uu.flatten(), (xx_gr, yy_gr), method=mth)
            vv_gr = griddata(xy, vv.flatten(), (xx_gr, yy_gr), method=mth)
            plt.streamplot(xx_gr, yy_gr, uu_gr, vv_gr,
                           density=n_sline/4, linewidth=0.3*30/n_sline, arrowsize=.3*30/n_sline,
                           color='w', start_points=seed_points)

        if title:
            plt.title(title)


        print(square, self.square, self.notNone(square, self.square)) 
        if self.notNone(square, self.square):
            plt.gca().set_aspect('equal', adjustable='box')

        if (logx or self.logx) or (logxy or self.logxy):
            plt.xscale("log", nonposx='clip')

        if (logx or self.logx) or (logxy or self.logxy):
            plt.yscale('log', nonposy='clip')

        if leg or self.leg:
            plt.legend(**self.args_leg)

        self.fig = fig
        # Saving
        if save:
            self.save(out)

        if result=="fig":
            return fig
        elif result=="class":
            return Struct(self) 

    def save(self, out):
        savefile = "%s/%s.%s" % (self.fig_dn, self.fn_wrapper(out), self.figext)
        self.fig.savefig(savefile, bbox_inches='tight')
        msg("   Saved %s to %s" % (out, savefile))
        plt.close()
        plt.clf()





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

c_def = ["#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#34495e",
         "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50"]
c_def_rgb = generate_cmap_rgb(c_def)
c_cycle = cycler(color=c_def)
mpl.rc('font', weight='bold', size=17)
mpl.rc('lines', linewidth=4, color="#2c3e50", markeredgewidth=0.9,
       marker=None, markersize=4)
mpl.rc('patch', linewidth=1, edgecolor='k')
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
