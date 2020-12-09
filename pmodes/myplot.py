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
debug_mode = 0
import pmodes.tools
from matplotlib.colors import BoundaryNorm, Normalize, LogNorm
from matplotlib.ticker import LogFormatter

import matplotlib.colors as mcol
from cycler import cycler
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

import logging
logger = logging.getLogger(__name__)
#handler = logging.NullHandler()
handler = logging.StreamHandler()
#handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter("[%(filename)s] %(levelname)s: %(message)s"))
#logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

#######################

logger.debug("%s is used." % os.path.abspath(__file__))
logger.debug("This is python %s" % pyver)
dpath_this_file = os.path.dirname(os.path.abspath(__file__))


class Struct:
    def __init__(self, entries):
        self.__dict__.update(entries)

class Params:
    def __init__(self, locals_dict):
        for k, v in locals_dict.items():
            if k is not "self":
                setattr(self, k, v)

    def gen_with_priority(self,*Params_classes):
        # higher to lower priority
        for cls in Params_classes:
            for k, v in cls.__dict__.items():
                #print(k,v)
                if hasattr(self, k):
                    setattr(self, k, notNone(getattr(self,k), v))
                else:
                    setattr(self, k, v)
        return self

class Plotter:
    # input --> Plotter class --> default value
    #                                  v
    # input --> plot function -------------------> used parameter

    def __init__(self, dpath_fig="F", fig_ext="pdf",
                 x=None, y=None, xlim=None, ylim=None, ctlim=None, xl="", yl="", cbl="",
                 c=[], ls=[], lw=[], alp=[], pm=False,
                 logx=False, logy=False, logxy=False, logcb=False, leg=False,
                 figsize=None, square=False, show=False, save=True, continu=False,
                 fn_wrapper=lambda s:s,
                 decorator=lambda y:y,
                 args_leg={"loc": 'best'}, args_fig={}):

        self.df = Params(locals())
#        if dpath_fig is not None:
#            self.dp_fig = os.path.abspath(dpath_fig)
#            self._mkdir(self.dp_fig)
#        else:
#            self.dp_fig = None

        ## common
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
        #x0 = x if x is not None else self.x

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
                    and (len(x) == len(vec[1])):
                return 2

            ## [k, y]
            elif checktype(vec, (str,  float)):
                return 3

            ## [x, y]
            elif checktype(vec, (vector,  vector))\
                    and (len(vec[0]) == len(vec[1])) \
                    and (not isinstance(vec[0][0], str)):
                return 4

            # y
            elif len(vec) == len(x) \
                    and isvector(vec):
                return 5

            else:
                return -1

        if isvector(l):
            itype = check_list_type(l)
            logger.debug("itype = %d",itype)
            if itype == 1:
                return [l]

            elif itype == 2:
                return [[l[0], x, l[1]]]

            elif itype == 3:
                return [[l[0], x, np.full_like(x,l[1])]]

            elif itype == 4:
                return [[0, l[0], l[1]]]

            elif itype == 5:
                return [[0, x, l]]
            else:
                jtype = check_list_type(l[0])
                logger.debug("jtype = %d", jtype)
                if jtype == 1:
                    return l

                elif jtype == 2:
                    return [[k, x, y] for k, y in l]

                elif jtype == 4:
                    return [[i, x, y] for i, x, y in enumerate(l)]

                elif jtype == 5:
                    return [[0, x, l]]

            raise Exception("Unknown type", l)

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
             logx=None, logy=None, logxy=None, pm=None,
             hl=[], vl=[], fills=None, arrow=[], lbs=None,
             datatype="", square=None, save=None, show=None, continu=None,
             dpath_fig=None, fn_wrapper=None, fig_ext=None,
             result="fig",fillalpha=0.2, zorder0=0,fillmode="shade",
             *args, **kwargs):

        #input_settings = locals()
        #for k, v in input_settings.items():
        #    setattr(self, "inp_"+k, v)
        inp = Params(locals())
        use = Params.gen_with_priority(inp, self.df)
        self.use = use

        # Start Plotting
        logger.debug("Plotting %s", out)
        if not use.continu:
            self.fig = plt.figure(figsize=use.figsize, **self.df.args_fig)
            self.ax = self.fig.add_subplot(111)
        else:
            pass
        # Preprocesing
        data = self.reform_ylist(use.x, y_list)

        # Pre-processing
        plt.title(title)


        if use.logx or use.logxy:
            plt.xscale("log", nonpositive='clip')

        if use.logy or use.logxy:
            plt.yscale('log', nonpositive='clip')

        if use.xlim:
            self.ax.set_xlim(use.xlim)
        else:
            self.ax.set_xlim(auto=True)

        if use.ylim:
            self.ax.set_ylim(use.ylim)
        else:
            self.ax.set_ylim(auto=True)

        self.ax.set_xlabel(notNone(use.xl, ''))
        self.ax.set_ylabel(notNone(use.yl, ''))

        # Main plot
        for i, (k, x, y) in enumerate(data):
            #print( x, use.x)
            # Preprocessing for each plot
            lb_i = lbs[i] if lbs else k
            #xy_i = (x, self.decorator(y)) if (x is not None) else (self.decorator(y),)
            xy_i = (x, use.decorator(y))
            c_i = c[i] if len(c) >= i+1 else None
            ls_i = ls[i] if len(ls) >= i+1 else None
            lw_i = lw[i] if len(lw) >= i+1 else None
            alp_i = alp[i] if len(alp) >= i+1 else None

            # Plot
            logger.debug('Inputs to plot {}'.format(xy_i))
            self.ax.plot(*xy_i, label=lb_i, c=c_i, ls=ls_i,
                    lw=lw_i, alpha=alp_i, zorder=-i+zorder0, **kwargs)
            if pm:
                self.ax.plot(x, -y, label=lb_i, c=c_i, ls=ls_i,
                        lw=lw_i, alpha=alp_i, zorder=-i+zorder0, **kwargs)

            # Enroll flags
            if lb_i and frg_leg == 0:
                frg_leg = 1

        # Fill
        if fills:
            for fil in fills:
                i = fil[0]
                falp_i  = (fillalpha[i] if len(fillalpha) >= i+1 else None) if isinstance(fillalpha, list) else fillalpha
                if fillmode=="shade":
                    self.ax.fill_between(
                        x, fil[1], fil[2], facecolor=c[i], alpha=falp_i, zorder=-i-0.5+zorder0)
                if fillmode=="mesh":
                    self.ax.fill_between(
                        x, fil[1], fil[2], facecolor="none", hatch=['/'*4, r"\ "*4, "|"*4][i], edgecolor=c[i], lw=1 , alpha=1, zorder=-i-0.5+zorder0)

        # Horizontal lines
        for h in hl:
            plt.axhline(y=h)

        # Vertical lines
        for v in vl:
            plt.axvline(x=v, alpha=0.5)

        # Arrow
        for i, ar in enumerate(arrow):
            self.ax.annotate('', xytext=(ar[0], ar[1]), xy=(ar[2], ar[3]), xycoords='data',
                           clip_on=True, annotation_clip=True, zorder=-i+zorder0,
                        arrowprops=dict(shrink=0, width=3, headwidth=8, headlength=8,lw=0,
                                        connectionstyle='arc3', facecolor=c[i], edgecolor=c[i])
                        )


        if use.leg and frg_leg == 1:
            self.ax.legend(**self.df.args_leg)

        if use.square:
            plt.gca().set_aspect('equal', adjustable='box')

        if use.show:
            self.show(out)

        # Saving
        if use.save:
            #self.use = use
            self.save(out, dpath_fig=use.dpath_fig, fig_ext=use.fig_ext, fn_wrapper=use.fn_wrapper)
            #self.save(out)
            #self.save(out)
        else:
            self.use = use

        plt.close()
        plt.clf()

        if result=="fig":
            return self.fig

        elif result=="class":
            return inp
    #
    # Plot function : draw a map
    #
    #   cmap=plt.get_cmap('cividis')
    #   cmap=plt.get_cmap('magma')

    def map(self, z=None, out=None, x=None, y=None, mode='contourf',
            xx=None, yy=None,
            points=None,
            c=[None], ls=[None], lw=[None], alp=[None],
            cmap=plt.get_cmap('inferno'),
            xl=None, yl=None, cbl=None,
            xlim=None, ylim=None,
            ctlim=None, ctdelta=None, ctax=None, Nct=10,
            cllim=None, cldelta=None, clax=None, Ncl=None,
            cnlim=None,
            logx=None, logy=None, logcb=None, logxy=False,
            pm=False, leg=None, hl=[], vl=[], title=None,
            fills=None, data="", Vector=None,
            n_sline=18, hist=False,
            square=None, seeds_angle=[0, np.pi/2],
            save=True, result="fig",
            **args):

        inp = Params(locals())
        use = Params.gen_with_priority(inp, self.df)
        self.use = use

        logger.debug("Plotting %s" % out)
        self.fig = plt.figure(**self.df.args_fig)
        self.ax = self.fig.add_subplot(111)
        datatype = "points" if isinstance(points, np.ndarray) else "list"

        if datatype=="list":
            x = use.x
            y = use.y
            z = use.decorator(z)
            ctlim_z = [np.nanmin( np.where(z < -1e100, +np.Inf, z) ),np.nanmax( np.where(z > 1e100, -np.Inf, z) )]
        elif datatype=="points":
            points_x = points[:,0]
            points_y = points[:,1]
            points_z = points[:,2]
            x = [min(points_x), max(points_x)]
            y = [min(points_y), max(points_y)]
            ctlim_z = [np.nanmin( np.where(points_z <= -1e100, +np.Inf, points_z) ),np.nanmax( np.where(points_z > 1e100, -np.Inf, points_z) )]
            if logcb:
                logpz = np.log10(points_z)
                ctlim_z[0] = 10**( logpz[np.isfinite(logpz)].min())
                points_z = points_z.clip(ctlim_z[0])
        else:
            raise Exception

        self.ax.set_xlim(notNone(use.xlim, [x[0], x[-1]]))
        self.ax.set_ylim(notNone(use.ylim, [y[0], y[-1]]))
        self.ax.set_xlabel(use.xl)
        self.ax.set_ylabel(use.yl)

        ## We need these information to make colorbar and map
        # 1. Ticks <-- ctax "color tick axis"
        # 2. Contour levels <-- clax "levels"
        # 3. Nomalizing range <-- cnlim

        ## Set ctax
        ctlim = notNone(use.ctlim, ctlim_z)
        print(ctlim)
        if ctdelta is None:
            ctax_data = np.linspace(ctlim[0], ctlim[-1], Nct+1) if not use.logcb else\
                        np.logspace(np.log10(ctlim[0]), np.log10(ctlim[-1]), Nct+1)
        else:
            ctax_data = np.range(ctlim[0], ctlim[-1]+ctdelta, ctdelta) if not use.logcb else\
                        10**np.range(np.log10(ctlim[0]), np.log10(ctlim[-1]*ctdelta), np.log10(ctdelta) )
        ctax = notNone(use.ctax, ctax_data)


        ## Set clax
        cllim = use.cllim
        Ncl = Nct if Ncl is None else Ncl
        if cllim is not None:
            if cldelta is None:
                clax_data = np.linspace(cllim[0], cllim[-1], Ncl+1) if not use.logcb else\
                            np.logspace(np.log10(cllim[0]), np.log10(cllim[-1]), Ncl+1)
            else:
                clax_data = np.range(cllim[0], cllim[-1]+cldelta, cldelta) if not use.logcb else\
                            10**np.range(np.log10(cllim[0]), np.log10(cllim[-1]*cldelta), np.log10(cldelta) )
            clax = notNone(use.clax, clax_data)
        else:
            clax = ctax

        ## Set cnlim
        cnlim = notNone(use.cnlim, ctlim)

        # You can choose if the color range fit to the range of z.
        norm = Normalize(vmin=cnlim[0], vmax=cnlim[1]) if not use.logcb else LogNorm(vmin=cnlim[0], vmax=cnlim[1])

        ## Make meshgrid
        if (len(x.shape) == 2) and (len(y.shape) == 2):
            xx = x
            yy = y

        if (len(x.shape) == 1) and (xx is None) and (yy is None):
            print("use meshgrid", len(x.shape))
            xx, yy = np.meshgrid(x, y, indexing='xy')

        if mode == "grid" and datatype=="list":
            # Note:
            # this guess of xi, yi relyies on the assumption where
            # grid space is equidistant.
            dx = x[1] - x[0] if len(x) != 1 else y[1] - y[0]
            dy = y[1] - y[0] if len(y) != 1 else x[1] - x[0]
            xxi = np.hstack((xx-dx/2., (xx[:,-1]+dx/2.).reshape(-1, 1)))
            xxi = np.vstack((xxi, xxi[-1]))
            yyi = np.vstack((yy-dy/2., (yy[-1,:]+dy/2.).reshape(1, -1)))
            yyi = np.hstack((yyi, yyi[:,-1].reshape(-1, 1)))
            img = plt.pcolormesh(xxi, yyi, z, norm=norm, rasterized=True, cmap=cmap)

        elif mode == "contourf" and datatype=="list":
            # Note:
            # if len(x) or len(y) is 1, contourf returns an eroor.
            # Instead of this, use "grid" method.
            # print(xx.shape, z.shape)
            img = self.ax.contourf(xx, yy, z, clax, extend='both', cmap=cmap, norm=norm)

        elif mode == "contour" and datatype=="list":
            jM, iM = np.unravel_index(np.argmax(z), z.shape)
            plt.scatter(xx[jM, iM], yy[jM, iM], c='y', s=6, zorder=12)
            img = plt.contour(xx, yy, z, cmap=cmap, levels=clax, linewidths=lw)

        elif mode == "scatter":
            if points is None:
                img = plt.scatter(xx, yy, vmin=cllim[0], vmax=cllim[1],
                                  c=z, s=1, cmap=cmap)
            else:
                img = plt.scatter(points_x, points_y, vmin=cllim[0], vmax=cllim[1],
                                  c=points_z, s=5, cmap=cmap)

        elif mode == "tricontourf" and datatype=="points":
            fillter = np.isfinite(points_z)
            img = plt.tricontourf(points_x[fillter], points_y[fillter], points_z[fillter],
#                                    norm=norm, cmap=cmap)
                                    clax, norm=norm, cmap=cmap, extend="both")
                                    #clax[:-4], norm=norm, cmap=cmap, extend="both")
                                    #clax[:-4], norm=norm, cmap=cmap, vmin=ctax[0], vmax=ctax[-4], extend="both")
        else:
            raise Exception(f"Do not support \"{mode}\" mode with \"{datatype}\" datatype")

        from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext, LogFormatterSciNotation
        self.cbar = self.fig.colorbar(img, ax=self.ax, ticks=ctax, format=ScalarFormatter(),
                     label=use.cbl, pad=0.02)

        if fills is not None:
            for fill in fills:
                i = fill[0]
                self.ax.contourf(xx, yy, fill[1], cmap=cmap, alpha=0.5)

        if Vector is not None:
            uu, vv = Vector# notNone(Vector,self.Vector)
            #seeds_angle = #notNone(seeds_angle, self.seeds_angle)
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
            plt.streamplot(xx_gr, yy_gr, uu_gr, vv_gr,
                           density=n_sline/4, linewidth=0.3*30/n_sline, arrowsize=.3*30/n_sline,
                           color='w', start_points=seed_points)

        if title:
            plt.title(title)

        if use.square:
            self.ax.set_aspect(1)
            #plt.gca().set_aspect('equal', adjustable='box')

        if use.logx or use.logxy:
            plt.xscale("log", nonpositive='clip')

        if use.logx or use.logxy:
            plt.yscale('log', nonpositive='clip')

        if use.leg:
            plt.legend(**self.df.args_leg)

        if use.save:
            #self.save(out, use.dpath_fig, use.fig_ext, use.fn_wrapper)
            self.save(out, dpath_fig=use.dpath_fig, fig_ext=use.fig_ext, fn_wrapper=use.fn_wrapper)
            #self.save(out)

        plt.close()
        plt.clf()

        if result=="fig":
            return self.fig
        elif result=="class":
            return Struct(self)



    #def save(self, out, dpath_fig, fig_ext, fn_wrapper):
    def save(self, out, dpath_fig=None, fig_ext=None, fn_wrapper=None,):
        if None in (dpath_fig, fig_ext, fn_wrapper):
            logger.info("Use values in default values")
            dpath_fig = self.use.dpath_fig
            fig_ext = self.use.fig_ext
            fn_wrapper = self.use.fn_wrapper

        if dpath_fig is not None:
            dp_fig = os.path.abspath(dpath_fig)
        else:
            raise Exception("dpath_fig is None")

        if os.path.exists(dp_fig)==False:
            logger.info("Make figure directory: ", dp_fig)
            self._mkdir(dp_fig)

        if dp_fig is not None:
            savefile = f"{dp_fig}/{fn_wrapper(out)}.{fig_ext}"
            self.fig.savefig(savefile, bbox_inches='tight')
            logger.debug("   Saved {} to {}".format(out, savefile))
        else:
            raise Exception("No self.dp_fig")

    def show(self, out):
        plt.show()

## easy plotting
def plot(ylist, *args, **kwargs):
    import matplotlib
    #matplotlib.use('TkAgg')
    pl = Plotter(show=True, save=False)
    pl.plot(ylist, *args, **kwargs)

def map(*args, **kwargs):
    import matplotlib
    #matplotlib.use('TkAgg')
    pl = Plotter(show=True, save=False)
    pl.map(*args, **kwargs)

#class Fill:
#    @mt.initializer
#    def __init__(self, y1, y2, c=None, alpha=None, zorder=None, mode="shade"):
#                if fillmode=="shade":
#                    ax.fill_between(
#                        x, fil[1], fil[2], facecolor=c[i], alpha=falp_i, zorder=-i-0.5+zorder0)
#                if fillmode=="mesh":
#                    ax.fill_between(
#                        x, fil[1], fil[2], facecolor="none", hatch=['/'*4, r"\ "*4, "|"*4][i], edgecolor=c[i], lw=1 , alpha=1, zorder=-i-0.5+zorder0)
def notNone(*args):
    for a in args:
        if a is not None:
            return a
    else:
        return None

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def generate_cmap_rgb(cmap):
    return [(int(c[1:3], 16)/255., int(c[3:5], 16)/255., int(c[5:7], 16)/255.) for c in cmap]

def darken(cmap_rgb):
    cmap_hsv = mcol.rgb_to_hsv(cmap_rgb)
    cmap_rgb_dark = mcol.hsv_to_rgb([1, 0.7, 0.7] * cmap_hsv)
    return cmap_rgb_dark

c_def = ["#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#34495e",
         "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50"]
c_def_rgb = generate_cmap_rgb(c_def)
c_def_rgb_dark = darken(c_def_rgb)
c_def_dark = c_def_rgb_dark
gray = '#707070'
dgray = '#404040'

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
       framealpha=0.9, edgecolor='#303030', handlelength=2, handleheight=0.5,
       fancybox=False)
golden_ratio = np.array([1, (1+np.sqrt(5))*0.5])
golden_ratio_rev = np.array([(1+np.sqrt(5))*0.5, 1])
mpl.rc('figure', figsize=golden_ratio_rev*5, dpi=160, edgecolor="k")
mpl.rc('savefig', dpi=200, facecolor='none', edgecolor='none')
mpl.rc('path', simplify=True, simplify_threshold=0.2)
mpl.rc('pdf', compression=9, fonttype=3)

