#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import radmc3dPy.analyze as rmca
import radmc3dPy.image as rmci
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool
from header import inp, dn_home, dn_radmc, dn_fig
import cst
import mytools
import logging
logger = logging.getLogger(__name__)

#####################################

def main():
    osim = ObsSimulator(dn_radmc, dn_fits=dn_radmc, **vars(inp.sobs))
    osim.observe()

class ObsSimulator():  # This class returns observation data
    def __init__(self, dn_radmc, dn_fits=None, filename="obs", dpc=None, iline=None,
                 sizex_au=None, sizey_au=None,
                 pixsize_au=None,
                 vwidth_kms=None, dv_kms=None,
                 incl=None, phi=None, posang=None,
                 rect_camera=True, omp=True, n_thread=1,**kwargs):

        for k, v in locals().items():
            if k != 'self' and k != 'kwargs':
                setattr(self, k, v)
                logger.info(k.ljust(20)+"is {:20}".format(v if v is not None else "None"))

        self.linenlam = 2*int(round(self.vwidth_kms/self.dv_kms)) + 1
        self.set_camera_info()
        logger.debug("Total cell number is {} x {} x {} = {}".format(
              self.npixx, self.npixy,self.linenlam,
              self.npixx*self.npixy*self.linenlam))

    def set_camera_info(self):
        if self.rect_camera and (self.sizex_au != self.sizey_au):
            #########################################
            # Note: Before using rectangle imaging, #
            # you need to fix a bug in radmc3dpy.   #
            # Also, dx needs to be equal dy.        #
            #########################################
            if self.sizey_au == 0:
                self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
                self.zoomau_y = [-self.pixsize_au/2, self.pixsize_au/2]
                self.npixx = int(round(self.sizex_au/self.pixsize_au)) # // or int do not work to convert into int.
                self.npixy = 1
            else:
                self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
                self.zoomau_y = [-self.sizey_au/2, self.sizey_au/2]
                self.npixx = int(round(self.sizex_au/self.pixsize_au))
                self.npixy = int(round(self.sizey_au/self.pixsize_au))

        else:
            self.zoomau_x = [-self.sizex_au/2, self.sizex_au/2]
            self.zoomau_y = [-self.sizex_au/2, self.sizex_au/2]
            self.npixx = int(round(self.sizex_au/self.pixsize_au))
            self.npixy = self.npixx

        dx = (self.zoomau_x[1] - self.zoomau_x[0])/self.npixx
        dy = (self.zoomau_y[1] - self.zoomau_y[0])/self.npixy
        if dx != dy:
            raise Exception("dx is not equal to dy")

    @staticmethod
    def find_proper_nthread( n_thread, n_divided):
        return max([i for i in range(n_thread, 0, -1)
                                if n_divided % i == 0])

    @staticmethod
    def divide_threads(n_thread, n_divided):
        ans = [n_divided//n_thread]*n_thread
        rem = n_divided % n_thread
        nlist = [ (n_thread-i//2 if i%2 else i//2) for i in range(n_thread)] #[ ( i if i%2==0 else n_thread -(i-1) ) for i in range(n_thread) ]
        for i in range(rem):
            ii = (n_thread-1 - i//2) if i%2 else i//2
            ans[ii] += 1
        return ans

    @staticmethod
    def calc_thread_seps(calc_points, thread_divs):
        ans = []
        sum_points = 0
        for npoints in thread_divs:
            ans.append( [calc_points[sum_points] , calc_points[sum_points+npoints-1]] )
            sum_points += npoints
        return np.array(ans)

    def observe(self):
        common = "incl %d phi %d posang %d" % (
            self.incl, self.phi, self.posang)
        option = "noscat nostar nodust" #fluxcons doppcatch"
        camera = "npixx {} npixy {} ".format(self.npixx, self.npixy)
        camera += "zoomau {:g} {:g} {:g} {:g} ".format(*(self.zoomau_x+self.zoomau_y))
        line = "iline {:d}".format(self.iline)
        v_calc_points = np.linspace( -self.vwidth_kms, self.vwidth_kms, self.linenlam )
        vseps = np.linspace( -self.vwidth_kms-self.dv_kms/2, self.vwidth_kms+self.dv_kms/2, self.linenlam+1   )

        if self.omp and (self.n_thread > 1):
            n_points = self.divide_threads(self.n_thread, self.linenlam )
            v_thread_seps = self.calc_thread_seps(v_calc_points, n_points)
            v_center = [ 0.5*(v_range[1] + v_range[0] ) for v_range in v_thread_seps ]
            v_width = [ 0.5*(v_range[1] - v_range[0] ) for v_range in v_thread_seps ]
            logger.info(f"All calc points: {format_array(v_calc_points)}")
            logger.info("Calc points in each thread:")
            for i, (vc, vw, ncp) in enumerate(zip(v_center, v_width, n_points)):
                logger.info(f"    {i}th thread: {format_array(np.linspace(vc-vw, vc+vw, ncp))}")

            def cmd(p):
                freq = "vkms {:g} widthkms {:g} linenlam {:d} ".format(
                        v_center[p], v_width[p], n_points[p])
                return " ".join(["radmc3d image", line, freq, camera, common, option])

            args = [ (self, (p, 'proc'+str(p), cmd(p)) ) for p in range(self.n_thread)]
            rets = Pool(self.n_thread).map(call_subcalc, args)

            for i, r in enumerate(rets):
                logger.debug(f"The{i}th return")
                for k, v in r.__dict__.items():
                    if isinstance(v, (np.ndarray)):
                        logger.debug("{}: shape is {}, range is [{}, {}]".format(k, v.shape, np.min(v), np.max(v)))
                    else:
                        logger.debug(f"{k}: {v}")

            data = rets[0]
            for ret in rets[1:]:
                data.image = np.append(data.image, ret.image, axis=2)
                data.imageJyppix = np.append(data.imageJyppix, ret.imageJyppix, axis=2)
                data.freq = np.append(data.freq, ret.freq, axis=-1)
                data.wav = np.append(data.wav, ret.wav, axis=-1)
                data.nfreq += ret.nfreq
                data.nwav += ret.nwav
            self.data = data
        else:
            freq = f"widthkms {self.vwidth_kms} linenlam {self.linenlam:d} "
            cmd = " ".join(["radmc3d image", line, freq, camera, common, option])
            logger.info(f"command is {cmd}")
            os.chdir(self.dn_radmc)
            subprocess.call(cmd, shell=True)
            self.data = rmci.readImage()

        if np.max(self.data.image) == 0:
            logger.warning("Zero image !")
#            raise Exception("zero image...")

        freq0 = (self.data.freq[0] + self.data.freq[-1])*0.5
        dfreq = self.data.freq[1] - self.data.freq[0]
        vkms = np.round(mytools.freq_to_vkms(freq0, self.data.freq-freq0), 8)

        logger.info("x_au is " + format_array(self.data.x/cst.au) )
        logger.info("v_kms is " + format_array(vkms) )
        self.save_instance()
        self.save_fits()

        return self.data

    def subcalc(self, args):
        p, dn, cmd = args
        logger.debug("execute: " + cmd)
        dpath_sub = self.dn_radmc + '/' + dn
        if not os.path.exists(dpath_sub):
            os.makedirs(dpath_sub)
        os.system("cp %s/{*.inp,*.dat} %s/" % (self.dn_radmc, dpath_sub))
        os.chdir(dpath_sub)
        if p == 1:
            logger.info("Calculating ray-tracing with RADMC3d...")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Log from RADMC3d")
                subprocess.call(cmd, shell=True)
            else:
                subprocess.call(cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            subprocess.call(cmd, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return rmci.readImage()

    def save_fits(self):
        fp_fitsdata = self.dn_fits+'/'+self.filename+'.fits'
        if os.path.exists(fp_fitsdata):
            logger.info(f"remove old fits file: {fp_fitsdata}")
            os.remove(fp_fitsdata)
        self.data.writeFits(fname=fp_fitsdata, dpc=self.dpc)
        logger.info(f"Saved fits file: {fp_fitsdata}")

    def save_instance(self):
        pd.to_pickle(self, self.dn_fits+'/'+self.filename+'.pkl')

    def read_instance(self):
        instance = pd.read_pickle(self.dn_fits+'/'+self.filename+'.pkl')
        for k,v in instance.__dict__.items():
            setattr(self, k, v)

def call_subcalc(args_list):
    "This fucntion is implemented for fucking Python2."
    instance, args = args_list
    return getattr(instance, "subcalc")(args)

def format_array(array):
    return f"[{min(array):.4g}:{max(array):.4g}] with delta = {abs(array[1]-array[0]):.4g} and N = {len(array)}"

if __name__ == '__main__':
    main()
