#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import subprocess
from radmc3dPy.natconst import *
from radmc3dPy.analyze import *
from radmc3dPy.image import *
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
dn_radmc = os.path.dirname(os.path.abspath(__file__)) + '/'
dn_home = os.path.abspath(os.path.dirname(__file__)+"/../../")
print("Execute %s:\n"%__file__)
sys.path.append(dn_home)

# Global parameters
iline = 2
incl = 90  # 90
phi = 0
posang = 0
n_thread = 16
rectangle_camera = 1
multi_threads = 1
fitsdir = dn_home+"/calc/sobs"

def make_fits_data():
	global n_thread
	widthkms = 3
	linenlam = 60  # ~ 0.1km/s
	sizeau = 1960  # 1400 #
	npix = 100	# ~ 0.1''
	common = "incl %d phi %d posang %d" % (incl, phi, posang)
	option = " "

	if rectangle_camera:
		##
		## Note: Before use rectangle imaging, 
		##		 you need to fix a bug in a radmc3dpy.
		## 
		Lh = sizeau / 2
		npix = [npix, 10]
		zoomau = [-Lh, Lh, -Lh * npix[1]/float(npix[0]),
				  Lh * npix[1] / float(npix[0])]
		camera = "npixx {:d} npixy {:d} ".format(*npix) + "zoomau {:f} {:f} {:f} {:f} truepix ".format(*zoomau)

	if multi_threads:
		from multiprocessing import Pool
		for i in range(n_thread, 0, -1):
			if linenlam % i == 0:
				n_thread = i
				break
		v_ranges = np.linspace(-widthkms, widthkms, n_thread + 1)
		dv = 2 * widthkms / linenlam

		def cmd(p): return "radmc3d image iline {:d} vkms {} widthkms {} linenlam {:d} ".format(
			iline, 0.5 * (v_ranges[p + 1] + v_ranges[p]), 
			0.5 * (v_ranges[p + 1] - v_ranges[p]), 
			int(linenlam / float(n_thread))) + camera + common + option
		rets = Pool(n_thread).map(subcalc, [('proc' + str(p), cmd(p)) for p in range(n_thread)])
		data = rets[0]
		for ret in rets[1:]:
			data.image = np.append(data.image, ret.image[:, :, 1:], axis=2)
			data.imageJyppix = np.append(data.imageJyppix, ret.imageJyppix[:, :, 1:], axis=2)
			data.freq = np.append(data.freq, ret.freq[1:], axis=-1)
			data.nfreq += ret.nfreq - 1
			data.wav = np.append(data.wav, ret.wav[1:], axis=-1)
			data.nwav += ret.nwav - 1
	else:
		cmd = "radmc3d image " + line + camera + common + option
		subprocess.call(cmd, shell=True)
		data = readImage()

	data.writeFits(fname=fitsdir+'/obs.fits', dpc=140)


def subcalc(args):
	dn , cmd = args
	print(cmd)
	dn = dn_radmc + dn
	if not os.path.exists(dn):
		os.makedirs(dn)
#	os.system("rm %s/*"%dn)
	os.system("cp %s/{*.inp,*.dat} %s/"%(dn_radmc,dn))
	os.chdir(dn)
	subprocess.call(cmd,shell=True)
	return readImage()

if __name__ == '__main__':
	make_fits_data()
