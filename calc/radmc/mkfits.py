#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import subprocess, argparse
from radmc3dPy.natconst import *
from radmc3dPy.analyze import *
from radmc3dPy.image import *
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

dn_here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
dn_home = os.path.abspath(dn_here + "/../../")
sys.path.append(dn_home)
dn_radmc = dn_home + '/calc/radmc/'
print("Execute %s:\n"%__file__)


parser = argparse.ArgumentParser(description='This code makes a fits file by synthetic observation.')
parser.add_argument('-d','--debug',action='store_true')
parser.add_argument('--incl',default=85,type=float)
parser.add_argument('--phi',default=0)
parser.add_argument('--posang',default=180)
parser.add_argument('--dpc',default=137)
parser.add_argument('--iline',default=2)
parser.add_argument('--n_thread',default=16)
parser.add_argument('--rect_camera',default=True)
parser.add_argument('--multi_threads',default=True)

args = parser.parse_args()

# Global parameters
n_thread = args.n_thread
fitsdir = dn_home+"/calc/sobs"

def make_fits_data():
	global n_thread
	widthkms = 3
	dvkms = 0.05
	linenlam = 2*widthkms/dvkms  # ~ 0.1km/s
	sizeau = args.dpc*15 #
	dxmasec = 0.05 # marcsec
	npix = int(sizeau/(args.dpc*dxmasec)) # ~ 0.1''
	common = "incl %d phi %d posang %d " % (args.incl, args.phi, args.posang)
	option = "noscat "
	for k, v in locals().items():
		print("{} is {}".format(k,v))	

	if args.rect_camera:
		##
		## Note: Before use rectangle imaging, 
		##		 you need to fix a bug in a radmc3dpy.
		## 
		Lh = sizeau / 2.0
		npixx = npix ## dx = L/npix
		npixy = 1
		pix_yx = npixy/float(npixx)
		zoomau = np.array([ -1, 1, -pix_yx, pix_yx ]) * Lh

		camera = "npixx {:d} npixy {:d} ".format(npixx,npixy) + "zoomau {:f} {:f} {:f} {:f} truepix ".format(*zoomau)

	if args.multi_threads:
		from multiprocessing import Pool
		for i in range(n_thread, 0, -1):
			if linenlam % i == 0:
				n_thread = i
				break
		v_ranges = np.linspace(-widthkms, widthkms, n_thread + 1)
		dv = 2 * widthkms / linenlam

		def cmd(p): return "radmc3d image iline {:d} vkms {} widthkms {} linenlam {:d} ".format(
			args.iline, 0.5 * (v_ranges[p + 1] + v_ranges[p]), 
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

	data.writeFits(fname=fitsdir+'/obs.fits', dpc=args.dpc)


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
