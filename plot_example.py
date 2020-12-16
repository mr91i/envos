
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate, integrate
#
from pmodes.header import inp, dpath_radmc, dpath_fig
import pmodes.myplot as mp
from pmodes import cst, tools

#msg = tools.Message(__file__, debug=False)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from skimage.feature import peak_local_max
import matplotlib.patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox


####################################################################################################

def plot_physical_model(D, r_lim=500, dpath_fig=dpath_fig):

    def slice_at_midplane(tt, *vals_rtp):
        iphi = 0
        if len(vals_rtp) >= 2:
            #return np.array([val_rtp.take(iphi, 2)[tt.take(iphi, 2) == np.pi/2] for val_rtp in vals_rtp])
            #print(vals_rtp[0].shape, vals_rtp[0].take(0, 2).shape, vals_rtp[0])

            return np.array([val_rtp[:, -1, 0] for val_rtp in vals_rtp])
        else:
            return vals_rtp.take(iphi, 2)[tt == np.pi/2]

    ####
    stamp = inp.object_name
    ph_ax = D.ph_ax if len(D.ph_ax) != 1 else np.linspace(-np.pi, np.pi, 91)
    r_mg, th_mg, ph_mg = np.meshgrid(D.r_ax, D.th_ax, ph_ax, indexing='ij')
    R_mg, z_mg = r_mg * [np.sin(th_mg), np.cos(th_mg)]
    x_mg, y_mg = R_mg * [np.cos(ph_mg), np.sin(ph_mg)]
    ux = D.ur * np.cos(ph_mg) - D.uph*np.sin(ph_mg)
    uy = D.ur * np.sin(ph_mg) + D.uph*np.cos(ph_mg)

    r_ew = D.cs * D.t

    pos0_list = [(r_ew, np.pi/180*80), (r_ew, np.pi/180*70), (r_ew, np.pi/180*60), (r_ew, np.pi/180*50)]
    pos_list = [trace_particle_2d_meridional(D.r_ax, D.th_ax, D.ur[:,:,0], D.uth[:,:,0], (0.1*cst.yr, 1e7*cst.yr), pos0) for pos0 in pos0_list]

    plmap = mp.Plotter(dpath_fig, x=R_mg.take(0, 2)/cst.au, y=z_mg.take(0, 2)/cst.au,
                       logx=False, logy=False, logcb=True, leg=False, square=True,
                       xl='Radius [au]', yl='Height [au]', xlim=[0, r_lim], ylim=[0, r_lim],
                       fn_wrapper=lambda s:'map_%s_%s'%(s, stamp),
                       decorator=lambda x: x.take(0,2))
    Vec = np.array([D.uR.take(0, 2), D.uz.take(0, 2)])
    # Density and velocity map
    print(D.rho)
    plmap.map(D.rho, 'rho', ctlim=[1e-20, 1e-16], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec, save=False)
    for pos in pos_list:
        plmap.ax.plot(pos.R/cst.au, pos.z/cst.au, c="orangered", lw=1., marker="o")
    plmap.save("rho_pt")

    plmap.map(D.rho, 'rho_L', ctlim=[1e-20, 1e-16], xlim=[0, 7000], ylim=[0, 7000], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec, save=False)
    for pos in pos_list:
        plmap.ax.plot(pos.R/cst.au, pos.z/cst.au, c="orangered", lw=1., marker="o")
    plmap.save("rho_L_pt")

    # Ratio between mu0 and mu : where these gas come from
    plmap.map(np.arccos(D.mu0)*180/np.pi, 'theta0', ctlim=[0, 90], cbl=r'$\theta_0$ [degree]', div=10, Vector=Vec, n_sl=40, logcb=False)

    plmap.map(D.rho*r_mg**2, 'rhor2', ctlim=[1e9, 1e14], cbl=r'log $\rho*r^2$ [g/cm]', div=10, n_sl=40)

    #from scipy.integrate import simps
    #integrated_value = np.where(r_mg < 300 * cst.au  , D.rho*r_mg**2*np.sin(th_mg), 0)
    #Mtot = 2*np.pi*simps( simps(integrated_value[:,:,0],D.th_ax, axis=1), D.r_ax)
    #print("Mtot is ", Mtot/cst.Msun)
    #plmap.map(D.zeta, 'zeta', ctlim=[1e-2, 1e2], cbl=r'log $\zeta$ [g/cm$^{3}$]', div=6, n_sl=40)


    def zdeco_plane(z):
        if z.shape[2] == 1:
            return np.concatenate([z]*91, axis=2).take(-1,1)
        else:
            return z.take(-1,1)

    Vec = np.array([ux.take(-1, 1), uy.take(-1, 1)])

    plplane = mp.Plotter(dpath_fig, x=x_mg.take(-1, 1)/cst.au, y=y_mg.take(-1, 1)/cst.au,
                       logx=False, logy=False, leg=False, square=True,
                       xl='x [au]', yl='y [au] (observer →)',
                       #xlim=[-1000, 1000], ylim=[-1000, 1000],
                       xlim=[-r_lim, r_lim], ylim=[-r_lim, r_lim],
                       fn_wrapper=lambda s:'plmap_%s_%s'%(s, stamp),
                       decorator=zdeco_plane)

    # Analyze radial profiles at the midplane
    # Slicing
    V_LS = x_mg/r_mg * D.uph - y_mg/r_mg*D.ur

    plplane.map(V_LS/1e5, 'Vls', ctlim=[-2.0, 2.0], cbl=r'$V_{\rm LS}$ [km s$^{-1}$]',
                   div=10, n_sl=20, logcb=False, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    plplane.map(D.rho, 'rho', ctlim=[1e-18, 1e-16], cbl=r'log Density [g/cm$^{3}$]',
                   div=10, n_sl=20, logcb=True, cmap=cm.get_cmap('seismic'), Vector=Vec, seeds_angle=[0,2*np.pi])

    rho0, uR0, uph0, rho_tot0 = slice_at_midplane(
        th_mg, D.rho, D.uR, D.uph, D.rho)

    pl = mp.Plotter(dpath_fig, x=D.r_ax/cst.au, leg=True, xlim=[0, 5000], xl="Radius [au]")

    # Density as a function of distance from the center
    pl.plot([['nH2_env', rho0/cst.mn], ['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'rhos_%s' % stamp, ylim=[1e3, 1e9],  xlim=[10, 10000],
            lw=[3, 3, 6], c=[None, None, 'k'], ls=['--', ':', '-'],
            logxy=True, vl=[2*D.r_CB/cst.au])

    pl.plot(['nH2_env', rho0/cst.mn], #['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'nenv_%s' % stamp, ylim=[1e3, 1e9],  xlim=[10, 10000],
            lw=[3], logxy=True, vl=[2*D.r_CB/cst.au])
    pl.plot(['nH2_env', rho0/cst.mn], #['nH2_disk', (rho_tot0-rho0)/cst.mn], ['nH2_tot', rho_tot0/cst.mn]],
            'nenv_%s_lin' % stamp, ylim=[1e3, 1e9],  xlim=[0, 500],
            lw=[3], logy=True, vl=[2*D.r_CB/cst.au])

    # Make a 'balistic' orbit similar procedure to Oya+2014
    pl.plot([['-uR', -uR0/cst.kms], ['uph', uph0/cst.kms]],
            'v_%s' % stamp, ylim=[-1, 3], xlim=[0, 500], yl=r"Velocities [km s$^{-1}$]",
            lw=[2, 2, 4, 4], ls=['-', '-', '--', '--'])
    pl.plot([['-uR', -uR0/max(np.abs(uph0))], ['uph', uph0/max(np.abs(uph0))]],
            'vnorm_%s' % stamp, ylim=[0, 1.5], x=D.r_ax/D.r_CB, xlim=[0, 3],  yl=r"Velocities [$u_{\phi,\rm CR}$]",
            lw=[2, 2, 4, 4], ls=['-', '-', '--', '--'])

    rhoint = vint(D.rho[:,:,0], R_mg[:,:,0], z_mg[:,:,0], R_mg[:,-1,0], R_mg[:,-1,0])
    pl.plot( ["coldens", rhoint*2], 'coldens_%s' % stamp, ylim=[1e-2, 100], logy=True, xlim=[0, 500], yl=r"Vertical Column Density [g cm $^{-2}$]")



####################################################################################################




def plot_radmc_data(rmc_data):
   xlim = [rmc_data.xauc[0], rmc_data.xauc[-1]]

   pl1d = mp.Plotter(rmc_data.dpath_fig, x=rmc_data.xauc,
                  logx=True, leg=False,
                  xl='Radius [au]', xlim=xlim,
                  fn_wrapper=lambda s:'rmc_%s_1d'%s)

   pl2d = mp.Plotter(rmc_data.dpath_fig, x=rmc_data.RR, y=rmc_data.zz,
                  logx=False, logy=False, logcb=True, leg=False,
                  xl='Radius [au]', yl='Height [au]', xlim=[0, 500], ylim=[0, 500],
                  fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)

   def plot_plofs(d, fname, lim=None, log=True, lb=None):
       pl1d.plot(d[:,-1], fname, ylim=lim, logy=log, yl=lb)
       pl2d.map(d, fname, ctlim=lim, logcb=log, cbl=lb)

   if rmc_data.use_gdens:
       nmin = rmc_data.ndens_mol.min()
       nmax = rmc_data.ndens_mol.max()
       maxlim =  10**(0.5+round(np.log10(nmax)))
       plot_plofs(rmc_data.ndens_mol, "nden", lim=[maxlim*1e-3, maxlim], lb=r"Number density [cm$^{-3}$]")

   if rmc_data.use_gtemp:
       pl1d.plot(rmc_data.gtemp[:,-1], "temp", ylim=[1,1000], logy=True, yl='Temperature [K]')
       pl1d.plot(rmc_data.gtemp[:, 0], "temp_pol", ylim=[1,1000], logy=True, yl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_in", ctlim=[0,200], xlim=[0,100],ylim=[0,100], logcb=False, cbl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_out", ctlim=[0,100], logcb=False, cbl='Temperature [K]')
       pl2d.map(rmc_data.gtemp, "temp_L", ctlim=[0,40], xlim=[0,7000],ylim=[0,7000], logcb=False, cbl='Temperature [K]')
       pl2d_log = mp.Plotter(rmc_data.dpath_fig, x=rmc_data.RR, y=rmc_data.zz,
                  logx=True, logy=True, logcb=True, leg=False,
                  xl='log Radius [au]', yl='log Height [au]', xlim=[1, 1000], ylim=[1, 1000],
                  fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)

       pl2d_log.map(rmc_data.gtemp, "temp_log", ctlim=[10**0.5, 10**2.5], cbl='log Temperature [K]')

   if rmc_data.use_gvel:
       lb_v = r"Velocity [km s$^{-1}$]"
       plot_plofs(-rmc_data.vr/1e5, "gvelr", lim=[0,5], log=False, lb=lb_v)
       plot_plofs(rmc_data.vt/1e5, "gvelt", lim=[-5,5], log=False, lb=lb_v)
       plot_plofs(np.abs(rmc_data.vp)/1e5, "gvelp", lim=[0,5], log=False, lb=lb_v)

   if rmc_data.use_gdens and rmc_data.use_gtemp:
       plot_plofs(rmc_data.t_dest/rmc_data.t_dyn, "tche", lim=[1e-3,1e3], lb="CCH Lifetime/Dynamical Timescale")

   if rmc_data.opac!="":
       with open(f"{dpath_radmc}/dustkappa_{rmc_data.opac}.inp", mode='r') as f:
           read_data = f.readlines()
           mode = int(read_data[0])
       if mode==2:
           lam, kappa_abs, kappa_sca = np.loadtxt(f"{dpath_radmc}/dustkappa_{rmc_data.opac}.inp", skiprows=2).T
       elif mode==3:
           lam, kappa_abs, kappa_sca, _ = np.loadtxt(f"{dpath_radmc}/dustkappa_{rmc_data.opac}.inp", skiprows=3).T

       mp.Plotter(rmc_data.dpath_fig).plot([["ext",kappa_abs+kappa_sca],["abs", kappa_abs],["sca",kappa_sca]], "dustopac",
           x=lam, xlim=[0.03,3e4], ylim=[1e-4,1e6], logx=True, logy=True,
           xl=r'Wavelength [$\mu$m]', yl=r"Dust Extinction Opacity [cm$^2$ g$^{-1}$]",
           ls=["--"], c=["k"], lw=[3,2,2])

#   if 1:
#       pl2d.map(D.rho, 'rho_L', ctlim=[1e-20, 1e-16], xlim=[0, 5000], ylim=[0, 5000], cbl=r'log Density [g/cm$^{3}$]', div=10, n_sl=40, Vector=Vec, save=False)
#       pl2d.ax.plot(rmc_data.pos_list[0].R/cst.au, rmc_data.pos_list[0].z/cst.au, c="orangered", lw=1.5, marker="o")
#       pl2d.save("rho_L_pt")

   if rmc_data.plot_tau:
       pl1d = mp.Plotter(rmc_data.dpath_fig, x=rmc_data.imx/cst.au,
                      logx=True, leg=False,
                      xl='Radius [au]', xlim=xlim,
                      fn_wrapper=lambda s:'rmc_%s_1d'%s)

       pl2d = mp.Plotter(rmc_data.dpath_fig, x=rmc_data.imx/cst.au, y=rmc_data.imy/cst.au,
                      logx=False, logy=False, logcb=True, leg=False,
                      xl='Radius [au]', yl='Height [au]', xlim=[-500/2, 500/2], ylim=[-500/2, 500/2],
                      fn_wrapper=lambda s:'rmc_%s_2d'%s, square=True)
       plot_plofs(rmc_data.tau/cst.au, "tau", lim=[1e-2, 1000], lb=r"tau")

def vint(value_rt, R_rt, z_rt, R_ax, z_ax, log=False):
    points = np.stack((R_rt.flatten(), z_rt.flatten()),axis=-1)
    npoints = np.stack( np.meshgrid(R_ax, z_ax),axis=-1 )
    if log:
        fltr = np.logical_and.reduce( ( [ np.all(np.isfinite(a)) for a in np.log10(points) ], np.isfinite(np.log10(value_rt))   ))
        fltr = fltr.flatten()
        v = value_rt.flatten()
        ret = 10**interpolate.griddata(np.log10(points[fltr]), np.log10(v[fltr] ), np.log10(npoints), method='linear')
    else:
        ret = interpolate.griddata(points, value_rt.flatten(), npoints, method='linear')
    s = np.array([ integrate.simps(r, z_ax) for r in np.nan_to_num(ret) ])
    return s

def trace_particle_2d_meridional(r_ax, th_ax, vr, vth, t_span, pos0):
    # There are some choice for coordination, but i gave up genelarixation for the code simplicity.
    # input axis : rth
    # velocity   : rth
    # return pos : rth

    vr_field = interpolate.RegularGridInterpolator((r_ax, th_ax), vr, bounds_error=False, fill_value=None)
    vth_field = interpolate.RegularGridInterpolator((r_ax, th_ax), vth, bounds_error=False, fill_value=None)

    def func(t, pos, hit_flag=0):
        if pos[0] > r_ax[-1]:
            raise Exception(f"Too large position. r must be less than {r_ax[-1]/cst.au} au.")

        if hit_midplane(t, pos) < 0:
            hit_flag = 1
        r, th = pos[0], pos[1]
        vr = vr_field((r, th))
        vth = vth_field((r, th))
        return np.array([vr, vth/r])

    def hit_midplane(t, pos):
        return np.pi/2 - pos[1]
    hit_midplane.terminal = True

    t_trace = np.logspace(np.log10(t_span[0]), np.log10(t_span[-1]), 600)
    if pos0[0] > r_ax[-1]:
        print(f"Too large position:r0 = {pos0[0]/cst.au} au. r0 must be less than {r_ax[-1]/cst.au} au. I use r0 = {r_ax[-1]/cst.au} au instead of r0 = {pos0[0]/cst.au} au")
        pos0 = [r_ax[-1], pos0[1]]

    #pos = integrate.solve_ivp(func, t_span, pos0, method='BDF', events=hit_midplane, t_eval=t_trace[1:-1])
    pos = integrate.solve_ivp(func, t_span, pos0, method='RK45', events=hit_midplane, rtol=1e-8)
    pos.R = pos.y[0] * np.sin(pos.y[1])
    pos.z = pos.y[0] * np.cos(pos.y[1])
    return pos


####################################################################################################



def draw_center_line(ax):
    draw_cross_pointer(ax, 0, 0, c='k', lw=2, s=10, alpha=1, zorder=1)

def draw_cross_pointer(ax, x, y, c='k', lw=2, s=10, alpha=1, zorder=1):
    ax.axhline(y=y, lw=lw, ls=":", c=c, alpha=alpha, zorder=zorder)
    ax.axvline(x=x, lw=lw, ls=":", c=c, alpha=alpha, zorder=zorder)
    ax.scatter(x, y, c=c, s=s, alpha=alpha, linewidth=0, zorder=zorder)

def draw_beamsize(ax, mode, beam_a_au, beam_b_au, beam_pa_deg, pangle_deg=None, vreso_kms=None, with_box=False):
    print( beam_a_au, beam_b_au, vreso_kms)

    if mode=="PV":
        cross_angle = (beam_pa_deg - pangle_deg) * np.pi/180
        beam_crosslength_au = 1/np.sqrt( (np.cos(cross_angle)/beam_a_au)**2 + (np.sin(cross_angle)/beam_b_au)**2)
        beamx = beam_crosslength_au
        beamy = vreso_kms
    elif mode=="mom0":
        beamx = beam_a_au
        beamy = beam_b_au

    if with_box:
        e1 = matplotlib.patches.Ellipse(xy=(0,0), width=beamx, height=beamy, lw=1, fill=True, ec="k", fc="0.6")
    else:
        e1 = matplotlib.patches.Ellipse(xy=(0,0), width=beamx, height=beamy, lw=0, fill=True, ec="k", fc="0.7", alpha=0.6)

    box = AnchoredAuxTransformBox(ax.transData, loc='lower left', frameon=with_box, pad=0., borderpad=0.4)
    box.drawing_area.add_artist(e1)
    box.patch.set_linewidth(1)
    ax.add_artist(box)

def plot_mom0_map(obsdata, pangle_deg=None,  poffset_au=None, dpath_fig=dpath_fig, n_lv=10):

    def position_line(xau, pangle_deg, poffset_au=0):
        pangle_rad = pangle_deg * np.pi/180
        pos_x = xau*np.cos(pangle_rad) - poffset_au*np.sin(pangle_rad)
        pos_y = xau*np.sin(pangle_rad) + poffset_au*np.sin(pangle_rad)
        return np.stack([pos_x, pos_y], axis=-1)

    pltr = mp.Plotter(dpath_fig, x=obsdata.xau, y=obsdata.yau)
    pltr.map(obsdata.Imom0, out="mom0map",
             xl="Position [au]", yl="Position [au]", cbl=r'Intensity [Jy pixel$^{-1}$ ]',
             div=n_lv, mode='grid', ctlim=[0,obsdata.Imom0.max()], square=True, save=False)

    draw_center_line(pltr.ax)

    if pangle_deg is not None:
        pline0 = position_line(obsdata.xau, pangle_deg)
        plt.plot(pline0[0], pline0[1], ls='--', c='w', lw=1)
        if poffset_au is not None :
            pline = position_line(obsdata.xau, pangle_deg, poffset_au=poffset_au)
            plt.plot(pline[0], pline[1], c='w', lw=1)

    if obsdata.convolution:
        #draw_beamsize(pltr, obsdata.conv_info, mode="mom0")
        draw_beamsize(pltr.ax, "mom0", obsdata.beam_a_au, obsdata.beam_b_au, obsdata.beam_pa_deg)


    pltr.save("mom0map")


def plot_chmap(obsdata, dpath_fig=dpath_fig, n_lv=20):
    cbmax = obsdata.Ippv_max
    pltr = mp.Plotter(dpath_fig, x=obsdata.xau, y=obsdata.yau,
                      xl="Position [au]", yl="Position [au]",
                      cbl=r'Intensity [Jy pixel$^{-1}$ ]')

    for i in range(obsdata.Nz):
        pltr.map(Ippv[i], out="chmap_{:0=4d}".format(i),
                 n_lv=n_lv, ctlim=[0, cbmax], mode='grid',
                 title="v = {:.3f} km/s".format(obsdata.vkms[i]) , square=True)


def plot_pvdiagram(PV, dpath_fig=dpath_fig, out='pvd', n_lv=5, Mstar_Msun=None, rCR_au=None, f_crit=None, mass_ip=False, mass_vp=False, mapmode='grid', oplot={}):

    def find_local_peak_position(x, y, i):
        if (2 <= i <= len(x)-3):
            grad_y = interpolate.InterpolatedUnivariateSpline(
                    x[i-2:i+3], y[i-2:i+3]).derivative(1)
            return optimize.root(grad_y, x[i]).x[0]
        else:
            return np.nan

    def get_localpeak_positions(x, y, min_distance=3, threshold_abs=None):
        maxis = [find_local_peak_position(x, y, mi)
                 for mi
                 in peak_local_max(y, min_distance=min_distance, threshold_abs=threshold_abs)[:, 0]]
        return np.array(maxis)

    def get_maximum_position(x, y):
        return find_local_peak_position(x, y, np.argmax(y))

  #  if normalize == "peak":
  #      if Imax > 0:
  #          Ipv /= Imax
  #          ctlim = [1/n_lv, 1]
  #          n_lv -= 1
  #      else:
  #          ctlim = [0, 0.1]
  #      unit = r'[$I_{\rm max}$]'
  #  else:
  #      ctlim = [0, Imax] if not self.Imax else [0, self.Imax]
    Ipv = PV.Ipv
    xau = PV.xau
    xas = xau/PV.dpc
    vkms = PV.vkms
    dpc = PV.dpc
    ctlim = [0, np.max(Ipv)]

    print(xau)
    pltr = mp.Plotter(dpath_fig, x=PV.xau, y=PV.vkms, xlim=[-700, 700], ylim=[-4, 4])

    pltr.map(z=Ipv, mode=mapmode, ctlim=ctlim,
             xl="Angular Offset [au]", yl=r"Velocity [km s$^{-1}$]",
             cbl=f'Intensity {PV.unit_I}',
             lw=1.5, div=n_lv, save=False, clabel=True)

    draw_center_line(pltr.ax)
    pltr.ax.minorticks_on()
    pltr.ax.tick_params("both",direction="inout")

    pltr.ax2 = pltr.ax.twiny()
    pltr.ax2.minorticks_on()
    pltr.ax2.tick_params("both", direction="inout")
    pltr.ax2.set_xlabel("Angular Offset [arcsec]")
    pltr.ax2.set_xlim(np.array(pltr.ax.get_xlim())/dpc)

    l = np.sqrt(cst.G*Mstar_Msun*cst.Msun*rCR_au*cst.au)
    a = (2*xau/rCR_au)

    overplot = {"KeplerRotation":False, "ire_fit":False, "LocalPeak_Pax": False, "LocalPeak_Vax":False, "LocalPeak_2D":False}
    overplot.update(oplot)
    if overplot['ire_fit']:
        pltr.ax.plot(xau, 2*l/rCR_au*(2*xau.clip(0)/rCR_au)**(-2/3) * 1/cst.kms, c="hotpink", ls=":", lw=1)

    if overplot['KeplerRotation']:
        vKep = np.sqrt(cst.G*Mstar_Msun*cst.Msun/(xau*cst.au)) * 1/cst.kms
        pltr.ax.plot(xau, vKep, c="cyan", ls=":", lw=1)

    if overplot['LocalPeak_Pax']:
        for Iv, v_ in zip(Ipv.transpose(0, 1), vkms):
            for xM in get_localpeak_positions(xau, Iv, threshold_abs=np.max(Ipv)*1e-10):
                pltr.ax.plot(xM, v_, c="red", markersize=1, marker='o')

    if overplot['LocalPeak_Vax']:
        for Ip, x_ in zip(Ipv.transpose(1, 0), xau):
            for vM in get_localpeak_positions(vkms, Ip, threshold_abs=np.max(Ipv)*1e-10):
                pltr.ax.plot(x_, vM, c="blue", markersize=1, marker='o')

    if overplot['LocalPeak_2D']:
        for jM, iM in peak_local_max(Ipv, num_peaks=4, min_distance=10):#  min_distance=None):
            pltr.ax.scatter(xau[iM], vkms[jM], c="k", s=20, zorder=10)
            print(f"Local Max:   {xau[iM]:.1f} au  , {vkms[jM]:.1f}    km/s  ({jM}, {iM})")

        del jM, iM

    def calc_M(xau, vkms, fac=1):
        # calc xau*cst.au * (vkms*cst.kms)**2 / (cst.G*cst.Msun)
        return 0.001127 * xau * vkms**2 * fac

    if mass_ip:
        ## M_ipeak
        jmax, imax = Ipv.shape
        i0 = imax//2
        j0 = jmax//2
        jpeak, ipeak = peak_local_max(Ipv[j0:,i0:], num_peaks=1)[0]
        xau_peak = xau[i0+ipeak]
        vkms_peak = vkms[j0+jpeak]

        vp_peaks = [(vkms[jM], xau[iM]) for jM, iM in peak_local_max(Ipv[i0:,j0:], num_peaks=1)]

        print(peak_local_max(Ipv[i0:,j0:], num_peaks=1))
        i = np.argmax( np.sign(vp_peaks[:,0])*vp_peaks[:,0] * vp_peaks[:,1]  )
        vkms_peak, xau_peak = vp_peaks[i]

        M_CR = calc_M(abs(xau_peak), vkms_peak, fac=1)
        M_CR_vpeak =  M_CR / np.sqrt(2)
        draw_cross_pointer(xau_peak/dpc, vkms_peak, mp.c_def[1])
        txt_Mip = rf"$M_{{\rm ip}}$={M_CR:.3f}"

    if mass_vp:
        ## M_vpeak
        x_vmax, I_vmax = np.array([[get_maximum_position(obsdata.xau, Iv), np.max(Iv)] for Iv in Ipv.transpose(0,1)]).T

        if (np.min(I_vmax) < f_crit*ctlim[1]) and (f_crit*ctlim[1] < np.max(I_vmax)):
            v_crit = tools.find_roots(vkms, I_vmax, f_crit*ctlim[1])
        else:
            v_crit = tools.find_roots(vkms, I_vmax, f_crit*np.max(Ippv) )

        if len(v_crit) == 0 :
            v_crit = [vkms[0]]

        x_crit = tools.find_roots(x_vmax, vkms, v_crit[0])
        M_CB = calc_M(abs(x_crit[0]), v_crit[0], fac=1/2)
        draw_cross_pointer(x_crit[0]/dpc, v_crit[0], mp.c_def[0])
        txt_Mvp = rf"$M_{{\rm vp, {f_crit*100}\%}}$={M_CB:.3f}"

    if mass_ip or mass_vp:
        plt.text(0.95, 0.05, txt_Mip+'\n'+txt_Mvp, transform=pltr.ax.transAxes,
                  ha="right", va="bottom", bbox=dict(fc="white", ec="black", pad=5))

    try:
        draw_beamsize(pltr.ax, "PV", PV.beam_a_au, PV.beam_b_au, PV.beam_pa_deg, PV.pangle_deg, PV.vreso_kms)
    except:
        pass

    pltr.save(out)


