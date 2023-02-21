#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import envos
import envos.nconst as nc
import numpy as np
import os
from itertools import product
from multiprocessing import Pool
import logging

from myplot import mpl_setting
mpl_setting.set_rc(lw=1.2, bold=False, fontsize=12)

#envos.log.set_level("debug")
#print(envos.log.loggers)
#lgr = envos.log.set_logger("envos.test")
#lgr.info("Test 1")

#envos.log.set_file()

def main(conf):
    #do_fiducial_run(conf)
    do_rshock_test(conf)
    #t2=unit2, plot_obspv(conf)
    #do_debug_run(conf)
    #do_parameter_study(conf)
    #exit()
    #do_pmap_survey_for_mass_estimate(conf)
    #do_pmap_survey_for_cube_correlation(conf)
    compare_bestmodel_obspvs(conf)
    # do_ring_test()
    #plot_obspv(conf)
    # do_disk_envelope_runs()
    # do_ps_survays()


#####################################################################
# Doing functions                                                   #
#####################################################################
#
# Fiducial run
#
def do_fiducial_run(conf):
    _conf = conf.replaced(fig_dir=f"run/fig_fid", n_thread=16,pixsize_au=5) #, dr_to_r=0.2, nphot=1e4)
    #_conf = conf.replaced(fig_dir=f"run/fig_fid", n_thread=1, dr_to_r=0.2, nphot=1e4)
    synobs(
        _conf,
        filename=f"lobs_fid.pkl",
        calc_model=0,
        calc_obs=0,
        plot_model_structure=1,
        #calc_obs=False,

    )
    exit()

def do_debug_run(conf):
    #label = f"{model}_M{M}_cr{CR}_i{i}"
    label="UCM_M0.07174726045295622_cr40.000000000000014_i60.0"
    M = 0.0717472604529562
    CR = 40.000
    i=60.0
    model="UCM"
    Vmax = np.sqrt(2 * nc.G * M * nc.Msun / (CR * nc.au)) / 1e5
    _conf = conf.replaced(
        Ms_Msun=M,
        CR_au=CR,
        incl=i,
        inenv=model,
        run_dir=f"./pmap2/run_" + label,
        n_thread=1,
        dr_to_r=0.01,
        size_au=4 * CR,
        vfw_kms=1.5 * Vmax * 2,
    )
    synobs(
        _conf,
        vkms_pvd=Vmax * 1.5,
        xlim_pvd=CR * 2,
        calc_model=0,
        calc_obs=False,
        plot_model_structure=0
    )

def do_parameter_study(config):

    def _obs(config, label, refcube=None, **kwargs):
        _conf = config.replaced(
            run_dir=f"./run/run_" + label,
            n_thread=16,
            #dr_to_r=0.05,
            #nphot=1e6,
        )
        synobs(
            _conf,
            filename=f"lobs_{label}.pkl",
            skip_if_exist=0, #True,
            #read_mg=1,
            #calc_model=0,
            calc_model=0,
            calc_obs=0,
            plot_model_structure=1,
            refdata=refcube,
            **kwargs,
        )

    #config = config.replaced(n_thread=16)
    #_obs(config, "fid")

    #_obs(config.replaced(incl=30), f"i{30}")
    #exit()

    refcube = envos.read_obsdata(envos.gpath.home_dir / "run" / "lobs_fid.pkl")
    refcube.norm_I("max")
    _obs(config.replaced(inenv="Simple", cavangle_deg=80), "SB", refcube)
    #exit()

    """
    for T0 in [10, 30, 100]:
        Tfunc = lambda m: T0 * (m.rr/10/envos.nc.au)
        _obs(config, f"T{T0}-p0", refcube, Tfunc=Tfunc)

    for p in [-0.5, -1]:
        Tfunc = lambda m: 30 * (m.rr/10/envos.nc.au)**p
        _obs(config, f"T30-p{p}", refcube, Tfunc=Tfunc)
    """

    for M in [0.1, 0.3]: # 0.15
        _obs(config.replaced(Ms_Msun=M), f"M{M}", refcube)

    for cr in [100, 300]:
        _obs(config.replaced(CR_au=cr), f"cr{cr}", refcube)

    for incl in [30, 60]:
        _obs(config.replaced(incl=incl), f"i{incl}", refcube)

    for ca in [0, 80]:
        _obs(config.replaced(cavangle_deg=ca), f"cav{ca}", refcube)

#    for bs in [50, 200]:
#        _obs(config.replaced(beam_maj_au=bs, beam_min_au=bs), f"bs{bs}")

#    for vr in [0.075, 0.3]:
#        _obs(config.replaced(vreso_kms=vr), f"vr{vr}")


def do_pmap_survey_for_mass_estimate(conf): # for mass estimate , not for cube correlation
    from multiprocessing import Pool
    from joblib import Parallel,delayed

    max_workers = 1
    chunk_size = 1

    def task(_conf, M, CR, i, model="UCM"):
        label = f"{model}_M{M}_cr{CR}_i{i}"
        Vmax = np.sqrt(2 * nc.G * M * nc.Msun / (CR * nc.au)) / 1e5
        _conf = _conf.replaced(
            Ms_Msun=M,
            CR_au=CR,
            incl=i,
            inenv=model,
            run_dir=f"./pmap2/run_" + label,
            n_thread=1,
            # convmode="null",
            dr_to_r=0.01,
            size_au=4 * CR,
            vfw_kms=1.5 * Vmax * 2,
         #   nphot=1e7,
        )
        synobs(
            _conf,
            vkms_pvd=Vmax * 1.5,
            xlim_pvd=CR * 2,
            skip_if_exist=False , # True
            skip_if_not_exist=True ,
            calc_model=False,
            calc_obs=False,
            plot_model_structure=0
        )

    model = ["UCM"]
    Mlist = np.geomspace(0.03, 3, 10)
    crlist = np.geomspace(40, 400, 10)
    #Mlist = np.geomspace(0.04, 3.2, 16)
    #crlist = np.geomspace(40, 320, 16)
    ilist = [90, 60]  # np.arange(90, 0, -20)
    params = np.array(list(product(Mlist, crlist, ilist)))

    logging.info("Start")
    result = Parallel(n_jobs=-1)([delayed(task)(conf, *p) for p in params])
    logging.info("End")


def do_pmap_survey_for_cube_correlation(conf, single=False): # for mass estimate , not for cube correlation
    from multiprocessing import Pool
    from joblib import Parallel,delayed

    single = 0

    def task(_conf, M, CR, model):
        label = f"{model}_M{M}_cr{CR}"
        if model == "Simple":
            _conf = _conf.replaced(cavangle_deg=80)

        _conf = _conf.replaced(
            Ms_Msun=M,
            CR_au=CR,
            inenv=model,
            #cavangle_deg=ca,
            run_dir=f"./cubes5/run_" + label,
            n_thread = 1,
            dr_to_r = 0.02,
        #    nphot=1e4, #1e7,
        )
        synobs(
            _conf,
            skip_if_exist=1, # True
            skip_if_not_exist=0,
            #read_mg=0,
            save_mg=0,
            calc_model=1,
            calc_obs=1,
            plot_model_structure=0
        )

    model = ["UCM", "Simple"]
    #model = ["Simple"]
    Mlist = np.linspace(0, 0.6, 30) # 0.025
    crlist = np.linspace(0, 600, 30) # 15
    # 0 -- 300
    #Mlist = np.linspace(0, 0.25, 10)
    #crlist = np.linspace(100, 250, 10)
    #params = np.array(list(product(Mlist[1:], crlist[1:], model)) )
    #params = np.array(list(product(Mlist[1:], crlist[1:], model)) )
    params = product( Mlist[1:], crlist[1:], model)
    #print(  product(Mlist[1:], crlist[1:], model)  )
    #print(params)
    #exit()

    if single:
        for p in params:
            task(conf, *p)
    else:
        logging.info("Start")
        result = Parallel(n_jobs=16)(
            [delayed(task)(conf, *p) for p in params]
        )
        logging.info("End")


def plot_obspv(conf):
    f_crit=0.3
    opv = get_obspv(conf.posang + 90 + 0)
    opv.norm_I("max")
    opv.set_I(opv.get_I().clip(0))
    envos.plot_tools.plot_pvdiagram(opv, mass_estimate=False, out="obspvd.pdf", f_crit=f_crit)
    envos.plot_tools.plot_pvdiagram(opv, mass_estimate=1, out="obspvd_mass_q4.pdf", f_crit=f_crit, quadrant=4)
    envos.plot_tools.plot_pvdiagram(opv, mass_estimate=1, out="obspvd_mass_q2.pdf", f_crit=f_crit, quadrant=2)
    #envos.plot_tools.plot_pvdiagram(opv.reversed_xv(), mass_estimate=1, out="obspvd_mass_rev.pdf", f_crit=f_crit)
    envos.plot_tools.plot_pvdiagram(
        opv, mass_estimate=False, loglog=True, out="obspvd_loglog.pdf", f_crit=f_crit
    )

def compare_bestmodel_obspvs(conf):
    cube = envos.read_obsdata(envos.gpath.home_dir / "run"/ "lobs_fid_best.pkl")
    cube.norm_I("max")
    for dpa in [0, 30, 60]:
        pa =  conf.posang + 90 + dpa
        refpv = get_obspv(pa)
        envos.plot_tools.plot_pvdiagram(
            cube.get_pv_map(pangle_deg=pa),
            mass_estimate=False,
            figsize=None,
            ylim=(-2, 2),
            clim=(0,1),
            #f_crit=0.3,
            #incl=conf.incl,
            refpv=refpv,
            out=f"pvd_dpa{dpa}_comp.pdf",
        )


def do_disk_envelope_runs():
    config = config.replaced(
        n_thread=16,
        dr_to_r=0.01,
        molname="c18o",
        molabun=3e-7,
        iline=1,
        f_dg=0.01,
        nphot=1e7,
    )
    synobs(config.replaced(fig_dir=f"run/fig_fid"), filename=f"lobs_fid.pkl")
    synobs(
        config.replaced(fig_dir=f"run/fig_disk", disk="exptail"),
        filename=f"lobs_disk.pkl",
    )
    synobs(
        config.replaced(fig_dir=f"run/fig_diskonly", disk="exptail", inenv=None),
        filename=f"lobs_diskonly.pkl",
    )



#
# Ring tests
#
def do_ring_test(conf):
    synobs(config.replaced(run_dir=f"run/run_ringall"), filename=f"lobs_ringall.pkl")
    crau = conf.CR_au
    for rring in np.arange(0.5 * crau, 3 * crau, 60):
        def rhofunc_after_thermal(model):
            return np.where(
                np.abs(model.rr / envos.nc.au - rring) < 30, model.rhogas, 0
            )
        synobs(
            config.replaced(run_dir=f"run/run_rring{rring}"),
            rhofunc_after_thermal=rhofunc_after_thermal,
            filename=f"lobs_rring{rring}.pkl",
        )

#
# Shariff Shock
#
def do_rshock_test(conf):
    crau = conf.CR_au
    def rhofunc_after_thermal(model):
        return np.where(
                (model.rr / envos.nc.au < 1.5 * crau) & (np.abs(model.tt - 0.5*np.pi) < 0.1),
                model.rhogas,
                0
                )
    synobs(conf.replaced(run_dir=f"run/run_rshock"), filename=f"lobs_rshock.pkl", rhofunc_after_thermal=rhofunc_after_thermal)

#
# PS survays
#
def do_ps_survays():
    def wrapfunc_model(M, CR, i, model="UCM"):
        label = f"{model}_M{M}_cr{CR}_i{i}"
        Vmax = np.sqrt(2 * nc.G * M * nc.Msun / (CR * nc.au) ) / 1e5
        conf = config.replaced(
            Ms_Msun=M,
            CR_au=CR,
            incl=i,
            inenv=model,
            run_dir=f"psurvey9/run_" + label,
            n_thread=1,
            #convmode="",
            dr_to_r=0.01,
            size_au=4 * CR,
            vfw_kms=1.5 * Vmax * 2,
            nphot=1e7,
        )
        synobs(conf, vkms_pvd=Vmax * 1.5, xlim_pvd=CR * 2, skip_if_exist=False)

    model = ["UCM"]
    Mlist = np.geomspace(0.03, 3, 20)
    crlist = np.geomspace(40, 400, 20)
    ilist = [95]  # np.arange(90, 0, -20)
    res = [wrapfunc_model(*args) for args in product(Mlist, crlist, ilist)]

def do_faceon_test():
    incl = 10
    conf = config.replaced(
        fig_dir=f"run/fig_i{incl}",
        incl=incl,
        n_thread=15,
        dr_to_r=0.04,
        modified_random_walk=1,
        convmode="null",
        nphot=1e6,
        nphi=1,
    )
    synobs(conf, filename=f"lobs_i{incl}.pkl")


###########################################################
# Best fit parameter using interp is M=0.17222925561366095 cr=157.36149356996: cor=0.3920513393330384
# Best fit parameter using interp is M=0.14397671494434056 cr=157.89468833002474: cor=0.3168025073954296

# Best fit parameter using interp is M=0.19222588413610658 cr=124.62313207624051: cor=0.37106233039132985
config_default = envos.Config(
    run_dir="./run",
    n_thread=1,
    rau_in=10,
    rau_out=1000,
    theta_out=np.pi,
    dr_to_r=0.02,
    aspect_ratio=1,
    nphi=91,
    nphot=1e7,
    CR_au=200, #290, #124.623,  #157.9,    #157.36,
    Ms_Msun=0.2, #0.154, #0.192226, #0.1439, #0.17223,
    T=None,
    Mdot_smpy=4.5e-6,
    cavangle_deg=45,
    f_dg=0.01,
    opac="MRN20",
    Lstar_Lsun=2.75,
    molname="c3h2",
    molabun=1e-20,
    iline=69,
    size_au=2000,
    pixsize_au=17.5,
    vfw_kms=8,
    dv_kms=0.05,
    beam_maj_au=109.90,
    beam_min_au=94.771,
    vreso_kms=0.1469,
    beam_pa_deg=179.3,
    convmode="scipy",
    incl=95,
    posang=95,
    dpc=140,
)


def synobs(
    conf,
    filename="lineobs.pkl",
    *,
    rhofunc_before_thermal=None,
    rhofunc_after_thermal=None,
    Tfunc=None,
    p=None,
    skip_if_exist=False,
    skip_if_not_exist=False,
    pa_pv=0,
    vkms_pvd=2.0,
    xlim_pvd=700,
    obsdust=False,
    refdata=None,
    save_mg=True,
    calc_model=True,
    calc_obs=True,
    plot_model_structure=True,
    plot_pv=True,
    plot_pv_loglog=True,
    plot_pv_for_mass_estimate=True,
    plot_mom0_map=True,
):

    """
    check if file exists and if so skip synobs
    """
    savefile = envos.gpath.run_dir / filename
    if savefile.exists() and skip_if_exist:
        print(f"Skip syonbs for {envos.gpath.run_dir/filename} because it already exists")
        return

    if ( not savefile.exists()) and skip_if_not_exist:
        print(f"Skip syonbs for {envos.gpath.run_dir/filename} because it does not exist")
        return

    print(f"Start synobs to make {filename}")

    """
    set logging
    """
    # envos.gpath.logfile = envos.gpath.run_dir / "log.dat"
    envos.log.update_logfile()
    conf.log()

    """
    calc kinematic structure of model
    """
    if (not calc_model):
        print("Try to read ", envos.gpath.run_dir / "mg.pkl" )

    if (not calc_model) and \
        (envos.gpath.run_dir / "mg.pkl").exists():
            mg = envos.ModelGenerator(readfile=envos.gpath.run_dir / "mg.pkl")

    elif calc_model:
        mg = envos.ModelGenerator(conf)
        mg.calc_kinematic_structure()

        if rhofunc_before_thermal is not None:
            mg.model.rhogas = rhofunc_before_thermal(mg.model)

        if conf.inenv == "Simple":
             mg.model.Tgas = np.ones_like(mg.model.rhogas) * 30
        else:
            if Tfunc is None:
                mg.calc_thermal_structure()
            else:
                mg.model.Tgas = Tfunc(mg.model)
                # mg.model.Tgas = T0 * (mg.model.rr/10/envos.nc.au)**(p)

        if rhofunc_after_thermal is not None:
            mg.model.rhogas = rhofunc_after_thermal(mg.model)
            # mg.model.rhogas *= np.where( model.rr/envos.nc.au < rout, 1, 0)
            # mg.model.rhogas *= np.where(np.abs(model.rr/envos.nc.au - rring) < 20, 1, 0)

        if save_mg:
            mg.save()
    else:
        print("Skip model calculation")


    if plot_model_structure:
        model = mg.get_model()
        # model.calc_midplane_average()
        envos.plot_tools.plot_density_map(model, streams=True)
        envos.plot_tools.plot_temperature_map(model, streams=True)
        envos.plot_tools.plot_midplane_velocity_map(model)
        # envos.plot_tools.plot_midplane_profiles(model, density=True, velocity=True, Temperature=True)
        # envos.plot_tools.plot_midplane_velocity_profile(model)
        # envos.plot_tools.plot_midplane_temperature_profile(model)
        envos.plot_tools.plot_midplane_density_velocity_profile(model)
        # exit()

    if calc_obs:
        osim = envos.ObsSimulator(conf)
        model = mg.get_model()
        osim.set_model(model)
        cube = osim.observe_line(obsdust=obsdust)
        cube.save(filename=filename)
    else:
        #cube = envos.read_obsdata(envos.gpath.run_dir / filename)
       # print(envos.gpath.home_dir / "run" / filename )
        cube = envos.read_obsdata(savefile)

    if plot_pv:
        opt = {
            "figsize": None,
            "xlim": (-xlim_pvd, xlim_pvd),
            "ylim": (-vkms_pvd, vkms_pvd),
            "f_crit": 0.3,
            "incl": conf.incl,
            "norm":"max",
            "smooth_contour":True,
            "Iunit": "I_{V,{\\rm max}}",
        }
        pv = cube.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv)
        envos.plot_tools.plot_pvdiagram(pv, mass_estimate=False, out="pvd.pdf", **opt)

    if plot_pv_for_mass_estimate:
        envos.plot_tools.plot_pvdiagram(
            pv, mass_estimate=True, out="pvd_mass.pdf", **opt
        )

    if plot_pv_loglog:
        envos.plot_tools.plot_pvdiagram(
            pv, mass_estimate=False, loglog=True, out="pvd_loglog.pdf", **opt
        )
        #envos.plot_tools.plot_pvdiagram(
        #    pv.reversed_x(), mass_estimate=False, loglog=True, out="pvd_loglog_rev.pdf", **opt
        #)

    if refdata is not None:
        refpv = refdata.get_pv_map(pangle_deg = conf.posang - 90 + pa_pv)
        envos.plot_tools.plot_pvdiagram(
            pv,
            refpv=refpv,
            out="pvd_comp.pdf",
            mass_estimate=False,
            **opt,
        )

    if plot_mom0_map:
        img = cube.get_mom0_map()
        img.trim(xlim=[-900,900])
        envos.plot_tools.plot_mom0_map(
            img,
            #pangle_deg=[conf.posang - 90 + pa_pv],
            arrow_angle_deg=conf.posang - 90 + pa_pv,
            norm="max"
        )

    envos.tools.clean_radmcdir()


#################################################################################


def get_refcube():
    #poscen = [69.97447525, 26.05269268]
    poscen =  [69.97444141, 26.05269268]
    refcube = envos.obs.read_cube_fits(
        "./ShareMori/260G_spw1_C3H2_v.fits",
        dpc=140,
        unit1="deg",
        unit2="deg",
        unit3="m/s",
        v0_kms=5.85,
    )
    refcube.set_center_pos(radec_deg=poscen)
    print(refcube)





#    print(refcube)
    #exit()
    #print(refcube)
    #refcube.move_position(poscen[0], ax="x", unit="deg")
    #print(refcube)
    #refcube.move_position(poscen[1], ax="y", unit="deg")

    refcube.trim(xlim=[-700, 700], ylim=[-700, 700], vlim=[-3, 3])
    return refcube


def get_obspv(pa):
    refcube = get_refcube()
    refcube.norm_I("max")
    #print(refcube)
    img = refcube.get_mom0_map()
    envos.plot_tools.plot_mom0_map(
        img,
        pangle_deg=[pa],#[95 - 90 + pa],
        #n_lv=11,
        out=f"mom0_{pa}.pdf",
    )
    opv = refcube.get_pv_map(pangle_deg=pa, norm=None)
    #opv.reverse_v()
    #opv.reverse_x()
    return opv


def plotpvd_angles(conf, filename="lineobs.pkl"):
    print("Start \n\n")
    cube = envos.read_obsdata(envos.gpath.run_dir + "/" + filename)
    if cube is None:
        print("Something wrong in opening ", "run/" + filename)
        return None
    refcube = get_refcube()
    pv0 = refcube.get_pv_map(pangle_deg=conf.posang - 90 + 0, Inorm=None)
    refcube.Ippv /= np.max(pv0.Ipv)
    for pa_pv in [0, 15, 30, 45, 60, 75, 90]:
        refpv = refcube.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv, Inorm=None)
        pv = cube.get_pv_map(pangle_deg=conf.posang - 90 + pa_pv, Inorm=None)
        envos.plot_tools.plot_pvdiagram(
            pv,
            mass_estimate=False,
            refpv=refpv,
            figsize=None,
            ylim=(-2.5, 2.5),
            clim=(0, 1),
            incl=conf.incl,
            out=f"pvd_pa{pa_pv:.0f}.pdf",
            n_lv=10,
            Iunit="$I_{V,{\rm max}}$",
            smooth_contour=True,
        )


######
main(config_default)
