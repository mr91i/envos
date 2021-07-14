import numpy as np
from envos.models import (
    CircumstellarModel,
    CassenMoosmanInnerEnvelope,
    SimpleBallisticInnerEnvelope,
    TerebeyOuterEnvelope,
    ExptailDisk,
)
from envos.radmc3d import RadmcController
from envos.log import set_logger
from envos.grid import Grid
from envos.physical_params import PhysicalParameters
from envos import tools
logger = set_logger(__name__)


class ModelGenerator:
    # def __init__(self, filepath=None, grid=None, ppar=None):
    def __init__(self, config=None):
        self.grid = None
        self.ppar = None
        self.inenv = None
        self.outenv = None
        self.disk = None
        self.model = CircumstellarModel()

        if config is not None:
            self.init_from_config(config)

    def init_from_config(self, config):
        self.config = config

        try:
            grid = Grid(
                config.ri_ax, config.ti_ax, config.pi_ax,
                rau_lim=[config.rau_in, config.rau_out],
                theta_lim=[config.theta_in, config.theta_out],
                phi_lim=[config.phi_in, config.phi_out],
                nr=config.nr,
                ntheta=config.ntheta,
                nphi=config.nphi,
                dr_to_r=config.dr_to_r,
                aspect_ratio=config.aspect_ratio,
                logr=config.logr,
            )
            self.set_grid(grid=grid)
        except Exception as e:
            logger.info("Failed to generate grid.")
            logger.info(e.args)

        self.set_physical_parameters(
            config.T,
            config.CR_au,
            config.Ms_Msun,
            config.t_yr,
            config.Omega,
            config.maxj,
            config.Mdot_smpy,
            config.meanmolw,
            config.cavangle_deg,
        )
        self.set_model(
            inenv=config.inenv, outenv=config.outenv, disk=config.disk
        )

    def set_grid(self, ri=None, ti=None, pi=None, grid=None):
        if grid is not None:
            self.grid = grid
        elif ri is not None:
            self.grid = Grid(
                ri_ax=ri,
                ti_ax=ti,
                pi_ax=pi,
            )
        else:
            logger.error("Failed in making grid.")

        self.model.set_grid(self.grid)

    def get_meshgrid(self):
        return self.grid.rr, self.grid.tt, self.grid.pp


    def set_physical_parameters(
        self,
        T: float = None,
        CR_au: float = None,
        Ms_Msun: float = None,
        t_yr: float = None,
        Omega: float = None,
        maxj: float = None,
        Mdot_smpy: float = None,
        meanmolw: float = 2.3,
        cavangle_deg: float = 0,
    ):

        self.ppar = PhysicalParameters(
            T=T,
            CR_au=CR_au,
            Ms_Msun=Ms_Msun,
            t_yr=t_yr,
            Omega=Omega,
            maxj=maxj,
            Mdot_smpy=Mdot_smpy,
            meanmolw=meanmolw,
            cavangle_deg=cavangle_deg,
        )

    def set_model(self, inenv=None, outenv=None, disk=None):
        self.inenv = inenv
        self.outenv = outenv
        self.disk = disk

    def set_gas_density(self, rho):
        self.model.set_gas_density(rho)

    def set_gas_velocity(self, vr, vt, vp):
        self.model.set_gas_velocity(vr, vt, vp)



    def calc_kinematic_structure(self):

        ### Set grid
        #if grid is not None:
        #    self.grid = grid
        if self.grid is None:
            raise Exception("grid is not set.")

        ### Set physical parameters
        #if ppar is not None:
        #    self.ppar = ppar
        if self.ppar is None:
            raise Exception("ppar is not set.")

        ### Set models
        logger.info("Calculating kinematic structure")
        logger.info("Models:")
        #logger.info(f"    inner-envelope: {inenv or self.inenv}")
        #logger.info(f"    outer-envelope: {outenv or self.outenv}")
        #logger.info(f"    disk: {disk or self.disk}\n")
        #self.set_inenv(inenv or self.inenv)
        #self.set_outenv(outenv or self.outenv)
        #self.set_disk(disk or self.disk)
        logger.info(f"    inner-envelope: {self.inenv}")
        logger.info(f"    outer-envelope: {self.outenv}")
        logger.info(f"    disk: {self.disk}\n")
        self.set_inenv(self.inenv)
        self.set_outenv(self.outenv)
        self.set_disk(self.disk)

        ### Make kmodel
        zeros = np.zeros_like(self.grid.rr)
        rho = np.copy(zeros)
        vr = np.copy(zeros)
        vt = np.copy(zeros)
        vp = np.copy(zeros)
        keys = ("rho", "vr", "vt", "vp")
        ismodel = lambda x: all(hasattr(x, k) for k in keys)

        if ismodel(self.inenv):
            logger.info("Setting inner envelop")
            cond = rho < self.inenv.rho
            rho[cond] = self.inenv.rho[cond]
            vr[cond] = self.inenv.vr[cond]
            vt[cond] = self.inenv.vt[cond]
            vp[cond] = self.inenv.vp[cond]

        if ismodel(self.outenv):
            logger.info("Setting outer envelop")
            cond1 = rho < self.outenv.rho
            cond2 = self.grid.rr > self.outenv.rin_lim
            cond = cond1 & cond2
            rho[cond] = self.outenv.rho[cond]
            vr[cond] = self.outenv.vr[cond]
            vt[cond] = self.outenv.vt[cond]
            vp[cond] = self.outenv.vp[cond]

        if ismodel(self.disk):
            logger.info("Setting disk")
            cond = rho < self.disk.rho
            rho[cond] = self.disk.rho[cond]
            vr[cond] = self.disk.vr[cond]
            vt[cond] = self.disk.vt[cond]
            vp[cond] = self.disk.vp[cond]

        logger.info("Calculated kinematic structure")
        #if grid is not None:
        #self.model.set_grid(self.grid)
        self.model.set_gas_density(rho)
        self.model.set_gas_velocity(vr, vt, vp)
        self.model.set_physical_parameters(self.ppar)

    def set_inenv(self, inenv):
        if inenv is None:
            return

        if hasattr(inenv, "rho"):
            self.inenv = inenv
        elif inenv == "UCM":
            self.inenv = CassenMoosmanInnerEnvelope(
                self.grid,
                self.ppar.Mdot,
                self.ppar.CR,
                self.ppar.Ms,
                self.ppar.cavangle,
            )

        elif inenv == "Simple":
            self.inenv = SimpleBallisticInnerEnvelope(
                self.grid,
                self.ppar.Mdot,
                self.ppar.CR,
                self.ppar.Ms,
                self.ppar.cavangle,
            )
        else:
            raise Exception("Unknown inenv type")

    def set_outenv(self, outenv):
        if outenv is None:
            return

        if hasattr(outenv, "rho"):
            self.outenv = outenv
        elif outenv == "TSC":
            self.outenv = TerebeyOuterEnvelope(
                self.grid,
                self.ppar.t,
                self.ppar.cs,
                self.ppar.Omega,
                self.ppar.cavangle,
            )
        else:
            raise Exception("Unknown outenv type")

    def set_disk(self, disk):
        if disk is None:
            return

        if hasattr(disk, "rho"):
            self.disk = disk
        elif disk == "exptail":
            self.disk = ExptailDisk(
                self.grid,
                self.ppar.Ms,
                self.ppar.CR,
                Td=30,
                fracMd=0.1,
                meanmolw=self.ppar.meanmolw,
                index=-1.0,
            )
        else:
            raise Exception("Unknown disk type")

    def calc_thermal_structure(self):
        logger.info("Calculating thermal structure")
        # conf = self.radmc_config
        radmc = RadmcController(config=self.config)
        radmc.clean_radmc_dir()

        #         radmc = RadmcController(
        #             nphot=conf.nphot,
        #             n_thread=conf.n_thread,
        #             scattering_mode_max=conf.scattering_mode_max,
        #             mc_scat_maxtauabs=conf.mc_scat_maxtauabs,
        #             f_dg=conf.f_dg,
        #             opac=conf.opac,
        #             Lstar_Lsun=conf.Lstar_Lsun,
        #             mfrac_H2=conf.mfrac_H2,
        #             T_const=conf.T_const,
        #             Rstar_Rsun=conf.Rstar_Rsun,
        #             temp_mode=conf.temp_mode,
        #             molname=conf.molname,
        #             molabun=conf.molabun,
        #             iline=conf.iline,
        #             mol_rlim=conf.mol_rlim,
        #             run_dir=conf.run_dir,
        #             radmc_dir=conf.radmc_dir,
        #             storage_dir=conf.storage_dir,
        #         )
        radmc.set_model(self.model)
        radmc.set_mctherm_inpfiles()
        radmc.run_mctherm()

        rho = radmc.get_gas_density()
        if not np.allclose(rho, self.model.rhogas, rtol=1e-07, atol=1e-20):
            logger.error(
                "Input value mismatches with that read by radmc3d: gas density"
            )
            raise Exception

        Tgas = radmc.get_gas_temperature()
        self.model.set_gas_temperature(Tgas)

    #        self.model.save("tkmodel.pkl")

    # radmc.set_lineobs_inpfiles()

    # self.model = radmc.get_model()
    # self.tkmodel.add_physical_parameters(self.ppar)
    # self.tkmodel.save("tkmodel.pkl")

    def get_model(self):
        return self.model


def read_model(path):
    if ".pkl" in path:
        return tools.read_pickle(path)
    else:
        raise Exception("Still constructing...Sorry")
        return CircumstellarModel(filepath=path)
