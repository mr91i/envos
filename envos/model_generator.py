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

logger = set_logger(__name__)


class ModelGenerator:
    # def __init__(self, filepath=None, grid=None, ppar=None):
    def __init__(self, config=None):
        self.grid = None
        self.ppar = None
        self.inenv = None
        self.outenv = None
        self.disk = None

        if config is not None:
            self.grid = config.grid
            self.ppar = config.ppar
            self._inenv_input = config.model.inenv
            self._outenv_input = config.model.outenv
            self._disk_input = config.model.disk
            self.radmc_config = config.radmc

    #        if filepath is not None:
    #            read_model(filepath, base=self)
    def calc_kinematic_structure(
        self, inenv="CM", disk=None, outenv=None, grid=None, ppar=None
    ):

        ### Set grid
        if grid is not None:
            self.grid = grid
        if self.grid is None:
            raise Exception("grid is not set.")

        ### Set physical parameters
        if ppar is not None:
            self.ppar = ppar
        if self.ppar is None:
            raise Exception("ppar is not set.")

        ### Set models
        if inenv is not None:
            self._inenv_input = inenv
        self.set_inenv(self._inenv_input)

        if outenv is not None:
            self._outenv_input = outenv
        self.set_outenv(self._outenv_input)

        if disk is not None:
            self._disk_input = disk
        self.set_disk(self._disk_input)

        logger.info("Calculating kinematic structure")
        logger.info("Models:")
        logger.info(f"    disk: {self._disk_input}")
        logger.info(f"    inner-envelope: {self._inenv_input}")
        logger.info(f"    outer-envelope: {self._outenv_input}\n")

        ### Make kmodel
        conds = [np.ones_like(self.grid.rr, dtype=bool)]
        regs = [self.inenv]

        if hasattr(self.outenv, "rho"):
            cond1 = self.outenv.rho > self.inenv.rho
            cond2 = self.grid.rr > self.outenv.r_exp
            conds.insert(0, cond1 & cond2)
            regs.insert(0, self.outenv)

        if hasattr(self.disk, "rho"):
            conds.insert(0, self.disk.rho > self.inenv.rho)
            regs.insert(0, self.disk)

        rho = np.select(conds, [r.rho for r in regs])
        vr = np.select(conds, [r.vr for r in regs])
        vt = np.select(conds, [r.vt for r in regs])
        vp = np.select(conds, [r.vp for r in regs])

        logger.info("Calculated kinematic structure")
        #self.kmodel = KinematicModel(self.grid, rho, vr, vt, vp)
        self.kmodel = CircumstellarModel(grid=self.grid)
        self.kmodel.set_density(rho)
        self.kmodel.set_velocity(vr, vt, vp)


    def set_inenv(self, inenv):
        if inenv is None:
            raise Exception("No Envelope.")

        if hasattr(inenv, "rho"):
            self.inenv = inenv
        elif inenv == "CM":
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

    def get_kinematic_structure(self):
        return self.kmodel

    def calc_thermal_structure(self):
        logger.info("Calculating thermal structure")
        conf = self.radmc_config
        radmc = RadmcController(**conf.__dict__)

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
        radmc.set_model(self.kmodel)
        radmc.set_mctherm_inpfiles()
        radmc.run_mctherm()
        radmc.set_lineobs_inpfiles()
        self.tkmodel = radmc.get_model()
        self.tkmodel.add_physical_parameters(self.ppar)
        self.tkmodel.save("tkmodel.pkl")

    def read_model(self):
        # will be cahnged
        self.tkmodel = ThermalKinematicModel("run/tkmodel.pkl")

    def get_model(self):
        return self.tkmodel
