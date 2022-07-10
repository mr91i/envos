import numpy as np
from dataclasses import dataclass, asdict, replace
from envos.log import set_logger, enable_saving_output
from envos import gpath
import textwrap
from pathlib import Path

logger = set_logger(__name__)


@dataclass
class Config:
    """
    Contains all running configurations.
    When one needs to change a parameter,
    one can directly change it in the configure instance.

    Parameters
    ----------
    rau_ax: list
        Radial coorinate of cell interface.
        One can directly input coordinates.

    nr : int
        number of cells in radial axis
    ntheta : int
        number of cells in theta axis
    nphi : int
        number of cells in phi axis


    """

    # General input
    storage_dir: str = None
    run_dir: str = None
    fig_dir: str = None
    radmc_dir: str = None
    logfile: str = None
    level_stdout: str = None
    level_logfile: str = None
    n_thread: int = 1

    # Grid input
    ri_ax: list = None
    ti_ax: list = None
    pi_ax: list = None

    rau_in: float = 10
    rau_out: float = 1000
    theta_in: float = 0
    theta_out: float = np.pi / 2
    phi_in: float = 0
    phi_out: float = 2 * np.pi
    nr: int = None
    ntheta: int = None
    nphi: int = 1
    dr_to_r: float = None
    aspect_ratio: float = 1.0
    logr: bool = True

    # Model input
    T: float = None  # 10
    CR_au: float = None  # 100
    Ms_Msun: float = None  # 1.0
    t_yr: float = None
    Omega: float = None
    jmid: float = None
    rexp_au: float = None
    Mdot_smpy: float = None
    meanmolw: float = 2.3
    cavangle_deg: float = 0
    inenv: str = "UCM"  # {"UCM", "Simple"}
    outenv: str = None
    disk: str = None # {"exptail"}
    rot_ccw: bool = False
    # usr_density_func: Callable = None

    # RADMC-3D input
    nphot: int = 1e6
    f_dg: float = 0.01
    opac: str = "silicate"
    Lstar_Lsun: float = 1.0
    mfrac_H2: float = 0.74
    Rstar_Rsun: float = 1.0
    # temp_mode: str = "mctherm"
    molname: str = "c18o"
    molabun: float = ""
    iline: int = 3
    scattering_mode_max: int = 0
    mc_scat_maxtauabs: float = 10.0
    tgas_eq_tdust: bool = True
    # mol_rlim: float = 1000.0
    lineobs_option: str = ""
    modified_random_walk: int = 0

    # Observarion input
    dpc: float = 100
    size_au: float = 1000
    sizex_au: float = None
    sizey_au: float = None
    pixsize_au: float = 10
    vfw_kms: float = 3
    dv_kms: float = 0.02
    convmode: str = "fft"
    beam_maj_au: float = 50
    beam_min_au: float = 50
    vreso_kms: float = 0.1
    beam_pa_deg: float = 0
    incl: float = 0
    phi: float = 0
    posang: float = 0

    def __str__(self):
        nchar_max = 19
        neglist = []
        space = "  "
        txt = self.__class__.__name__
        txt += "("
        for k, v in asdict(self).items():
            if v is None:
                neglist += [k]
                continue

            txt += "\n"
            var = str(k).ljust(nchar_max) + " = "
            if isinstance(v, (list, np.ndarray)):
                # txt += "\n"
                # txt += space*2 + str(v).replace('\n', '\n'+space*2)
                space_var = " " * len(var)
                txt += space + var + str(v).replace("\n", "\n" + space + space_var)
            else:
                txt += space + var + str(v)
            # txt += ","
        else:
            # txt = txt[:-1]
            txt += "\n"
            txt += space + "-" * (nchar_max * 2 + 3) + "\n"
            txt += space + "Neglected parameters:\n"
            # txtng = ""
            # for k in neglist:
            #    if
            #        txtng
            #    txtng +=

            # txt += space +  "\n".join(textwrap.wrap(", ".join(neglist), 80))
            txt += space * 2 + textwrap.fill(", ".join(neglist), 70).replace(
                "\n", "\n" + space * 2
            )
            txt += "\n" + ")"
        return txt

    def __post_init__(self):

        if self.storage_dir is not None:
            gpath.storage_dir = Path(self.storage_dir)

        if self.run_dir is not None:
            gpath.set_rundir(Path(self.run_dir), update=True)

        if self.fig_dir is not None:
            gpath.fig_dir = Path(self.fig_dir)

        if self.radmc_dir is not None:
            gpath.radmc_dir = Path(self.radmc_dir)

        if self.logfile is not None:
            print("set logfile")
            gpath.logfile = Path(self.logfile)
            enable_saving_output()

    def replaced(self, **changes):
        return replace(self, **changes)

    def log(self):
        logger.info(self.__str__())
