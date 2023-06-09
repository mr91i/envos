import numpy as np
import textwrap
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from .log import logger
from envos import gpath

@dataclass
class Config:
    """
    The Config class contains parameters for the simulation run. 
    When one needs to change a parameter,
    one can directly change it in the configure instance.

    Parameters
    ----------
    General Parameters
    ------------------
    storage_dir : str, default=None
        Path for storing the results. 
    run_dir : str, default=None
        Path where the run files are stored. 
    fig_dir : str, default=None
        Path where the figures are stored.
    radmc_dir : str, default=None
        Path where the RADMC-3D files are stored.
    logfile : str, default=None
        Path where the log file is stored.
    level_stdout : str, default=None
        The logging level for the standard output, can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    level_logfile : str, default=None
        The logging level for the log file, can be "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL".
    n_thread : int, default=1
        Number of threads used for the simulation.

    Grid Parameters
    ---------------
    ri_ax : list, default=None
        List of radial coordinates of cell interfaces.
    ti_ax : list, default=None
        List of θ coordinates of cell interfaces.
    pi_ax : list, default=None
        List of φ coordinates of cell interfaces.
    rau_in : float, default=10
        Inner r-boundary in au. 
    rau_out : float, default=1000
        Outer r-boundary in au.
    theta_in : float, default=0
        Inner θ-boundary in radians (from z-axis).
    theta_out : float, default=np.pi / 2
        Outer θ-boundary in radians (from z-axis).
    phi_in : float, default=0
        Inner φ-boundary in radians (from x-axis).
    phi_out : float, default=2 * np.pi
        Outer φ-boundary in radians (from x-axis).
    nr : int, default=None
        Number of cells in the radial direction.
    ntheta : int, default=None
        Number of cells in the θ direction.
    nphi : int, default=1
        Number of cells in the φ direction.
    dr_to_r : float, default=None
        Ratio of cell width to cell r coordinate.
    aspect_ratio : float, default=1.0
        Aspect ratio of the grid.
    logr : bool, default=True
        If True, use logarithmic scaling for radius.
    ringhost : bool, default=False
        If True, use ghost cells for inner radius.

    Model Parameters
    ----------------
    T : float, default=None
        Temperature, in K.
    CR_au : float, default=None
        Centrifugal radius, in au. 
    Ms_Msun : float, default=None
        Stellar mass, in Solar mass.
    t_yr : float, default=None
        Time in years.
    Omega : float, default=None
        Angular velocity of the cloud core, in rad s^-1.
    jmid : float, default=None
        Midplane specific angular momentum of the cloud core, in cm^2 s^-1.
    rexp_au : float, default=None
        Radius of expansion surface in the collapsing cloud radius, in au.
    Mdot_smpy : float, default=None
        Mass accretion rate, in Solar mass per year.
    meanmolw : float, default=2.3
        Mean molecular weight, in atomic mass units.
    cavangle_deg : float, default=0
        Cavity angle, in degrees.
    inenv : str, default="UCM"
        Inner envelope model. Options are "UCM", "Simple".
    outenv : str, default=None
        Outer envelope model. Options are "TMC", None (=extrapolate inner envelope).
    disk : str, default=None
        Disk model. Options are "exptail" (exponential-tail disk). 
        The detail configuration can be set by `disk_config`
    rot_ccw : bool, default=False
        If True, the rotation is counterclockwise.
    disk_config : dict, default=None
        Dictionary containing additional disk configuration.

    RADMC-3D Parameters
    -------------------
    nphot : int, default=1e6
        Number of photon packages.
    f_dg : float, default=0.01
        Dust to gas ratio.
    opac : str, default="silicate"
        Dust opacity model.
    Lstar_Lsun : float, default=1.0
        Stellar luminosity, in Solar luminosity.
    mfrac_H2 : float, default=0.74
        Mass fraction of H2.
    Rstar_Rsun : float, default=4.0
        Stellar radius in Solar radii.
    molname : str, default="c18o"
        Name of the molecule.
    molabun : float, default=<unknown>
        Molecular abundance.
    iline : int, default=3
        Line transition index.
    scattering_mode_max : int, default=0
        Scattering mode maximum. Options are 0 (isotropic), 1 (anisotropic), and 2 (coherent backscattering).
    mc_scat_maxtauabs : float, default=10.0
        Maximum optical depth for absorption in Monte Carlo simulations.
    tgas_eq_tdust : bool, default=True
        If True, the gas temperature equals the dust temperature.
    lineobs_option : str, default=""
        Line observation options used in RADMC-3D.
    modified_random_walk : int, default=0
        Modified random walk method. Options are 0 (disabled) and 1 (enabled).
    nonlte : int, default=0
        Non-LTE level population calculation. Options are 0 (LTE) and 1 (Non-LTE). **Not tested**

    Observation Parameters
    ----------------------
    dpc : float, default=100
        Distance to the object, in parsecs.
    size_au : float, default=1000
        Size of the image, in au.
    sizex_au : float, default=None
        Size of the image in the x direction, in au.
    sizey_au : float, default=None
        Size of the image in the y direction, in au.
    pixsize_au : float, default=10
        Pixel size, in au.
    vfw_kms : float, default=3
        Full width of the velocity range, in km/s.
    dv_kms : float, default=0.02
        Velocity resolution, in km/s.
    convmode : str, default="scipy"
        Convolution mode. Options are "normal", "fft", "scipy", "null".
        "normal": use astropy.convolve.
        "fft": use astropy.convolve_fft.
        "scipy":use scipy.signal.convolve. Probably fastest.   
        "null": do nothing. 
    beam_maj_au : float, default=50
        Major axis of the beam in au.
    beam_min_au : float, default=50
        Minor axis of the beam in au.
    vreso_kms : float, default=0.1
        Velocity resolution in km/s.
    beam_pa_deg : float, default=0
        Position angle of the beam in degrees.
    incl : float, default=0
        Inclination angle, in deg.
    phi : float, default=0
        Azimuthal angle, in deg.
    posang : float, default=0
        Position angle, in deg.
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
    theta_out: float = 0.5 * np.pi
    phi_in: float = 0
    phi_out: float = 2 * np.pi
    nr: int = None
    ntheta: int = None
    nphi: int = 1
    dr_to_r: float = None
    aspect_ratio: float = 1.0
    logr: bool = True
    ringhost: bool = False

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
    disk_config: dict = None

    # RADMC-3D input
    nphot: int = 1e6
    f_dg: float = 0.01
    opac: str = "silicate"
    Lstar_Lsun: float = 1.0
    mfrac_H2: float = 0.74
    Rstar_Rsun: float = 4.0
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
    nonlte: int=0

    # Observarion input
    dpc: float = 100
    size_au: float = 1000
    sizex_au: float = None
    sizey_au: float = None
    pixsize_au: float = 10
    vfw_kms: float = 3
    dv_kms: float = 0.02
    convmode: str = "scipy"
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
                space_var = " " * len(var)
                txt += space + var + str(v).replace("\n", "\n" + space + space_var)
            else:
                txt += space + var + str(v)
        else:
            txt += "\n"
            txt += space + "-" * (nchar_max * 2 + 3) + "\n"
            txt += space + "Neglected parameters:\n"
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
            log.set_logfile("on")

    def replaced(self, **changes):
        return replace(self, **changes)

    def log(self):
        logger.info(self.__str__())
