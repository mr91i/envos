#import os
from pathlib import Path


#def joinpath(basepath, relpath):
#    return os.path.abspath(os.path.join(basepath, relpath))

def set_homedir(path, update=True):
    global home_dir
    home_dir = Path(path)
    if update:
        update_all_dirs_dependent_on_homedir(home_dir)


def set_rundir(path, update=True):
    global run_dir
    run_dir = Path(path)
    make_dirs(run=path)
    if update:
        update_all_dirs_dependent_on_rundir(run_dir)


def set_storagedir(path):
    global storage_dir
    storage_dir = Path(path)


def set_radmcdir(path):
    global run_dir
    run_dir = Path(path)


def set_logfile(path):
    global logfile
    logfile = Path(path)


def update_all_dirs_dependent_on_homedir(home_dir):
    global storage_dir, run_dir, radmc_dir, fig_dir, logfile
    storage_dir = home_dir/"radmc_storage"
    make_dirs(storage=storage_dir)
    run_dir = home_dir/"run"
    make_dirs(run=run_dir)
    update_all_dirs_dependent_on_rundir(run_dir)


def update_all_dirs_dependent_on_rundir(run_dir):
    global radmc_dir, fig_dir, logfile
    radmc_dir = run_dir / "radmc"
    make_dirs(radmc=radmc_dir)
    fig_dir = run_dir / "fig"
    make_dirs(fig=fig_dir)
    logfile = run_dir / "log.dat"

    from envos.log import update_file_handler_for_all_loggers
    update_file_handler_for_all_loggers()


def make_dirs(radmc=None, run=None, storage=None, fig=None):
    global storage_dir, run_dir, radmc_dir, fig_dir

    if storage is not None:
        storage_dir = Path(storage)
        storage_dir.mkdir(exist_ok=True,parents=True)

    if run is not None:
        run_dir = Path(run)
        run_dir.mkdir(exist_ok=True,parents=True)

    if radmc is not None:
        radmc_dir = Path(radmc)
        radmc_dir.mkdir(exist_ok=True,parents=True)

    if fig is not None:
        fig_dir = Path(fig)
        fig_dir.mkdir(exist_ok=True,parents=True)


def remove_radmcdir():
    import shutil
    shutil.rmtree(radmc_dir)


##############################################################################
this_file_path = Path(__file__)
working_dir = Path("./")
home_dir = this_file_path.parents[1]
storage_dir = home_dir / "storage"
run_dir = working_dir / "run"
radmc_dir = run_dir / "radmc"
fig_dir = run_dir / "fig"
logfile = run_dir / "log.dat"
