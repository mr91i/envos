import os


def joinpath(basepath, relpath):
    return os.path.abspath(os.path.join(basepath, relpath))


def set_homedir(path, update=True):
    global home_dir
    home_dir = os.path.abspath(path)
    if update:
        update_all_dirs_dependent_on_homedir(home_dir)


def set_rundir(path, update=True):
    global run_dir
    run_dir = os.path.abspath(path)
    make_dirs(run=path)

    if update:
        update_all_dirs_dependent_on_rundir(run_dir)


def set_storagedir(path):
    global storage_dir
    storage_dir = os.path.abspath(path)


def set_radmcdir(path):
    global run_dir
    run_dir = os.path.abspath(path)


def set_logfile(path):
    global logfile
    logfile = os.path.abspath(path)


def update_all_dirs_dependent_on_homedir(home_dir):
    global storage_dir, run_dir, radmc_dir, fig_dir, logfile
    storage_dir = joinpath(home_dir, "radmc_storage")
    make_dirs(storage=storage_dir)
    run_dir = joinpath(home_dir, "run")
    make_dirs(run=run_dir)
    update_all_dirs_dependent_on_rundir(run_dir)


def update_all_dirs_dependent_on_rundir(run_dir):
    global radmc_dir, fig_dir, logfile
    radmc_dir = joinpath(run_dir, "radmc")
    make_dirs(radmc=radmc_dir)
    fig_dir = joinpath(run_dir, "fig")
    make_dirs(fig=fig_dir)
    logfile = joinpath(run_dir, "log.dat")
    from envos.log import update_file_handler_for_all_loggers

    update_file_handler_for_all_loggers()


def make_dirs(radmc=None, run=None, storage=None, fig=None):
    global storage_dir, run_dir, radmc_dir, fig_dir

    if storage is not None:
        storage_dir = storage
        os.makedirs(storage, exist_ok=True)

    if run is not None:
        run_dir = run
        os.makedirs(run, exist_ok=True)

    if radmc is not None:
        radmc_dir = radmc
        os.makedirs(radmc, exist_ok=True)

    if fig is not None:
        fig_dir = fig
        os.makedirs(fig_dir, exist_ok=True)

def remove_radmcdir():
    import shutil
    shutil.rmtree(radmc_dir)

##############################################################################

home_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
storage_dir = joinpath(home_dir, "storage")

run_dir = joinpath("./", "run")
radmc_dir = joinpath(run_dir, "radmc")
fig_dir = joinpath(run_dir, "fig")
logfile = joinpath(run_dir, "log.dat")
