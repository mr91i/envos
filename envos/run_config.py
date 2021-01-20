import os


def joinpath(basepath, relpath):
    return os.path.abspath(os.path.join(basepath, relpath))


dp_home = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
dp_storage = joinpath(dp_home, "radmc_storage")
dp_run = joinpath(dp_home, "run")
dp_radmc = joinpath(dp_run, "radmc")
dp_fig = joinpath(dp_run, "fig")
fp_log = joinpath(dp_run, "log.dat")


def set_homedir(path, update=True):
    global dp_home
    dp_home = os.path.abspath(path)
    if update:
        dp_storage = joinpath(dp_home, "radmc_storage")
        dp_run = joinpath(dp_home, "run")
        dp_radmc = joinpath(dp_run, "radmc")
        fp_log = joinpath(dp_run, "log.dat")


def set_storagedir(path):
    global dp_storage
    dp_storage = os.path.abspath(path)


def set_rundir(path, update=True):
    global dp_run
    dp_run = os.path.abspath(path)
    if update:
        dp_radmc = joinpath(dp_run, "radmc")
        fp_log = joinpath(dp_run, "log.dat")


def set_radmcdir(path):
    global dp_run
    dp_run = os.path.abspath(path)


def set_logfile(path):
    global fp_log
    fp_log = os.path.abspath(path)


def make_dirs(radmc=None, run=None, storage=None, fig=None):
    global dp_storage, dp_run, dp_radmc, dp_fig

    if storage is not None:
        dp_storage = storage
        os.makedirs(storage, exist_ok=True)

    if run is not None:
        dp_run = run
        os.makedirs(run, exist_ok=True)

    if radmc is not None:
        dp_radmc = radmc
        os.makedirs(radmc, exist_ok=True)

    if fig is not None:
        dp_fig = fig
        os.makedirs(dp_fig, exist_ok=True)
