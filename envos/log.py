import os
import logging
import pathlib

loggers = {}
logger_level = logging.INFO
stream_level = logging.INFO
file_level = logging.INFO
debug_logger = 1


def set_logger(name, htype="stream", filepath=None):
    """
    #This function will be called initially in every modules.
    #One can add file handlers after importing.
    Envos' logger

    """

    global loggers, logger_level, stream_level, file_level

    if debug_logger:
        print("set logger: name = ", name)

    ## generate a new logger, which belongs to logging
    logger = logging.getLogger(name)

    ## if it alredy exists, skip this process
    #if logger.hasHandlers():
    #    #logger.warning("This logger is already set. ", name)
    #    print("This logger is already set. ", name)
    #    return logger

    ## set a level of the logger to be a global variable logger_level
    # logger.setLevel(logger_level)

    ## enroll stream handler to the logger
    #if not logger.hasHandlers():

    #if htype == "stream":
    #    _add_stream_handler(logger, stream_level)
    #elif htype == "file":
    #    _add_file_handler(logger, filepath, file_level)

    ## Not allow logger to send messages to its parent logger
    ## if child logger name is xxxx.aaaa, parent logger name is xxxx.
    logger.propagate = False

#    if enable_saving:
#        _add_file_handler(logger)

    ## loggers holds all loggers and their name as a dictionary
    # loggers.update({name: logger})

    return logger


def get_level(level_name):
    #return logging._nameToLevel[level_name.upper()]
    return logging.getLevelName(level_name.upper())

def add_stream_hdlr(logger, stream_level=stream_level):
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(MyStreamFormatter())
    hdlr.setLevel(stream_level)
    logger.addHandler(hdlr)
    if debug_logger:
        print("logger after _add_stream_handler", logger)


def add_file_hdlr(logger, path, file_level=file_level):
    #global file_level
    #from . import gpath
    # gpath.make_dirs(run=gpath.run_dir)
    # os.makedirs(os.path.dirname(gpath.logfile), exist_ok=True)

    _logfile = pathlib.Path(path)
    _logfile.parent.mkdir(exist_ok=True, parents=True)
    #gpath.logfile.parent.mkdir(exist_ok=True, parents=True)

    hdlr = logging.FileHandler(_logfile, "a", "utf-8")
    hdlr.setFormatter(MyFileFormatter())
    hdlr.setLevel(file_level)

    logger.addHandler(hdlr)
    if debug_logger:
        print("logger after _add_file_handler", logger)

def update_file_handler_for_all_loggers():
    global loggers, enable_saving, file_level

    for logger in loggers.values():
        for hdlr in logger.handlers:
            if type(hdlr) == logging.FileHandler:
                logger.removeHandler(hdlr)
        if enable_saving:
            _add_file_handler(logger, file_level)

#    for logger in loggers.values():
#        for hdlr in logger.handlers:
#            print(logger.name, hdlr)




##

## What user wants to do:
##  - set logging level
##  - set logger only for result output
##  - controll log stdout/file/etc

## Set a file handler of all loggers
##  - turn on/off etire logs
##  - newly set logfile
#
# user_logger = set_logger(name="", type="file", filepath="**", )
# user_logger.info(" ***** ")
def set_file(filename=None, filepath=None, switch="on", level_name=None):
    """
    add a file handler to all loggers in envos
    """

    global file_level

    if switch == "on":
        _file_level = file_level if level_name is None else get_level(level_name)

        from . import gpath
        if os.path.isfile(gpath.logfile):
            os.remove(gpath.logfile)

#        if filepath is None:
#            filepath =
        rootlogger = logging.getLogger()
        logger = logging.getLogger("envos")
        print(logger)
        exit()

        for logger in loggers.values():
            for hdlr in logger.handlers:
                _add_file_handler(logger, filepath, file_level)

    elif switch == "off":

        for logger in loggers.values():
            for hdlr in logger.handlers:
                if type(hdlr) == logging.FileHandler:
                    logger.removeHandler(hdlr)


## Controll Level of Handlers
def set_level(level_name, target="all", ver=2):
    global loggers, stream_level, file_level

    level = get_level(level_name)
    htype = {
        "stream":logging.StreamHandler,
        "file":logging.FileHandler,
        "all": None,
    }[target]

    for name, logger in loggers.items():
        for hdlr in logger.handlers:
            if (type(hdlr) == htype) or (target == "all"):
                if debug_logger:
                    print(f"Now setting level to be {level_name}:{level} for {type(hdlr)}")
                if ver == 2:
                    hdlr.setLevel(level)
                elif ver == 1:
                    logger.removeHandler(hdlr)
                    hdlr.setLevel(level)
                    logger.addHandler(hdlr)

    if debug_logger:
        show_loggers()

def show_loggers():
    global loggers
    print("All loggers:")
    for n, logger in loggers.items():
        print("    ", n, logger, ":")
        for hdlr in logger.handlers:
            print("        ", hdlr)

# for user

#   """
#   def enable_saving_output(level_name=None):
#       global enable_saving, file_level
#
#       enable_saving = True
#       if level_name is not None:
#           file_level = get_level(level_name)
#
#       from . import gpath
#
#       if os.path.isfile(gpath.logfile):
#           os.remove(gpath.logfile)
#
#       update_file_handler_for_all_loggers()
#
#
#   def disable_saving_output(logpath=None):
#       global enable_saving
#
#       enable_saving = False
#       update_file_handler_for_all_loggers()
#   """



#    """
#
#
#
#    def set_level_for_all_loggers(level_name):
#        global loggers, stream_level, file_level
#
#        level = get_level(level_name)
#
#        for logger in loggers.values():
#            logger.setLevel(level)
#
#        set_handler_level(level_name)
#        set_handler_level(level_name)
#
#
#
#
#    def set_level_for_stream_handler(level_name):
#        global loggers, stream_level
#
#        stream_level = get_level(level_name)
#
#        for name, logger in loggers.items():
#            for hdlr in logger.handlers:
#                if type(hdlr) == logging.StreamHandler:
#                    logger.removeHandler(hdlr)
#                    hdlr.setLevel(stream_level)
#                    logger.addHandler(hdlr)
#
#
#    def set_level_for_file_handler(level_name):
#        global loggers, file_level
#
#        file_level = get_level(level_name)
#
#        for name, logger in loggers.items():
#            for hdlr in logger.handlers:
#                if type(hdlr) == logging.FileHandler:
#                    logger.removeHandler(hdlr)
#                    hdlr.setLevel(file_level)
#                    logger.addHandler(hdlr)
#
#    """

class color:
    BLACK = "\033[30m"
    GRAY = "\033[96m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RETURN = "\033[07m"
    ACCENT = "\033[01m"
    FLASH = "\033[05m"
    RED_FLASH = "\033[05;41m"
    END = "\033[0m"


class MyStreamFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

        self._fmtdict = {
            logging.DEBUG: "(Debug) %(message)s",
            logging.INFO: "%(message)s",
            logging.WARN: "Warning! -- %(message)s",
            logging.ERROR: "!!ERROR!! %(message)s",
        }

    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno in self._fmtdict.keys():
            self._style._fmt = self._fmtdict[record.levelno]
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result

class MyFileFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

        self._fmtdict = {
            logging.DEBUG: "%(asctime)s [%(filename)s] (Debug) %(message)s",
            logging.INFO: "%(asctime)s [%(filename)s] %(message)s",
            logging.WARN: "%(asctime)s [%(filename)s] Warning! -- %(message)s",
            logging.ERROR: "%(asctime)s [%(filename)s] !!ERROR!! -- %(message)s",
        }

    def format(self, record):
        format_orig = self._style._fmt
        if record.levelno in self._fmtdict.keys():
            self._style._fmt = self._fmtdict[record.levelno]
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result

#############################################
# Set envos logger
#
logger = logging.getLogger("envos")
logger.setLevel(logger_level)
add_stream_hdlr(logger)

