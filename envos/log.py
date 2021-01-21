import logging

DEBUG = False
enable_saving = False
loggers = {}
logger_level = logging.INFO
stream_level = logging.INFO
file_level = logging.INFO


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


class MyFormatter(logging.Formatter):
    def __init__(self, handler):
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s")

        if handler == "stream":
            self._fmtdict = {
                logging.DEBUG: "(Debug) %(message)s",
                logging.INFO: "%(message)s",
                logging.WARN: "Warning! -- %(message)s",
                logging.ERROR: "!!ERROR!! %(message)s",
            }
        elif handler == "file":
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


def set_logger(name):
    global loggers

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.warning("This logger is already set. ", name)
        return logger

    logger.setLevel(logger_level)

    _add_stream_handler(logger)

    if enable_saving:
        _add_file_handler(logger)

    loggers.update({name: logger})

    return logger


# for user
def enable_saving_output(level_name=None):
    global enable_saving, file_level

    enable_saving = True
    if level_name is not None:
        file_level = get_level(level_name)
    update_file_handler_for_all_loggers()


def disable_saving_output(logpath=None):
    global enable_saving

    enable_saving = False
    update_file_handler_for_all_loggers()


def get_level(level_name):
    return logging._nameToLevel[level_name]


def update_file_handler_for_all_loggers():
    global loggers, enable_saving

    for logger in loggers.values():
        for hdlr in logger.handlers:
            if type(hdlr) == logging.FileHandler:
                logger.removeHandler(hdlr)
            if enable_saving:
                _add_file_handler(logger)


def set_level_for_all_loggers(level_name):
    global loggers, stream_level, file_level

    level = get_level(level_name)

    for logger in loggers.values():
        logger.setLevel(level)

    set_level_for_stream_handler(level_name)
    set_level_for_file_handler(level_name)


def set_level_for_stream_handler(level_name):
    global loggers, stream_level

    stream_level = get_level(level_name)

    for name, logger in loggers.items():
        for hdlr in logger.handlers:
            if type(hdlr) == logging.StreamHandler:
                logger.removeHandler(hdlr)
                hdlr.setLevel(stream_level)
                logger.addHandler(hdlr)


def set_level_for_file_handler(level_name):
    global loggers, file_level

    file_level = get_level(level_name)

    for name, logger in loggers.items():
        for hdlr in logger.handlers:
            if type(hdlr) == logging.FileHandler:
                logger.removeHandler(hdlr)
                hdlr.setLevel(file_level)
                logger.addHandler(hdlr)


def _add_stream_handler(logger):
    global stream_level

    stream_handler = logging.StreamHandler()
    fmt = MyFormatter("stream")
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)


def _add_file_handler(logger):
    global file_level

    import envos.global_paths as gp

    gp.make_dirs(run=gp.run_dir)
    file_handler = logging.FileHandler(gp.logfile, "a", "utf-8")
    file_handler = logging.FileHandler(gp.logfile, "a", "utf-8")
    fmt = MyFormatter("file")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)
