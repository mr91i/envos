import sys
import logging

## Color
class pycolor:
    BLACK = '\033[30m'
    GRAY = '\033[96m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RETURN = '\033[07m' #反転
    ACCENT = '\033[01m' #強調
    FLASH = '\033[05m' #点滅
    RED_FLASH = '\033[05;41m' #赤背景+点滅
    END = '\033[0m'

## logging
class MyFormatter(logging.Formatter):

    def __init__(self, color=True):
        #super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')
        super().__init__(fmt="[%(filename)s] %(levelname)s: %(message)s", datefmt=None, style='%')
        self.dbg_fmt  = "[%(filename)s] Debug: %(message)s"
        self.info_fmt = "[%(filename)s] %(message)s"
        self.warn_fmt = "\n[%(filename)s] Warning!: %(message)s\n"
        self.err_fmt =  "\n[%(filename)s] Error!!: %(message)s\n"

        if color:
            self.dbg_fmt  = pycolor.GRAY + self.dbg_fmt + pycolor.END
            self.info_fmt = self.info_fmt
            self.warn_fmt = pycolor.YELLOW + self.warn_fmt + pycolor.END
            self.err_fmt = pycolor.RED + self.err_fmt + pycolor.END

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt
        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.dbg_fmt
        elif record.levelno == logging.INFO:
            self._style._fmt = self.info_fmt
        elif record.levelno == logging.WARNING:
            self._style._fmt = self.warn_fmt
        elif record.levelno == logging.ERROR:
            self._style._fmt = self.err_fmt
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        # Restore the original format configured by the user
        self._style._fmt = format_orig
        return result


def set_logger(name, logpath=None, ini=False):
    if ini:
        name += "_initial"
    print("Setting logger with ",name)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        print(vars(logger))
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = MyFormatter(color=True)
    str_hdlr = logging.StreamHandler(sys.stdout)
    str_hdlr.setFormatter(fmt)
    str_hdlr.setLevel(logging.DEBUG)

    if logpath is not None:
        print("Set FileHandler in logger")
        fil_hdlr = logging.FileHandler(logpath, "a", "utf-8")
        fil_hdlr.setFormatter(MyFormatter(color=False))
        fil_hdlr.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        print("Add StreamHandler")
        logger.addHandler(str_hdlr)
        if logpath is not None:
            print("Add FileHandler ", logpath)
            logger.addHandler(fil_hdlr)
    print(vars(logger))
    return logger


