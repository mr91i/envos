# import envos.model
# import

## Check Python version
import sys
py_version = sys.version_info
if py_version[0] + py_version[1]*0.1 < 3.3:
    logging.error('''\n
This module does not work with Python2 or <3.3 .
At 1 Jan 2020, most packages stopped to support Python2 (see https://python3statement.org).
Please use Python3 and update your Python3 to the newest version.
I have confirmed that package works in Python 3.8.1.''')
    exit()


from . import tsc
