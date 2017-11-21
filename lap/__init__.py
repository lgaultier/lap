"""Lagrangian advection of particles

   Some useful online help commands for the package:
   * help(lap):  Help for the package.  A list of all modules in
     this package is found in the "Package Contents" section of the
     help output.
   * help(lap.M):  Details of each module "M", where "M" is the
     module's name.

#-----------------------------------------------------------------------
#                       Additional Documentation
# Author: Lucile Gaultier
#
# Modification History:
# - Jul 2016:  Original by Lucile Gaultier
# - Nov 2016: Beta version
# - Feb 2017: Version 1.0
#
# Notes:
# - Written for Python 3.4, tested with Python 3.6
#
# Copyright (c)
#
#-----------------------------------------------------------------------
"""
# -----------------------------------------------------------------------

# ---------------- Module General Import and Declarations ---------------

# - Set module version to package version:

__version__ = '2.0.0'
__author__  = 'Lucile Gaultier <lucile.gaultier@oceandatalab.com>'
__date__ = '2017-12-01'
__email__ = 'lucile.gaultier@oceandatalab.com'
__url__ = ''
__description__ = ('The lap package includes lagrangian tools')
__author_email__ = ('lucile.gaultier@oceandatalab.com')
__keywords__ = ()


# - If you're importing this module in testing mode, or you're running
#  pydoc on this module via the command line, import user-specific
#  settings to make sure any non-standard libraries are found:

import os
import sys
if (__name__ == "__main__") or \
   ("pydoc" in os.path.basename(sys.argv[0])):
    import user


# - Find python version number
__python_version__ = sys.version[:3]

# - Import numerical array formats
try:
    import numpy
except:
    print(''' Numpy is not available on this platform,
          ''')

# - Import scientific librairies
try:
    import scipy
except:
    print("""Scipy is not available on this platform,
          """)


# - Import netcdf reading librairies
try:
    import netCDF4
except:
    print(''' netCDF4 is not available on this machine,
          ''')
