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
# - Feb 2015: Version 1.0
# - Dec 2015: Version 2.0
# - Dec 2017: Version 3.0
# - Jan 2021: Version 4.0
#
# Notes:
# - Written for Python 3.4, tested with Python 3.11
#
Copyright (C) 2015-2024 OceanDataLab
This file is part of lap_toolbox.

lap_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

lap_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with lap_toolbox.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------
"""
# -----------------------------------------------------------------------

# ---------------- Module General Import and Declarations ---------------

# - Set module version to package version:

__version__ = '2.0.0'
__author__  = 'Lucile Gaultier <lucile.gaultier@oceandatalab.com>'
__date__ = '2021-01-01'
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
