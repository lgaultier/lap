'''
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
'''

from math import pi

# - GEOPHYSICAL CONSTANTS
deg2km = 111000.
sec2day = 1/86400.
day2sec = 86400.
Rearth = 6371000.0
factor = 180./(pi*Rearth)
visc = 1.83e-6
omega = 7.2921*10**(-4)

# - UNITS AND NAME
unit = {"U": "m/s", "V": "m/s", "T": "degC", "lambda": "/days", "lon": "deg E",
        "lat": "deg N", "time": "seconds since 1970-01-01T00:00:00.000000Z",
        "FSLE": "1/day", "FTLE": "1/day",
        "lat_hr": "deg N", "lon_hr": "deg E", "lat_lr": "deg N",
        "lon_lr": "deg E", "time_hr": "day", }
long_name = {"U": "zonal velocity",
             "V": "meridional velocity",
             "T": "Sea Surface temperature",
             "FSLE": "Finite-Size Lyapunov Exponent",
             "FTLE": "Finite-Time Lyapunov Exponent",
             "lon": "longitude", "lat": "latitude", "time": "time",
             "lon_hr": "High temporal resolution longitude",
             "lat_hr": "'High temporal resolution latitude",
             "time_hr": "High temporal resolution time",
             }

glob_attributes = {"description": "Tracer advected by Lagrangian advection"
                   "tool lap"}
