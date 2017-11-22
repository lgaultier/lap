from math import pi

# - GEOPHYSICAL CONSTANTS
deg2km = 111000.
sec2day = 1/86400.
day2sec = 86400.
Rearth = 6371000.0
factor = 180./(pi*Rearth)


# - UNITS AND NAME
unit = {"U": "m/s", "V": "m/s", "T": "degC", "lambda": "/days", "lon": "deg E",
        "lat": "deg N", "time": "day", "FSLE": "1/day", "FTLE": "1/day",
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
