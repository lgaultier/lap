from math import pi

# - GEOPHYSICAL CONSTANTS
deg2km = 111000.
sec2day = 1/86400.
day2sec = 86400.
Rearth = 6371000.0
factor = 180./(pi*Rearth)


# - UNITS AND NAME
unit = {"U": "m/s", "V": "m/s", "T": "degC", "lambda": "/days", "lon": "degE",
        "lat": "degN", "time": "day"}
long_name = {"U": "zonal velocity", "V": "meridional velocity",
             "T": "Sea Surface temperature", "lambda": "Lyapunov exponent",
             "lon": "longitude", "lat": "latitude", "time": "time"}
glob_attributes = {"description": "Tracer advected by Lagrangian advection"
                   "tool lap"}
