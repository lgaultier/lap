import sys
import os
from math import sqrt, pi
import lap.const as const
import numpy
import logging
logger = logging.getLogger(__name__)


def load_python_file(file_path):
    """Load a file and parse it as a Python module."""
    if not os.path.exists(file_path):
        raise IOError('File not found: {}'.format(file_path))

    full_path = os.path.abspath(file_path)
    python_filename = os.path.basename(full_path)
    module_name, _ = os.path.splitext(python_filename)
    module_dir = os.path.dirname(full_path)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    module = __import__(module_name, globals(), locals(), [], 0)
    return module


def make_default(p):
    # Advection grid
    p.make_grid = getattr(p, 'make_grid', True)
    p.output_step = getattr(p, 'output_step', 1.)
    p.box = getattr(p, 'box', None)

    # Collocated or advected tracer
    p.list_tracer = getattr(p, 'list_tracer', None)
    p.list_grid = getattr(p, 'list_grid', None)
    p.list_num = getattr(p, 'list_num', None)
    p.tracer_filter = getattr(p, 'tracer_filter', (0, 0))

    # Velocity parameters
    p.list_vel = list(getattr(p, 'list_vel'))
    p.vel_format = getattr(p, 'vel_format', 'regular_netcdf')
    p.vel_step = getattr(p, 'vel_step', 1.)
    if p.vel_format == 'regular_netcdf':
        p.name_lon = getattr(p, 'name_lon', 'lon')
        p.name_lat = getattr(p, 'name_lat', 'lat')
        p.name_u = getattr(p, 'name_u', 'u')
        p.name_v = getattr(p, 'name_v', 'v')
        # p.name_h = getattr(p, 'name_h', 'h')
    elif p.vel_format == 'nemo':
        p.name_lon = getattr(p, 'name_lon', 'nav_lon')
        p.name_lat = getattr(p, 'name_lat', 'nav_lat')
        p.name_u = getattr(p, 'name_u', 'vozocrtx')
        p.name_v = getattr(p, 'name_v', 'vomecrty')
        # p.name_h = getattr(p, 'name_h', 'sossheig')
    p.vel_filter = getattr(p, 'vel_filter', None)
    p.vel_step = getattr(p, 'vel_step', 1.0)
    p.output_step = getattr(p, 'output_step', 1.0)
    p.stationary = getattr(p, 'stationary', True)
    p.name_h = getattr(p, 'name_h', None)

    # Advection parameters
    p.K = getattr(p, 'K', 0.)
    p.B = sqrt(2 * float(p.K)) / const.deg2km
    p.scale = getattr(p, 'scale', 1.)
    p.gamma = getattr(p, 'gamma', None)

    # outputs
    p.fill_value = getattr(p, 'fill_value', -1e36)
    p.save_U = getattr(p, 'save_U', False)
    p.save_V = getattr(p, 'save_V', False)
    p.save_OW = getattr(p, 'save_OW', False)
    p.save_RV = getattr(p, 'save_RV', False)
    p.save_S = getattr(p, 'save_S', False)
    p.save_traj = getattr(p, 'save_traj', False)
    p.output_dir = getattr(p, 'output_dir', './')

    # misc
    p.parallelisation = getattr(p, 'parallelisation', False)
    p.factor = 180.0 / (pi * const.Rearth)
    return None


def update_progress(progress, arg1, arg2):
    barLength = 30  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "Error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    if arg1 and arg2:
        text = "\rPercent: [{0}] {1}%, {2}, {3}".format("#"*block + "-"*(barLength - block), "%.2f" % (progress*100), arg1 + ', ' + arg2, status)
    elif arg1:
        text = "\rPercent: [{0}] {1}%, {2}, {3}".format("#"*block + "-"*(barLength - block), "%.2f" % (progress*100), arg1, status)
    else:
        text = "\rPercent: [{0}] {1}%, {2} ".format("#"*block + "-"*(barLength - block), "%.2f" % (progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def lin_1Dinterp(a, delta):
    if len(a) > 1:
        y = a[0]*(1 - delta) + a[1] * delta
    elif len(a) == 1:
        y = a[0]
    else:
        raise Exception('empty array in 1d interpolation.')
        #sys.exit(1)
    return y


def lin_2Dinterp(a, delta1, delta2):
    x = lin_1Dinterp(a[0, :], delta1)
    y = lin_1Dinterp(a[1, :], delta1)
    z = lin_1Dinterp([x, y], delta2)
    return z


def bissextile(year):
    biss = 0
    if numpy.mod(year, 400) == 0:
        biss = 1
    if (numpy.mod(year, 4) == 0 and numpy.mod(year, 100) != 0):
        biss = 1
    return biss


def dinm(year, month):
    if month > 12:
        logger.error("wrong month in dinm")
        sys.exit(1)
    biss = bissextile(year)
    if (month == 4 or month == 6 or month == 9 or month == 11):
        daysinmonth = 30
    elif month == 2:
        if biss:
            daysinmonth = 29
        else:
            daysinmonth = 28
    else:
        daysinmonth = 31
    return daysinmonth


def jj2date(sjday):
    #  sys.exit("to be written")
    jday = int(sjday)
    year = 1950
    month = 1
    day = 1

    for iday in numpy.arange(1, jday + 1):
        day += 1
        daysinmonth = dinm(year, month)
        if (day > daysinmonth):
            day = 1
            month = month + 1
        if (month > 12):
            month = 1
            year = year + 1
    return year, month, day


def haversine(lon1, lon2, lat1, lat2):
    lon1 = numpydeg2rad(lon1)
    lon2 = numpydeg2rad(lon2)
    lat1 = numpydeg2rad(lat1)
    lat2 = numpydeg2rad(lat2)
    havlat = numpy.sin((lat2 - lat1) / 2)**2
    havlon = numpy.cos(lat1) * numpy.cos(lat2)
    havlon = havlon * numpy.sin((lon2 - lon1) / 2)**2
    d = 2 * const.Rearth * numpy.arcsin(numpy.sqrt(havlat + havlon))
    return d


def available_tracer_collocation():
    dict_tracer = {}
    dict_tracer['var'] = {'ostia': 'sst', 'cbpm': 'npp', 'SMOS': 'sss',
                          'SMOSL4': 'sss', 'SMOSL4_wind': 'wind\ speed',
                          'SMOSL4_MLD': 'MLD', 'SMOSL4_sst': 'sst',
                          'SMOSL4_evapo': 'evaporation',
                          'SMOSL4_precip': 'precipitation', 'MODIS': 'SST',
                          'MODISAm': 'acdm', 'MODISAp': 'par',
                          'MODISAc': 'chl',
                          'aquarius_cap': 'sss', 'aquarius': 'sss'}
    dict_tracer['name'] = {'ostia': 'ostia_', 'cbpm': 'cbpm_',
                           'SMOS': 'smos_locean', 'SMOSL4': 'smosL4_',
                           'SMOSL4_wind': 'smosL4_', 'SMOSL4_MLD': 'smosL4_',
                           'SMOSL4_sst': 'smosL4_', 'SMOSL4_evapo': 'smosL4_',
                           'SMOSL4_precip': 'smosL4_', 'MODIS': 'modis',
                           'MODISAm': 'modisAcdm_', 'MODISAp': 'modisApar_',
                           'MODISAc': 'modisAchl_',
                           'aquarius_cap': 'aquarius_cap_',
                           'aquarius': 'aquarius_'}
    dict_tracer['dir'] = {'ostia': 'ostia', 'cbpm': 'cbpm',
                          'SMOS': 'smos_locean', 'SMOSL4': 'smosl4',
                          'SMOSL4_wind': 'smosl4', 'SMOSL4_MLD': 'smosl4',
                          'SMOSL4_sst': 'smosl4', 'SMOSL4_evapo': 'smosl4',
                          'SMOSL4_precip': 'smosl4', 'MODIS': 'modis',
                          'MODISAm': 'modisAcdm', 'MODISAp': 'modisApar',
                          'MODISAc': 'modisAchl',
                          'aquarius_cap': 'aquarius_cap',
                          'aquarius': 'aquarius'}
    dict_tracer['tstep'] = {'ostia': 1, 'cbpm': 8, 'SMOS': 10, 'SMOSL4': 7,
                            'SMOSL4_wind': 7, 'SMOSL4_MLD': 7, 'SMOSL4_sst': 7,
                            'SMOSL4_evapo': 7, 'SMOSL4_precip': 7, 'MODIS': 1,
                            'MODISAm': 1, 'MODISAp': 1, 'MODISAc': 1,
                            'aquarius_cap': 1, 'aquarius': 1}
    return dict_tracer


def convert(x, y, u, v):
    """
    Convert U V components from metric to angular system units
    """
    assert(u.shape == v.shape)

    # nx = len(x)
    # ny = len(y)

    # if nx == u.shape[0]:
    #     assert(u.shape == (nx, ny))
    #     transpose = False
    # else:
    #     assert(u.shape == (ny, nx))
    #     transpose = True

    # Keep longitude between -180, 180
    #x[numpy.where(x > 180)] -= 360
    x0 = + x
    y0 = + y
    x0[numpy.where(x0 > 180)] -= 360
    # Conversion from spherical to cartesian coordinates and move it
    # for one second using U and V component
    lon = numpy.radians(x0)
    lat = numpy.radians(y0)
    sin_x = numpy.sin(lon)
    cos_x = numpy.cos(lon)
    sin_y = numpy.sin(lat)
    cos_y = numpy.cos(lat)
    xc = -u * sin_x - v * cos_x * sin_y
    yc = u * cos_x - v * sin_y * sin_x
    zc = v * cos_y
    xc =  xc + const.Rearth * cos_y * cos_x
    yc = yc + const.Rearth * cos_y * sin_x
    zc = zc + const.Rearth * sin_y

    # Conversion from cartesian to spherical coordinates
    x1 = numpy.degrees(numpy.arctan2(yc, xc))
    y1 = numpy.degrees(numpy.arcsin(
                       zc / numpy.sqrt(xc * xc + yc * yc + zc * zc)))
    dx = x1 - x0
    dy = y1 - y0
    from matplotlib import pyplot
    pyplot.figure(figsize=(10,10))
    s = 3
    pyplot.quiver(x0[::s, ::s], y0[::s, ::s], dx[::s, ::s], dy[::s, ::s])
    pyplot.savefig('figtool.png')
    # Return the velocity in degrees/s
    return numpy.mod(dx + 180.,  360.) - 180., dy


def convert1d(x, y, u, v):
    """
    Convert U V components from metric to angular system units
    """

    # nx = len(x)
    # ny = len(y)

    # Keep longitude between -180, 180
    x[x > 180] -= 360
    x0 = x
    y0 = y

    # Conversion from spherical to cartesian coordinates and move it
    # for one second using U and V component
    lon = numpy.radians(x0)
    lat = numpy.radians(y0)
    sin_x = numpy.sin(lon)
    cos_x = numpy.cos(lon)
    sin_y = numpy.sin(lat)
    cos_y = numpy.cos(lat)
    xc = -u * sin_x - v * cos_x * sin_y
    yc = u * cos_x - v * sin_y * sin_x
    zc = v * cos_y
    xc += const.Rearth * cos_y * cos_x
    yc += const.Rearth * cos_y * sin_x
    zc += const.Rearth * sin_y

    # Conversion from cartesian to spherical coordinates
    x1 = numpy.degrees(numpy.arctan2(yc, xc))
    y1 = numpy.degrees(numpy.arcsin(
        zc / numpy.sqrt(xc * xc + yc * yc + zc * zc)))
    dx = x1 - x0
    dy = y1 - y0

    # Return the velocity in degrees/s
    return (dx + 180) % 360 - 180, dy
