import numpy
import os
import sys
import datetime
import glob
import xarray
import lap.utils.read_utils as read_utils
import lap.utils.write_utils as write_utils
import lap.utils.tools as tools
import lap.const as const
import lap.utils.idf as idf_io
from scipy.ndimage import filters
from scipy import interpolate
from math import pi
import logging
logger = logging.getLogger(__name__)

# #############################
# #        READ TRACER       ##
# #############################
def make_mask(p, VEL):
    # mask_grid
    # VEL.lon = (VEL.lon + 360.) % 360.
    masku = VEL['u']['array'][0, :, :]
    masku[abs(masku) > 50.] = numpy.nan
    masku[abs(VEL['v']['array'][0, :, :]) > 50.] = numpy.nan
    vlat = VEL['u']['lat']
    vlon = VEL['u']['lon']
    Teval = interpolate.RectBivariateSpline(vlat, vlon,
                                            numpy.isnan(masku), kx=1, ky=1,
                                            s=0)

    return Teval


def make_grid(p, VEL, coord):
    _coord = list(p.parameter_grid)
    if len(_coord) == 6:
        lon0, lon1, dlon, lat0, lat1, dlat = list(p.parameter_grid)
    else:
        logger.error('Grid parameters must be specified in parameter_grid '
                     '(lon0, lon1, dlon, lat0, lat1, dlat)')
        sys.exit(1)
    # Build grid
    Tr = read_utils.initclass()
    lontmp = numpy.linspace(lon0, lon1, int((lon1 - lon0) / dlon))
    lattmp = numpy.linspace(lat0, lat1, int((lat1 - lat0) / dlat))
    Tr.lon, Tr.lat = numpy.meshgrid(lontmp, lattmp)
    # TODO PROVIDE A MASK FILE
    Teval = make_mask(p, VEL)
    masktmp = Teval(lattmp, lontmp)
    shape_tra = numpy.shape(Tr.lon)
    masktmp = masktmp.reshape(shape_tra)
    # TODO create real mask
    # masktmp = numpy.ones(shape_tra)
    Tr.mask = numpy.ma.getdata((masktmp > 0))
    return Tr


# #############################
# #        READ VEL         ##
# #############################
def sort_files(p):
    list_file = sorted(glob.glob(os.path.join(p.vel_input_dir,
                                              f'*{p.pattern}*')))
    list_date = []
    list_name = []
    frequency = []
    for ifile in list_file:
        match = None
        if p.MATCH is not None:
            match = p.MATCH(ifile)
        if match is None:
            logger.info('Listing all files in the directory as match is None')
            list_name.append(ifile)
        else:
            _date = datetime.datetime(int(match.group(1)), int(match.group(2)),
                                      int(match.group(3)))
            fdate = p.first_date
            ldate = p.last_date
            if p.first_date > p.last_date:
                fdate = p.last_date
                ldate = p.first_date
            if _date >= fdate and _date <= ldate:
                list_date.append(_date)
                list_name.append(ifile)
    if not list_name:
        logger.error(f'{p.pattern} files not found in {p.vel_input_dir}')
        sys.exit(1)
    # The frequency between the grids must be constant.
    if list_date:
        s = len(list_date) -1
        _ind = numpy.argsort(list_date)
        #list_date = list_date[_ind]
        #list_name = list_name[_ind]
        diff = [(list_date[x + 1] - list_date[x]).total_seconds() for x in range(s)]
        _ind = numpy.where(numpy.array(diff)>86400)

        frequency = list(set(diff))
    if frequency:
        frequency = frequency[0]
    if (p.stationary) is True and list_date:
        list_name = [list_name[0]]
        list_date = [list_date[0]]
    #if len(frequency) != 1:
    #    raise RuntimeError(f"Time series does not have a constant step between"
    #                        " two grids: {frequency} seconds")
    return list_name, list_date, frequency


def check_crossing(lon1: float, lon2: float, validate: bool = True):
    """
    Assuming a minimum travel distance between two provided longitude,
    checks if the 180th meridian (antimeridian) is crossed.
    """
    if lon1 > 180:
        lon1 -= 360
    if lon2 > 180:
        lon2 -= 360
    return abs(lon2 - lon1) > 180.0


def read_velocity(p, get_tie=None):
    # Make_list_velocity
    list_vel, list_date, freq = sort_files(p)
    # Read velocity
    filename = os.path.join(p.vel_input_dir, list_vel[0])
    ds = xarray.open_mfdataset(list_vel, concat_dim="time", combine="nested")
    #VEL = ds.sel(depth=0, latitude=slice(p.box[2], p.box[3]))
    IDL = check_crossing(p.box[0], p.box[1])
    box = p.box
    name_lon = p.name_lon
    name_lat = p.name_lat
    _box1 = p.box[1]
    _box0 = p.box[0]
    lon_ctor = getattr(ds, name_lon)
    if (lon_ctor.values > 185).any():
        _box0 = numpy.mod(box[0] + 360, 360)
        _box1 = numpy.mod(box[1] + 360, 360)
        if box[0] > box[1]:
            IDL = True

    if IDL is True:
        ds = ds.sortby(numpy.mod(lon_ctor + 360, 360))
        if p.depth is not None:
            if name_lat == 'lat':
                VEL = ds.sel(depth=p.depth, lat=slice(box[2], box[3]))
            else:
                VEL = ds.sel(depth=p.depth, latitude=slice(p.box[2], p.box[3]))
        else:
            if name_lat == 'lat':
                VEL = ds.sel(lat=slice(p.box[2], p.box[3]))
            else:
                VEL = ds.sel(latitude=slice(p.box[2], p.box[3]))
            if 'depth' in VEL.dims:
                VEL = VEL.isel(depth=0)
    else:
        if p.depth is not None:
            if name_lat == 'lat':
                VEL = ds.sel(depth=p.depth, lat=slice(p.box[2], p.box[3]),
                             lon=slice(_box0, _box1))
            else:
                VEL = ds.sel(depth=p.depth, latitude=slice(p.box[2], p.box[3]),
                             longitude=slice(_box0, _box1))
        else:
            if name_lat == 'lat':
                VEL = ds.sel(lat=slice(p.box[2], p.box[3]),
                             lon=slice(_box0, _box1))
            else:
                VEL = ds.sel(latitude=slice(p.box[2], p.box[3]),
                             longitude=slice(_box0, _box1))
            if 'depth' in VEL.dims:
                VEL = VEL.isel(depth=0)
    # Intialize empty matrices
    ds.close()
    del ds
    tlength = (p.last_date - p.first_date).total_seconds()
    coord = {}
    dic = {}
    lon_ctor = getattr(VEL, p.name_lon)
    lat_ctor = getattr(VEL, p.name_lat)
    coord['lonu'] = numpy.mod(lon_ctor[:].values + 360., 360.)
    coord['latu'] = lat_ctor[:].values
    coord['lonv'] = numpy.mod(lon_ctor[:].values + 360., 360.)
    coord['latv'] = lat_ctor[:].values
    if (not coord['lonu'].any()) or (not coord['latu'].any()):
        logger.error('Check box or parameter_grid keys in your parameter file'
                     ' as no data are found in your area')
        sys.exit(1) 
    # TODO check this format?
    if len(lon_ctor.shape) == 2:
        lon2du, lat2du = (coord['lonu'], coord['latu'])
        lon2dv, lat2dv = (coord['lonv'], coord['latv'])
    else:
        lon2du, lat2du = numpy.meshgrid(coord['lonu'], coord['latu'])
        lon2dv, lat2dv = numpy.meshgrid(coord['lonv'], coord['latv'])
    # coord['lon2du'] = numpy.mod(lon2du + 360., 360.)
    # coord['lat2du'] = lat2du
    # coord['lon2dv'] = numpy.mod(lon2du + 360., 360.)
    # coord['lon2dv'] = lat2dv
    coord['time'] = [numpy.datetime64(x) for x in VEL.time.values]
    coord['time'] = numpy.array(coord['time'])
    # Mask data
    #VEL.fillna(0)
    uo_ctor = getattr(VEL, p.name_u)
    vo_ctor = getattr(VEL, p.name_v)
    dic['ums'] = {'array': uo_ctor.values, 'lon': coord['lonu'], 'lat': coord['latu']}
    dic['vms'] = {'array': vo_ctor.values, 'lon': coord['lonv'], 'lat': coord['latv']}
    dic['u'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                  'lat': coord['latu']}
    dic['v'] = {'array': numpy.zeros(vo_ctor.shape), 'lon': coord['lonv'],
                  'lat': coord['latv']}
    if p.name_h is not None:
        ho_ctor = getattr(VEL, p.name_h)
        dic['h'] = {'array': ho_ctor, 'lon': coord['lonu'], 'lat': coord['latv']}
    if p.save_S or p.save_RV or p.save_OW:
        dic['sn'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}
        dic['ss'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}
        dic['rv'] = {'array': numpy.zeros(uo_ctor.shape), 'lon': coord['lonu'],
                     'lat': coord['latv']}

    for t in range(VEL.time.shape[0]):
        if p.stationary is False:
            perc = float(t / (VEL.time.shape[0]))
            tools.update_progress(perc, '', '')
        # Convert velocity from m/s to degd
        utmp, vtmp = tools.ms2degd(lon2du, lat2du, dic['ums']['array'][t, :, :],
                                   dic['vms']['array'][t, :, :])
        dic['u']['array'][t, :, :] = utmp
        dic['v']['array'][t, :, :] = vtmp
        if p.save_S or p.save_RV or p.save_OW:
            mlat = numpy.deg2rad(numpy.mean((lat2du + lat2dv)/2))
            gfo = (9.81 / numpy.cos(mlat) / (2 * const.omega))
            dxlatu = (lat2du[2:, 1: -1] - lat2du[: -2, 1: -1])
            dylonu = (lon2du[1: -1, 2:] - lon2du[1: -1, : -2])
            dxlatv = (lat2dv[2:, 1: -1] - lat2dv[:-2, 1: -1])
            dylonv = (lon2dv[1: -1, 2:] - lon2dv[1: -1, : -2])
            dyutmp = utmp[1: -1, 2:] - utmp[1: -1, : -2]
            dyvtmp = vtmp[1: -1, 2:] - vtmp[1: -1, : -2]
            dxutmp = utmp[2:, 1: -1] - utmp[:-2, 1: -1]
            dxvtmp = vtmp[2:, 1: -1] - vtmp[:-2, 1: -1]
        if p.save_RV or p.save_OW:
            RV = gfo*(dyvtmp / dylonv - dxutmp / dxlatu)
            dic['rv']['array'][t, 1: -1, 1: -1] = RV
        if p.save_S or p.save_OW:
            Sn = gfo*(dyutmp / dylonu - dxvtmp / dxlatv)
            Ss = gfo*(dxutmp / dxlatu + dyvtmp / dylonv)
            dic['sn']['array'][t, 1: -1, 1: -1] = Sn
            dic['ss']['array'][t, 1: -1, 1: -1] = Ss
    del VEL, lon2du, lat2du
    return dic, coord

'''
# #############################
# #        READ NUDGING      ##
# #############################
def Read_amsr_t(p):
    import os
    # Read AMSR
    if os.path.isfile(p.DIRA + os.sep + p.filea[0]):
        dataamsr = amsr.OISST(p.DIRA + os.sep + p.filea[0])
        lonamsr = dataamsr.variables['longitude']
        latamsr = dataamsr.variables['latitude']
        iamsr0 = numpy.where((lonamsr < (p.box[1] + 5))
                             & (lonamsr > (p.box[0] - 5)))[0][0]
        iamsr1 = numpy.where((lonamsr < (p.box[1] + 5))
                             & (lonamsr > (p.box[0] - 5)))[0][-1]
        jamsr0 = numpy.where((latamsr < (p.box[3] + 5))
                             & (latamsr > (p.box[2] - 5)))[0][0]
        jamsr1 = numpy.where((latamsr < (p.box[3] + 5))
                             & (latamsr > (p.box[2] - 5)))[0][-1]
        sstamsr = dataamsr.variables['sst'][jamsr0: jamsr1, iamsr0: iamsr1]
        dataamsr.lont = lonamsr[iamsr0: iamsr1]
        dataamsr.latt = latamsr[jamsr0: jamsr1]

        varamsr = numpy.zeros((int(abs(p.tadvection)) + 1,
                               numpy.shape(dataamsr.latt)[0],
                               numpy.shape(dataamsr.lont)[0]))
        for t in range(0, abs(p.tadvection)):
            if p.gamma:
                if os.path.isfile(p.DIRA + os.sep + p.filea[t]):
                    AMSR = amsr.OISST(p.DIRA + os.sep + p.filea[t])
                    # varamsr[t+1,:,:]=filters.gaussian_filter(
                    # AMSR.variables['sst'][jamsr0:jamsr1, iamsr0:iamsr1],4)
                    varamsr[t+1, :, :] = AMSR.variables['sst'][jamsr0: jamsr1,
                                                               iamsr0: iamsr1]
                else:
                    varamsr[t + 1, :, :] = numpy.nan
            else:
                AMSR = None
    else:
        AMSR = None
        p.gamma = 0.
    if p.gamma:
        varamsr[varamsr > 50] = -999.
        varamsr[varamsr <= 0] = numpy.nan
        AMSR.tra = varamsr
        AMSR.lont = dataamsr.lont
        AMSR.latt = dataamsr.latt
    return AMSR
'''

# ###########################
# #       WRITE            ##
# ###########################


def write_drifter(p, drifter, listTr, idf=False):
    start = p.first_date.strftime('%Y%m%d')
    stop = p.first_date + datetime.timedelta(days=p.tadvection)
    stop = stop.strftime('%Y%m%d')
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    global_attr = idf_io.global_idf
    global_attr['time_coverage_start'] = p.first_date.strftime(idf_io.idf_fmt)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_listracer_1d(p.output, drifter, p, listTr)


def write_diagnostic_2d(p, data, description='', **kwargs):
    start = p.first_date.strftime('%Y%m%d')
    stop = p.first_date + datetime.timedelta(days=p.tadvection)
    stop = stop.strftime('%Y%m%d')
    file_default = f'{p.out_pattern}_{start}_{stop}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    global_attr = idf_io.global_idf
    global_attr['time_coverage_start'] = p.first_date.strftime(idf_io.idf_fmt)
    _end = p.first_date + datetime.timedelta(days=1)
    global_attr['time_coverage_end'] = _end.strftime(idf_io.idf_fmt)
    global_attr['idf_spatial_resolution'] = p.parameter_grid[5]*111110
    global_attr['idf_spatial_resolution_units'] = "m"
    global_attr['id'] = f'{p.out_pattern}'
    global_attr['idf_granule_id'] = file_default
    data['lon'] = data['lon'][0, :]
    data['lat'] = data['lat'][:, 0]
    _result = idf_io.compute_gcp(data['lon'], data['lat'],
                                 gcp_lat_spacing=5, gcp_lon_spacing=5)

    data['lon_gcp'], data['lat_gcp'], data['index_lat_gcp'], data['index_lon_gcp'] = _result
    p.output = getattr(p, 'output', default_output)
    write_utils.write_velocity(data, p.output, description=description,
                               unit=const.unit, long_name=const.long_name,
                               meta=global_attr,
                               fill_value=-1e36, **kwargs)


def write_advected_tracer(p, data_out):
    start = p.first_date.strftime('%Y%m%d')
    stop = p.first_date + datetime.timedelta(days=p.tadvection)
    stop = stop.strftime('%Y%m%d')
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_tracer_2d(p.output, data_out)


def write_params(p, time):
    outformat = '%Y%m%dT%H%M%S'
    stime = time.strftime(outformat)
    output = f'params_output_{stime}'
    write_utils.write_params(p, output)
