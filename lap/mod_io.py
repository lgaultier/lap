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

import numpy
import os
import sys
import datetime
import glob
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


def get_regular_tracer(p, tracer, get_coord=True):
    ''' Get indices and mask values out of tracer and extract area '''
    tracer.read_var()
    if get_coord is True:
        tracer.read_coord()
        # Get area
        iindx = numpy.where((tracer.lon[0, :] <= p.box[1])
                            & (tracer.lon[0, :] >= p.box[0]))[0]
        iindy = numpy.where((tracer.lat[:, 0] >= p.box[2])
                            & (tracer.lat[:, 0] <= p.box[3]))[0]
        slice_x = slice(iindx[0], iindx[-1] + 1)
        slice_y = slice(iindy[0], iindy[-1] + 1)
        tracer.slice_x = slice_x
        tracer.slice_y = slice_y
        # Extract data on area
        tracer.lon = tracer.lon[slice_y, slice_x]
        tracer.lat = tracer.lat[slice_y, slice_x]
        # Fill table of index
        tracer.dlon = numpy.mean(tracer.lon[0, 1:] - tracer.lon[0, : -1])
        tracer.dlat = numpy.mean(tracer.lat[1:, 0] - tracer.lat[: -1, 0])
        shape_tracer = numpy.shape(tracer.lon)
        tracer.i = numpy.zeros(shape_tracer)
        tracer.j = numpy.zeros(shape_tracer)
        tracer.i.astype(int)
        tracer.j.astype(int)
        for i in range(shape_tracer[0]):
            tracer.i[i, :] = int(i)
        for j in range(shape_tracer[1]):
            tracer.j[:, j] = int(j)
    # Extract data on area
    tracer.var = tracer.var[tracer.slice_y, tracer.slice_x]
    # Mask invalid values
    tracer.var[numpy.where(abs(tracer.var) > 500.)] = -999.
    tracer.mask = numpy.isnan(tracer.var)
    if p.tracer_filter[0] != 0 or p.tracer_filter[1] != 0:
        tracer.var = filters.gaussian_filter(tracer.var, p.tracer_filter)
    return None


def read_tracer_netcdf(p):
    Tr = read_utils.tracer_netcdf(file=p.filet, var=p.name_tracer, time='time')
    get_regular_tracer(p, Tr, get_coord=True)
    Tr.iit = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                          numpy.shape(Tr.i)[0]))
    Tr.ijt = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                          numpy.shape(Tr.i)[0]))
    return Tr


def read_list_tracer(p, dict_tracer: dict) -> list:
    listTr = []
    tref = p.tref
    # Loop on all tracer provided in the parameter file
    for tra in p.list_tracer:
        logger.info(f'Reading {tra}')
        i = 0
        tref = p.first_date
        # For each tracer Find first available data
        year, month, day = tools.jj2date(tref)
        tra_name = dict_tracer['name'][tra]
        tra_step = dict_tracer['tstep'][tra]
        tra_dir = dict_tracer['dir'][tra]
        tra_var = dict_tracer['var'][tra]
        filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
        pathfilet = os.path.join(p.tracer_input_dir, tra_dir, filet)
        # If no data is found, look for the previous step
        while ((not os.path.exists(pathfilet))
                and abs(tref - p.first_date) <= tra_step):
            tref -= 1
            year, month, day = tools.jj2date(tref)
            filet = f'{tra_name}{year:02d}{month:02d}{day:02d}.nc'
        # If no data is found in the previous step, there is no available data
        if abs(tref - p.first_date) > tra_step:
            logger.error(f'Tracer files {tra} ({filet}) are not found')
            sys.exit(1)
        # Build list of tracer datasets which have to be collocated
        nstep = p.tadvection / abs(p.tadvection) * tra_step
        tstop = tref + p.tadvection + nstep
        listdate = list(numpy.arange(tref, tstop, numpy.floor(nstep)))
        num_date = len(listdate)
        # Read all necessary time steps
        for jdate in listdate:
            year, month, day = tools.jj2date(jdate)
            filet = f'{tra_name}{year:02d}{month:02d}{day:02d}.nc'
            pathfilet = os.path.join(p.DIR, tra_dir, filet)
            Tr = read_utils.tracer_netcdf(filename=pathfilet, var=tra_var,
                                          time='time', lon='lon', lat='lat')
            get_regular_tracer(p, Tr, get_coord=True)
            Tr.read_time()
            if jdate == listdate[0]:
                shape_var = numpy.shape(Tr.var)
                var2 = numpy.zeros((num_date, shape_var[0], shape_var[1]))
                time = numpy.zeros((num_date))
            time[i] = Tr.time
            nan_mask = numpy.isnan(Tr.var)
            Tr.mask = (Tr.var.mask or nan_mask)
            var2[i, :, :] = Tr.var
            i += 1
        # Initialize empty matrices at the correct resolution
        Tr.iit = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                             numpy.shape(Tr.i)[0]))
        Tr.ijt = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                             numpy.shape(Tr.i)[0]))
        # Store time variable and append tracer to list of tracers
        Tr.var = var2
        Tr.time = time
        listTr.append(Tr)
    return listTr


def read_list_grid(p, dict_tracer: dict) -> list:
    listTr = []
    tref = p.first_date
    # Loop on all tracer grid provided in the parameter file
    for tra in p.list_grid:
        logger.info(f'Reading {tra}')
        jdate = tref
        # For each tracer Find first available data
        year, month, day = tools.jj2date(jdate)
        tra_name = dict_tracer['name'][tra]
        tra_step = dict_tracer['tstep'][tra]
        tra_dir = dict_tracer['dir'][tra]
        tra_var = dict_tracer['var'][tra]
        filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
        pathfilet = os.path.join(p.tracer_input_dir, tra_dir, filet)
        # If no data is found, look for the previous step
        while (not os.path.exists(pathfilet)
               and abs(jdate - p.first_date) <= tra_step):
            jdate -= 1
            year, month, day = tools.jj2date(jdate)
            filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
            pathfilet = os.path.join(p.DIR, tra_dir, filet)
        # If no data is found in the previous step, there is no available data
        if abs(jdate - p.first_date) > tra_step:
            logger.error(f'Grid file {tra} does not exist')
            sys.exit(1)
        Tr = read_utils.tracer_netcdf(filename=pathfilet, var=tra_var,
                                      time='time', lon='lon', lat='lat')
        get_regular_tracer(p, Tr, get_coord=True)
        Tr.iit = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                             numpy.shape(Tr.i)[0]))
        Tr.ijt = numpy.zeros((2, int(abs(p.tadvection) / p.output_step) + 1,
                             numpy.shape(Tr.i)[0]))
        listTr.append(Tr)
    return listTr


# #############################
# #        READ GRID         ##
# #############################


def read_grid_netcdf(p):
    # # TODO, lon= ... lat= ...
    Tr = read_utils.read_tracer(filename=p.fileg, var=p.name_tracer,
                                time='time')
    get_regular_tracer(p, Tr, get_coord=True)
    return Tr

def read_points(p):
    # # TODO, lon= ... lat= ...
    dic = read_utils.read_trajectory(p.filetrajectory, ('lon_hr', 'lat_hr',
                                                        'mask_hr',
                                                        'zonal_velocity'))
    Tr = read_utils.initclass()
    Tr.lon = dic['lon_hr'][-1, :]
    Tr.lat = dic['lat_hr'][-1, :]
    Tr.mask = dic['mask_hr'][-1, :]
    #numpy.ones(numpy.shape(Tr.lon), dtype=bool)
    return Tr

'''
# TODO is it used?
def read_grid_tiff(p, Tr):
    import gdal
    ds = gdal.Open(p.file_grid)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    x = numpy.arange(ds.RasterXSize)
    y = numpy.arange(ds.RasterYSize)
    x = numpy.array([x, ]*ds.RasterYSize)
    y = numpy.array([y, ]*ds.RasterXSize).transpose()
    xp = numpy.zeros(numpy.shape(x))
    yp = numpy.zeros(numpy.shape(y))
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    Tr.lon = xp[arr > 0]
    Tr.lat = yp[arr > 0]
    Tr.var = arr[arr > 0]
    mask = numpy.isnan(Tr.var)
    return Tr
'''


# TODO is it used?
def Read_listgrid_netcdf(p, dict_tracer: dict, jdate: float) -> list:
    listTr = []
    for tra in p.list_grid:
        year, month, day = tools.jj2date(jdate)
        tra_name = dict_tracer['name'][tra]
        tra_dir = dict_tracer['dir'][tra]
        tra_var = dict_tracer['var'][tra]
        filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
        pathfilet = os.path.join(p.tracer_input_dir, tra_dir, filet)
        if not os.path.isfile(pathfilet):
            logger.error(f'File {pathfilet} not found')
            sys.exit(1)
        Tr = read_utils.tracer_netcdf(filename=pathfilet, var=tra_var,
                                      time='time', lon='lon', lat='lat')
        get_regular_tracer(p, Tr, get_coord=True)
        listTr.append(Tr)
    return listTr


def Read_grid_model(p):
    Tr = read_utils.MODISCDF(file=p.fileg, var='T', time='time')
    Tr.read_var()
    Tr.read_coord()
    iindx = numpy.where((Tr.lon[0, :] <= p.box[1])
                        & (Tr.lon[0, :] >= p.box[0]))[0]
    iindy = numpy.where((Tr.lat[:, 0] >= p.box[2])
                        & (Tr.lat[:, 0] <= p.box[3]))[0]
    Tr.lon = Tr.lon[iindy[0]: iindy[-1] + 1, iindx[0]: iindx[-1] + 1]
    Tr.lat = Tr.lat[iindy[0]: iindy[-1] + 1, iindx[0]: iindx[-1] + 1]
    Tr.var = Tr.var[iindy[0]: iindy[-1] + 1, iindx[0]: iindx[-1] + 1]
    # T.read_mask()
    # masque = numpy.greater(abs(T.var), 40)
    # T.var = numpy.ma.array(T.var, mask=masque)
    Tr.var[numpy.where(numpy.isnan(Tr.var))] = -999.
    Tr.var[numpy.where(abs(Tr.var) > 39.)] = -999.
    Tr.var[numpy.where(Tr.var < 0.)] = numpy.nan
    Tr.var[numpy.where(numpy.isnan(Tr.var))] = numpy.nan
    Tr.mask = numpy.isnan(Tr.var)
    return Tr


def make_mask(p, VEL):
    # mask_grid
    # VEL.lon = (VEL.lon + 360.) % 360.
    masku = + VEL['u']['array'][0, :, :]
    masku[abs(masku) > 50.] = numpy.nan
    masku[abs(VEL['v']['array'][0, :, :]) > 50.] = numpy.nan
    masku[VEL['v']['array'][0, :, :] == 0.] = numpy.nan
    masku[VEL['u']['array'][0, :, :] == 0.] = numpy.nan
    vlat = VEL['u']['lat']
    vlon = VEL['u']['lon']
    Teval = interpolate.RectBivariateSpline(vlat, vlon,
                                            numpy.isnan(masku), kx=1, ky=1,
                                            s=0)
    return Teval


def make_grid(p, VEL, coord, mask=True):
    _coord = list(p.parameter_grid)
    if len(_coord) == 6:
        lon0, lon1, dlon, lat0, lat1, dlat = list(p.parameter_grid)
    else:
        logger.error('Grid parameters must be specified in parameter_grid '
                     '(lon0, lon1, dlon, lat0, lat1, dlat)')
        sys.exit(1)
    # Build grid
    lon0 = numpy.mod(lon0 + 360, 360)
    lon1 = numpy.mod(lon1 + 360, 360)
    Tr = read_utils.initclass()
    lontmp = numpy.linspace(lon0, lon1, int((lon1 - lon0) / dlon))
    lattmp = numpy.linspace(lat0, lat1, int((lat1 - lat0) / dlat))
    Tr.lon, Tr.lat = numpy.meshgrid(lontmp, lattmp)
    # TODO PROVIDE A MASK FILE
    Teval = make_mask(p, VEL)
    masktmp = Teval(lattmp, lontmp)
    shape_tra = numpy.shape(Tr.lon)
    if mask == True:
        Teval = make_mask(p, VEL)
        masktmp = Teval(lattmp, lontmp)
        masktmp = masktmp.reshape(shape_tra)
    else:
        masktmp = numpy.ones(shape_tra)
    # TODO create real mask
    #masktmp = numpy.zeros(shape_tra)
    Tr.mask = numpy.ma.getdata((masktmp > 0))
    if numpy.ma.getdata(Tr.mask).all():
        logger.info(f'no data in box {lon0}, {lon1}, {lat0}, {lat1}')
        sys.exit(0)
    #Tr.mask = numpy.ma.getdata()
    return Tr


# #############################
# #        READ VEL         ##
# #############################
def sort_files(p):
    list_file = sorted(glob.glob(os.path.join(p.vel_input_dir,
                                              f'*{p.pattern}*')))
    list_date = []
    list_name = []
    for ifile in list_file:
        match = p.MATCH(ifile)
        if match is None:
            continue
        _date = datetime.datetime(int(match.group(1)), int(match.group(2)),
                                  int(match.group(3)))
        if _date >= p.first_date and _date <= p.last_date:
            list_date.append(_date)
            list_name.append(ifile)
    if not list_name:
        logger.error(f'{p.pattern} files not found in {p.vel_input_dir}')
        sys.exit(1)
    # The frequency between the grids must be constant.
    s = len(list_date) -1
    _ind = numpy.argsort(list_date)
    #list_date = list_date[_ind]
    #list_name = list_name[_ind]
    diff = [(list_date[x + 1] - list_date[x]).total_seconds() for x in range(s)]
    _ind = numpy.where(numpy.array(diff)>86400)
    #print(_ind)
    #print(diff[264], diff[480])

    #print(list_name[264], list_name[480])
    frequency = list(set(diff))
    #if len(frequency) != 1:
    #    raise RuntimeError(f"Time series does not have a constant step between"
    #                        " two grids: {frequency} seconds")
    return list_name, list_date, frequency[0]


def read_velocity(p, get_time=None):
    # Make_list_velocity
    list_vel, list_date, freq = sort_files(p)
    # Read velocity
    filename = os.path.join(p.vel_input_dir, list_vel[0])
    if p.vel_format == 'regular_netcdf':
        VEL = read_utils.velocity_netcdf(filename=filename, varu=p.name_u,
                                         varv=p.name_v, lon=p.name_lon,
                                         lat=p.name_lat, box=p.box)
    elif p.vel_format == 'nemo':
        VEL = read_utils.nemo(filename=filename, varu=p.name_u,
                              varv=p.name_v, lon=p.name_lon,
                              lat=p.name_lat, box=p.box, subsample=p.subsample)
    else:
        logger.error(f'{p.vel_format} format is not handled')
        sys.exit(1)
    VEL.read_coord()
    VEL.read_vel()
    # TODO check this format?
    if len(VEL.lon.shape) == 2:
        lon2du, lat2du = (VEL.lon, VEL.lat)
        lon2dv, lat2dv = (VEL.lon, VEL.lat)
    else:
        lon2du, lat2du = numpy.meshgrid(VEL.lon, VEL.lat)
        lon2dv, lat2dv = numpy.meshgrid(VEL.lon, VEL.lat)
    # Intialize empty matrices
    tlength = (p.last_date - p.first_date).total_seconds()
    if len(numpy.shape(VEL.varu)) == 3 and p.stationary is False:
        ninfilestep = numpy.shape(VEL.varu)[0]
    else:
        ninfilestep = 1

    if p.stationary is True:
        num_steps = 1
    else:
        num_steps = int(tlength / freq *  ninfilestep) + 2
    coord = {}
    dic = {}
    coord['lonu'] = numpy.mod(VEL.lon[:] + 360., 360.)
    coord['latu'] = VEL.lat[:]
    coord['lonv'] = numpy.mod(VEL.lon[:] + 360., 360.)
    coord['latv'] = VEL.lat[:]
    coord['lon2du'] = numpy.mod(lon2du + 360., 360.)
    coord['lat2du'] = lat2du
    coord['lon2dv'] = numpy.mod(lon2du + 360., 360.)
    coord['lon2dv'] = lat2dv
    coord['time'] = []
    dic['u'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latu']}
    dic['v'] = {'array': [], 'lon': coord['lonv'], 'lat': coord['latv']}
    dic['ums'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latu']}
    dic['vms'] = {'array': [], 'lon': coord['lonv'], 'lat': coord['latv']}
    if p.name_h is not None:
        dic['h'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latv']}
    if p.save_S or p.save_RV or p.save_OW:
        dic['sn'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latv']}
        dic['ss'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latv']}
        dic['rv'] = {'array': [], 'lon': coord['lonu'], 'lat': coord['latv']}

    for t in range(len(list_date)):
        if p.stationary is False:
            perc = float(t / (len(list_date)))
            tools.update_progress(perc, '', '')
        filename = list_vel[t]
        # Initialize velocity object
        if p.vel_format == 'regular_netcdf':
            VEL = read_utils.velocity_netcdf(filename=filename, varu=p.name_u,
                                             varv=p.name_v, lon=p.name_lon,
                                             lat=p.name_lat, var=p.name_h,
                                             box=p.box)
        elif p.vel_format == 'nemo':
            VEL = read_utils.nemo(filename=filename, varu=p.name_u,
                                  varv=p.name_v, lon=p.name_lon,
                                  lat=p.name_lat, var=p.name_h,
                                  box=p.box, subsample=p.subsample)
        else:
            logger.error('Undefined Velocity file type')
            sys.exit(1)
        # Read velocity variable, extract box and filter if necessary
        if t == 0:
            VEL.read_vel(size_filter=p.vel_filter)
            slice_x = VEL.slice_x
            slice_y = VEL.slice_y
        else:
            VEL.read_vel(size_filter=p.vel_filter, slice_xy=(slice_x, slice_y))
        VEL.read_time()
        coord['time'].append(VEL.time)
        # Read SSH if needed
        if p.name_h is not None:
            VEL.read_var(size_filter=p.vel_filter)
            VEL.var[numpy.where(abs(VEL.var) > 100)] = numpy.nan
        # Mask data
        mask = (numpy.ma.getmaskarray(VEL.varu)
                | numpy.ma.getmaskarray(VEL.varv)
                | numpy.isnan(VEL.varu) | numpy.isnan(VEL.varv)
                | (VEL.varu == p.missing_value)
                | (VEL.varv == p.missing_value))
        VEL.varu[mask] = 0
        VEL.varv[mask] = 0

        for nt in range(ninfilestep):
        # Convert velocity from m/s to degd
            utmp, vtmp = tools.ms2degd(lon2du, lat2du, VEL.varu[nt, :, :],
                                       VEL.varv[nt, :, :])
            dic['u']['array'].append(utmp[:, :])
            dic['v']['array'].append(vtmp[:, :])
            dic['ums']['array'].append(VEL.varu[nt, :, :])
            dic['vms']['array'].append(VEL.varv[nt, :, :])
            if p.name_h is not None:
                h[nt, :, :] = + VEL.var[nt, :, :]
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
                RV = numpy.zeros(numpy.shape(utmp))
                RV[1: -1, 1: -1] = gfo*(dyvtmp / dylonv - dxutmp / dxlatu)
                dic['rv']['array'].append(filters.gaussian_filter(RV, 4))
            if p.save_S or p.save_OW:
                Sn = numpy.zeros(numpy.shape(utmp))
                Ss = numpy.zeros(numpy.shape(vtmp))
                Sn[1: -1, 1: -1] = gfo*(dyutmp / dylonu - dxvtmp / dxlatv)
                Ss[1: -1, 1: -1] = gfo*(dxutmp / dxlatu + dyvtmp / dylonv)
                dic['sn']['array'].append(Sn)
                dic['ss']['array'].append(Ss)
    for key, value in dic.items():
        dic[key]['array'] = numpy.array(value['array'])
    coord['time'] = numpy.array(coord['time']).flatten()
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
    stop = p.last_date.strftime('%Y%m%d')
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    global_attr = idf_io.global_idf
    global_attr['time_coverage_start'] = p.first_date.strftime(idf_io.idf_fmt)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_listracer_1d(p.output, drifter, p, listTr)


def write_diagnostic_2d(p, data, description='', **kwargs):
    start = p.first_date.strftime('%Y%m%d')
    stop = p.last_date.strftime('%Y%m%d')
    #stop = stop.strftime('%Y%m%d')
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
    #data['time'] = data['time'] * 86400
    #print(data['time'], data.keys())
    write_utils.write_velocity(data, p.output, description=description,
                               unit=const.unit, long_name=const.long_name,
                               meta=global_attr,
                               fill_value=-1e36, **kwargs)


def write_advected_tracer(p, data_out):
    start = p.first_date.strftime('%Y%m%d')
    stop = p.last_date.strftime('%Y%m%d')
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_tracer_2d(p.output, data_out)


def write_params(p, time):
    outformat = '%Y%m%dT%H%M%S'
    stime = time.strftime(outformat)
    output = f'params_output_{stime}'
    write_utils.write_params(p, output)
