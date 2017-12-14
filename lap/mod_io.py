import numpy
import os
import sys
import lap.utils.read_utils as read_utils
import lap.utils.write_utils as write_utils
import lap.mod_tools as mod_tools
import lap.const as const
from scipy.ndimage import filters
from scipy import interpolate
from math import pi
import logging
logger = logging.getLogger(__name__)

# #############################
# #        READ TRACER       ##
# #############################


def get_regular_tracer(p, tracer, get_coord=True):
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


def read_list_tracer(p, dict_tracer):
    listTr = []
    tref = p.tref
    # Loop on all tracer provided in the parameter file
    for tra in p.list_tracer:
        logger.info(f'Reading {tra}')
        i = 0
        tref = p.first_day
        # For each tracer Find first available data
        year, month, day = mod_tools.jj2date(tref)
        tra_name = dict_tracer['name'][tra]
        tra_step = dict_tracer['tstep'][tra]
        tra_dir = dict_tracer['dir'][tra]
        tra_var = dict_tracer['var'][tra]
        filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
        pathfilet = os.path.join(p.tracer_input_dir, tra_dir, filet)
        # If no data is found, look for the previous step
        while ((not os.path.exists(pathfilet))
                and abs(tref - p.first_day) <= tra_step):
            tref -= 1
            year, month, day = mod_tools.jj2date(tref)
            filet = f'{tra_name}{year:02d}{month:02d}{day:02d}.nc'
        # If no data is found in the previous step, there is no available data
        if abs(tref - p.first_day) > tra_step:
            logger.error(f'Tracer files {tra} ({filet}) are not found')
            sys.exit(1)
        # Build list of tracer datasets which have to be collocated
        nstep = p.tadvection / abs(p.tadvection) * tra_step
        tstop = tref + p.tadvection + nstep
        listdate = list(numpy.arange(tref, tstop, numpy.floor(nstep)))
        num_date = len(listdate)
        # Read all necessary time steps
        for jdate in listdate:
            year, month, day = mod_tools.jj2date(jdate)
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


def read_list_grid(p, dict_tracer):
    listTr = []
    tref = p.first_day
    # Loop on all tracer grid provided in the parameter file
    for tra in p.list_grid:
        logger.info(f'Reading {tra}')
        jdate = tref
        # For each tracer Find first available data
        year, month, day = mod_tools.jj2date(jdate)
        tra_name = dict_tracer['name'][tra]
        tra_step = dict_tracer['tstep'][tra]
        tra_dir = dict_tracer['dir'][tra]
        tra_var = dict_tracer['var'][tra]
        filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
        pathfilet = os.path.join(p.tracer_input_dir, tra_dir, filet)
        # If no data is found, look for the previous step
        while (not os.path.exists(pathfilet)
               and abs(jdate - p.first_day) <= tra_step):
            jdate -= 1
            year, month, day = mod_tools.jj2date(jdate)
            filet = f'{tra_name}{year:04d}{month:02d}{day:02d}.nc'
            pathfilet = os.path.join(p.DIR, tra_dir, filet)
        # If no data is found in the previous step, there is no available data
        if abs(jdate - p.first_day) > tra_step:
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
    Tr = read_utils.MODISCDF(file=p.fileg, var=p.name_tracer, time='time')
    get_regular_tracer(p, Tr, get_coord=True)
    return Tr


# TODO is it used?
def Read_listgrid_netcdf(p, dict_tracer, jdate):
    listTr = []
    for tra in p.list_grid:
        year, month, day = mod_tools.jj2date(jdate)
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


def make_mask(p):
    # mask_grid
    filename = os.path.join(p.vel_input_dir, p.list_vel[0])
    if not os.path.isfile(filename):
        logger.error(f'File {filename} not found')
        sys.exit(1)
    if p.vel_format == 'regular_netcdf':
        VEL = read_utils.velocity_netcdf(filename=filename, varu=p.name_u,
                                         varv=p.name_u, lon=p.name_lon,
                                         lat=p.name_lat, box=p.box)

    elif p.vel_format == 'nemo':
        VEL = read_utils.nemo_netcdf(filename=filename, varu=p.name_u,
                                     varv=p.name_v, lon=p.name_lon,
                                     lat=p.name_lat, box=p.box)

    VEL.read_coord()
    VEL.read_vel()
    # VEL.lon = (VEL.lon + 360.) % 360.
    lon2du, lat2du = numpy.meshgrid(VEL.lon, VEL.lat)
    masku = VEL.varu
    masku[abs(masku) > 50.] = numpy.nan
    masku[abs(VEL.varv) > 50.] = numpy.nan
    Teval = interpolate.RectBivariateSpline(VEL.lat, VEL.lon,
                                            numpy.isnan(masku), kx=1, ky=1,
                                            s=0)
    return Teval


def make_grid(p):
    try:
        lon0, lon1, dlon, lat0, lat1, dlat = list(p.parameter_grid)
    except:
        logger.error('Grid parameters must be specified in parameter_grid '
                     '(lon0, lon1, dlon, lat0, lat1, dlat)')
        sys.exit(1)
    # Build grid
    Tr = read_utils.initclass()
    lontmp = numpy.linspace(lon0, lon1, int((lon1 - lon0) / dlon))
    lattmp = numpy.linspace(lat0, lat1, int((lat1 - lat0) / dlat))
    Tr.lon, Tr.lat = numpy.meshgrid(lontmp, lattmp)
    # TODO PROVIDE A MASK FILE
    Teval = make_mask(p)
    masktmp = Teval(lattmp, lontmp)
    shape_tra = numpy.shape(Tr.lon)
    masktmp = masktmp.reshape(shape_tra)
    # TODO create real mask

    Tr.mask = (masktmp > 0)
    return Tr


# #############################
# #        READ VEL         ##
# #############################
def read_velocity(p, get_time=None):
    # Read velocity
    filename = os.path.join(p.vel_input_dir, p.list_vel[0])
    if not os.path.isfile(filename):
        logger.error(f'File {filename} not found')
        sys.exit(1)
    if p.vel_format == 'regular_netcdf':
        VEL = read_utils.velocity_netcdf(filename=filename, varu=p.name_u,
                                         varv=p.name_v, lon=p.name_lon,
                                         lat=p.name_lat, box=p.box)
    elif p.vel_format == 'nemo':
        VEL = read_utils.nemo_netcdf(filename=filename, varu=p.name_u,
                                     varv=p.name_v, lon=p.name_lon,
                                     lat=p.name_lat, box=p.box)
    else:
        logger.error(f'{p.vel_format} format is not handled')
        sys.exit(1)
    VEL.read_coord()
    # TODO check this format?
    if len(VEL.lon.shape) == 2:
        lon2du, lat2du = (VEL.lon, VEL.lat)
        lon2dv, lat2dv = (VEL.lon, VEL.lat)
    else:
        lon2du, lat2du = numpy.meshgrid(VEL.lon, VEL.lat)
        lon2dv, lat2dv = numpy.meshgrid(VEL.lon, VEL.lat)
    # Intialize empty matrices
    if p.stationary is True:
        num_steps = 1
    else:
        num_steps = int(abs(p.tadvection) / p.vel_step) + 1
    shape_vel_u = (num_steps, numpy.shape(lat2du)[0],
                   numpy.shape(lon2du)[1])
    shape_vel_v = (num_steps, numpy.shape(lat2dv)[0],
                   numpy.shape(lon2dv)[1])
    u = numpy.zeros(shape_vel_u)
    v = numpy.zeros(shape_vel_v)
    usave = numpy.zeros(shape_vel_u)
    vsave = numpy.zeros(shape_vel_v)
    h = numpy.zeros(shape_vel_u)
    Sn = numpy.zeros(shape_vel_u)
    Ss = numpy.zeros(shape_vel_u)
    RV = numpy.zeros(shape_vel_u)
    for t in range(0, num_steps):
        filename = os.path.join(p.vel_input_dir, p.list_vel[t])
        if not os.path.isfile(filename):
            logger.error(f'File {filename} not found')
            sys.exit(1)
        if p.vel_format == 'regular_netcdf':
            VEL = read_utils.velocity_netcdf(filename=filename, varu=p.name_u,
                                             varv=p.name_v, lon=p.name_lon,
                                             lat=p.name_lat, var=p.name_h,
                                             box=p.box)
        elif p.vel_format == 'nemo':
            VEL = read_utils.nemo_netcdf(filename=filename, varu=p.name_u,
                                         varv=p.name_v, lon=p.name_lon,
                                         lat=p.name_lat, var=p.name_h,
                                         box=p.box)
        else:
            logger.error('Undefined Velocity file type')
            sys.exit(1)
        if t == 0:
            VEL.read_vel(size_filter=p.vel_filter)
            slice_x = VEL.slice_x
            slice_y = VEL.slice_y
        else:
            VEL.read_vel(size_filter=p.vel_filter, slice_xy=(slice_x, slice_y))
        # TODO: to change
        try:
            VEL.read_var(size_filter=p.vel_filter)
        except:
            pass
        if t == num_steps - 1:
            VEL.read_coord()
            VEL.Vlonu = numpy.mod(VEL.lon[:] + 360., 360.)
            VEL.Vlatu = VEL.lat[:]
            VEL.Vlonv = numpy.mod(VEL.lon[:] + 360., 360.)
            VEL.Vlatv = VEL.lat[:]
        # # TODO A CHANGER , Initialize u and v here
        # VEL.masku=numpy.ma.masked_where(VEL.varu), VEL.varu[numpy.isnan(
        # VEL.varu) and (abs(VEL.varu)>10) and (abs(VEL.varv)>10)]
        # VEL.varu *=1.3
        # VEL.varv *=1.3
        mask = (numpy.ma.getmaskarray(VEL.varu)
                | numpy.ma.getmaskarray(VEL.varv)
                | numpy.isnan(VEL.varu) | numpy.isnan(VEL.varv))
        # VEL.varu[numpy.where(abs(VEL.varu) > 10)] = 0
        # VEL.varv[numpy.where(abs(VEL.varv) > 10)] = 0
        try:
            VEL.var[numpy.where(abs(VEL.var) > 100)] = numpy.nan
        except:
            pass
        #utmp, vtmp = mod_tools.convert(lon2du, lat2du, VEL.varu, VEL.varv)
        utmp = VEL.varu /(111.10**3 * numpy.cos(numpy.deg2rad(lat2du)))
        vtmp = VEL.varv / 111.10**3
        VEL.varu[mask] = 0
        VEL.varv[mask] = 0

        # Compute Strain Relative Vorticity and Okubo Weiss
        if p.save_S or p.save_RV or p.save_OW:
            gfo = (9.81 / numpy.cos(numpy.mean((lat2du + lat2dv)/2)*pi/180.)
                   / (2*7.2921*10**(-4)))
            dxlatu = (lat2du[2:, 1: -1] - lat2du[: -2, 1: -1])
            dylonu = (lon2du[1: -1, 2:] - lon2du[1: -1, : -2])
            dxlatv = (lat2dv[2:, 1: -1] - lat2dv[:-2, 1: -1])
            dylonv = (lon2dv[1: -1, 2:] - lon2dv[1: -1, : -2])
            dyutmp = utmp[1: -1, 2:] - utmp[1: -1, : -2]
            dyvtmp = vtmp[1: -1, 2:] - vtmp[1: -1, : -2]
            dxutmp = utmp[2:, 1: -1] - utmp[:-2, 1: -1]
            dxvtmp = vtmp[2:, 1: -1] - vtmp[:-2, 1: -1]
        if p.save_RV or p.save_OW:
            RV[t, 1: -1, 1: -1] = gfo*(dyvtmp / dylonv - dxutmp / dxlatu)
        if p.save_S or p.save_OW:
            Sn[t, 1: -1, 1: -1] = gfo*(dyutmp / dylonu - dxvtmp / dxlatv)
            Ss[t, 1: -1, 1: -1] = gfo*(dxutmp / dxlatu + dyvtmp / dylonv)
        u[t, :, :] = utmp
        v[t, :, :] = vtmp
        usave[t, :, :] = VEL.varu
        vsave[t, :, :] = VEL.varv
        try:
            h[t, :, :] = VEL.var
        except:
            pass
    step = numpy.sign(p.tadvection) * p.vel_step
    stop = p.first_day + num_steps * numpy.sign(p.tadvection)
    VEL.time = numpy.arange(p.first_day, stop, step)
    VEL.u = u
    VEL.v = v
    VEL.us = usave
    VEL.vs = vsave
    VEL.h = h
    if p.save_S or p.save_OW:
        VEL.Ss = filters.gaussian_filter(Ss, 4)
        VEL.Sn = filters.gaussian_filter(Sn, 4)
    if p.save_RV or p.save_OW:
        VEL.RV = filters.gaussian_filter(RV, 4)
    return VEL


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


def write_drifter(p, drifter, listTr):
    start = int(p.first_day)
    stop = int(p.first_day + p.tadvection)
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_listracer_1d(p.output, drifter, p, listTr)


def write_diagnostic_2d(p, data, description='', **kwargs):
    start = int(data['time'][0])
    stop = int(data['time'][-1])
    file_default = f'{p.diagnostic[0]}_{start}_{stop}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_velocity(data, p.output, description=description,
                               unit=const.unit, long_name=const.long_name,
                               fill_value=-1e36, **kwargs)


def write_advected_tracer(p, data_out):
    start = int(p.first_day)
    stop = int(p.first_day + p.tadvection)
    file_default = f'Advection_{start}_{stop}_K{p.K}s{p.sigma}.nc'
    default_output = os.path.join(p.output_dir, file_default)
    p.output = getattr(p, 'output', default_output)
    write_utils.write_tracer_2d(p.output, data_out)
