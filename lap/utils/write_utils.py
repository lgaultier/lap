'''
Module to read and write data \n
Contains tracer and velocity classes to read  netcdf, cdf and hdf files
Classes are:

'''

import sys
import netCDF4
import numpy
import lap.const as const
import logging
logger = logging.getLogger(__name__)

# -------------------#
#      GENERAL       #
# -------------------#
# TO CHECK


def write_velocity(data, outfile, description='AVISO-like data',
                   unit=const.unit, long_name=const.long_name,
                   fill_value=-1.36e9, **kwargs):
    '''Save netcdf file '''
    lon = data['lon']
    lat = data['lat']

    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(outfile, 'w')
    fid.description = description

    # - Create dimensions
    ndim_lon = 'lon'
    ndim_lat = 'lat'
    ndim_time = 'time'
    dim_lon = len(numpy.shape(lon))
    fid.createDimension(ndim_lat, numpy.shape(lat)[0])
    if dim_lon == 1:
        fid.createDimension(ndim_lon, numpy.shape(lon)[0])
    elif dim_lon == 2:
        fid.createDimension(ndim_lon, numpy.shape(lon)[1])
    else:
        logger.error(f'Wrong number of dimension in longitude, is {dim_lon}'
                     'should be one or two')
        sys.exit(1)
    fid.createDimension(ndim_time, None)

    # - Create and write Variables
    if dim_lon == 1:
        vlon = fid.createVariable('lon', 'f4', (ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lon))
        vlon[:] = lon
        vlat[:] = lat
    elif dim_lon == 2:
        vlon = fid.createVariable('lon', 'f4', (ndim_lat, ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lat, ndim_lon))
        vlon[:, :] = lon
        vlat[:, :] = lat
    vlon.units = unit['lon']
    vlat.units = unit['lat']
    vlon.long_name = long_name['lon']
    vlat.long_name = long_name['lat']
    for key, value in kwargs.items():
        if value.any():
            dim_value = len(value.shape)
            if dim_value == 1:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ),
                                         fill_value=fill_value)
                var[:] = value
            if dim_value == 2:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[0, :, :] = value
            elif dim_value == 3:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[:, :, :] = value
            if str(key) in unit.keys():
                var.units = unit[str(key)]
            if str(key) in long_name.keys():
                var.long_name = long_name[str(key)]
    fid.close()
    return None


def write_tracer_1d(data, namevar, outfile,  unit=const.unit,
                    long_name=const.long_name, fill_value=-1.36e9):
    '''Write new tracer in a cdf file. '''
    lon = data['lon']
    lat = data['lat']
    var = data['tracer']
    time = data['time']
    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(outfile, 'w')
    fid.description = const.glob_attribute['description']

    # - Create dimensions
    dim_part = 'obs'
    dim_time = 'time'
    fid.createDimension(dim_part, numpy.shape(lon)[1])
    fid.createDimension(dim_time, None)
    dim_time = 'time'

    # - Create and write Variables
    vtime = fid.createVariable('time', 'f4', (dim_time))
    vlon = fid.createVariable('lon', 'f4', (dim_time, dim_part))
    vlat = fid.createVariable('lat', 'f4', (dim_time, dim_part))
    vtra = fid.createVariable(namevar, 'f4', (dim_part), fill_value=fill_value)
    vtime[:] = time
    vtime.units = "days"
    vlon[:, :] = lon
    vlon.units = unit['lon']
    vlon.long_name = long_name['lon']
    vlat[:, :] = lat
    vlat.units = unit['lat']
    vlat.long_name = long_name['lat']
    vtra[:] = var
    vtra.units = unit['T']
    vtra.long_name = long_name['T']
    fid.close()
    return None


def write_listracer_1d(wfile, T, p, listTr):
    '''Write list of tracer in a cdf file. '''
    fill_value = p.fill_value
    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(wfile, 'w')
    fid.description = "Drifter advected by lagrangian tool"

    # - Create dimensions
    dim_part = 'obs'
    dim_time = 'time'
    dim_time_hr = 'time_hr'
    fid.createDimension(dim_part, numpy.shape(T['lon_lr'])[1])
    fid.createDimension(dim_time, None)
    if p.save_traj is True:
        fid.createDimension(dim_time_hr, numpy.shape(T['lon_hr'])[0])

        # - Create and write Variables
        vlon = fid.createVariable('lon_hr', 'f4', (dim_time_hr, dim_part))
        vlon[:, :] = T['lon_hr']
        vlon.units = "deg E"
        vlon.long_name = 'High temporal resolution longitude'
        vlat = fid.createVariable('lat_hr', 'f4', (dim_time_hr, dim_part))
        vlat[:, :] = T['lat_hr']
        vlat.units = "deg N"
        vlat.long_name = 'High temporal resolution latitude'
        vmask = fid.createVariable('mask_hr', 'f4', (dim_time_hr, dim_part))
        vmask[:, :] = T['mask_hr']
        vtime = fid.createVariable('time_hr', 'f4', (dim_time_hr))

        vtime[:] = T.time_hr
        vtime.units = "days"
        vtime.long_name = ('High temporal resolution time from the reference'
                           'time')
        if p.save_U is True:
            vu = fid.createVariable('zonal_velocity', 'f4', (dim_time_hr,
                                    dim_part), fill_value=fill_value)
            vu[:, :] = T['vel_u_hr']
            vu.units = 'm/s'
            vu.long_name = 'High temporal resolution zonal velocity'
        if p.save_V is True:
            vv = fid.createVariable('meridional_velocity', 'f4',
                                    (dim_time_hr, dim_part),
                                    fill_value=fill_value)
            vv[:, :] = T['vel_v_hr']
            vv.units = 'm/s'
            vv.long_name = 'High temporal resolution meridional velcity'
        if p.save_S is True:
            vS = fid.createVariable('Strain', 'f4', (dim_time_hr, dim_part),
                                    fill_value=fill_value)
            vS[:, :] = T['S_hr']
            vS.units = 's-1'
            vS.long_name = 'Strain'
        if p.save_RV is True:
            vRV = fid.createVariable('Vorticity', 'f4', (dim_time_hr,
                                     dim_part), fill_value=fill_value)
            vRV[:, :] = T['RV_hr']
            vRV.units = 's-1'
            vRV.long_name = 'Relative Vorticity'
        if p.save_OW is True:
            vOW = fid.createVariable('OW', 'f4', (dim_time_hr, dim_part),
                                     fill_value=fill_value)
            vOW[:, :] = T['OW_hr']
            vOW.units = 's-1'
            vOW.long_name = 'Okubo-Weiss'

    vlon = fid.createVariable('lon', 'f4', (dim_time, dim_part),
                              fill_value=fill_value)
    vlat = fid.createVariable('lat', 'f4', (dim_time, dim_part),
                              fill_value=fill_value)
    vtime = fid.createVariable('time', 'f4', (dim_time), fill_value=fill_value)
    vlon[:, :] = T['lon_lr']
    vlon.units = "deg E"
    vlat[:, :] = T['lat_lr']
    vlat.units = "deg N"
    vtime[:] = T['time']
    vlon.units = "days"
    vmask = fid.createVariable('mask_lr', 'f4', (dim_time, dim_part))
    vmask[:, :] = T['mask_lr']
    if p.list_tracer is not None:
        for i in range(len(listTr)):
            vtra = fid.createVariable(p.listtracer[i], 'f4', (dim_time,
                                      dim_part), fill_value=fill_value)
            vtra[:, :] = listTr[i].newvar
    fid.close()
    return None


# TO CHECK
def write_aviso(wfile, VEL, t, fill_value=-1.e36):

    '''Write AVISO data, Strain, Vorticity and Okubo Weiss in a cdf file. '''
    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(wfile, 'w')
    fid.description = ('Strain, Vorticity and OW parameters from AVISO'
                       ' velocity fields')

    # - Create dimensions
    dim_time = 'time'
    dim_lon = 'lon'
    dim_lat = 'lat'
    fid.createDimension(dim_lon, numpy.shape(VEL.lon)[0])
    fid.createDimension(dim_lat, numpy.shape(VEL.lat)[0])
    fid.createDimension(dim_time, None)

    # - Create and write Variables
    vlon = fid.createVariable('lon', 'f4', (dim_lon))
    vlat = fid.createVariable('lat', 'f4', (dim_lat))
    vtime = fid.createVariable('time', 'f4', (dim_time))
    vlon[:] = VEL.lon
    vlon.units = "deg E"
    vlon.long_name = "Longitudes"
    vlat[:] = VEL.lat
    vlat.units = "deg N"
    vlat.long_name = "Latitudes"
    vtime[:] = VEL.time
    vtime.units = "Julian days (CNES)"
    vH = fid.createVariable('H', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vH[0, :, :] = VEL.h[:, :]
    else:
        vH[0, :, :] = VEL.h[t, :, :]
    vH.units = 'm'
    vH.long_name = 'Sea Surface Height'
    vu = fid.createVariable('U', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vu[0, :, :] = VEL.us[:, :]
    else:
        vu[0, :, :] = VEL.us[t, :, :]
    vu.units = 'm/s'
    vu.long_name = 'zonal velocity'
    vv = fid.createVariable('V', 'f4', (dim_time, dim_lat, dim_lon),
                            fill_value=fill_value)
    if t is None:
        vv[0, :, :] = VEL.vs[:, :]
    else:
        vv[0, :, :] = VEL.vs[t, :, :]
    vv.units = 'm/s'
    vv.long_name = 'meridional velocity'
    if hasattr(VEL, 'Sn'):
        vSn = fid.createVariable('Sn', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vSn[0, :, :] = VEL.Sn[t, :, :]
        vSn.units = "s-1"
        vSn.long_name = "Normal Strain"
    if hasattr(VEL, 'Ss'):
        vSs = fid.createVariable('Ss', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vSs[0, :, :] = VEL.Ss[t, :, :]
        vSs.units = "s-1"
        vSs.long_name = "Shear Strain"
    if hasattr(VEL, 'RV'):
        vRV = fid.createVariable('Vorticity', 'f4', (dim_time, dim_lat,
                                 dim_lon), fill_value=fill_value)
        vRV[0, :, :] = VEL.RV[t, :, :]
        vRV.units = "s-1"
        vRV.long_name = 'Relative Vorticity'
    if hasattr(VEL, 'OW'):
        vOW = fid.createVariable('OW', 'f4', (dim_time, dim_lat, dim_lon),
                                 fill_value=fill_value)
        vOW[0, :, :] = VEL.OW[t, :, :]
        vOW.units = "s-1"
        vOW.long_name = 'Okubo-Weiss'
    fid.close()
    return None
