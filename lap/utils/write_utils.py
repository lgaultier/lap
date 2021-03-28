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

def write_params(params, pfile):
    """ Write parameters that have been selected to run swot_simulator. """
    with open(pfile, 'w') as f:
        for key in dir(params):
            if not key[0:2] == '__':
                f.write('{} = {}\n'.format(key, params.__dict__[key]))


def write_velocity(data, outfile, description='AVISO-like data',
                   unit=const.unit, long_name=const.long_name,
                   fill_value=-1.36e9, meta={}, **kwargs):
    '''Save netcdf file '''
    lon = data['lon']
    lat = data['lat']

    # - Open Netcdf file in write mode
    fid = netCDF4.Dataset(outfile, 'w')
    fid.description = description

    # - Create dimensions
    ndim_lon = 'lon'
    ndim_lat = 'lat'
    print(data.keys())
    isidf = False
    if 'lon_gcp' in data.keys():
        isidf = True
    if isidf:
        ndim_glon = 'lon_gcp'
        ndim_glat = 'lat_gcp'
    ndim_time = 'time'
    ndim_time1 = 'time1'
    dim_lon = len(numpy.shape(lon))
    fid.createDimension(ndim_lat, numpy.shape(lat)[0])
    if isidf:
        fid.createDimension(ndim_glat, numpy.shape(data['lat_gcp'])[0])
    if dim_lon == 1:
        fid.createDimension(ndim_lon, numpy.shape(data['lon'])[0])
        if isidf:
            fid.createDimension(ndim_glon, numpy.shape(data['lon_gcp'])[0])
    elif dim_lon == 2:
        fid.createDimension(ndim_lon, numpy.shape(lon)[1])
        if isidf:
            fid.createDimension(ndim_glon, numpy.shape(data['lon_gcp'])[1])
    else:
        logger.error(f'Wrong number of dimension in longitude, is {dim_lon}'
                     'should be one or two')
        sys.exit(1)
    fid.createDimension(ndim_time, None)
    fid.createDimension(ndim_time1, 1)
    print(' shape', numpy.shape(data['index_lon_gcp']), numpy.shape(data['lon_gcp']))

    # - Create and write Variables
    if dim_lon == 1:
        vlon = fid.createVariable('lon', 'f4', (ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lat))
        vlon[:] = lon
        vlat[:] = lat
        if isidf:
            vglon = fid.createVariable('lon_gcp', 'f4', (ndim_glon))
            vglat = fid.createVariable('lat_gcp', 'f4', (ndim_glat))
            vilon = fid.createVariable('index_lon_gcp', 'f4', (ndim_glon))
            vilat = fid.createVariable('index_lat_gcp', 'f4', (ndim_glat))
            vglon[:] = data['lon_gcp']
            vglat[:] = data['lat_gcp']
            vilon[:] = data['index_lon_gcp']
            vilat[:] = data['index_lat_gcp']
    elif dim_lon == 2:
        vlon = fid.createVariable('lon', 'f4', (ndim_lat, ndim_lon))
        vlat = fid.createVariable('lat', 'f4', (ndim_lat, ndim_lon))
        vlon[:, :] = lon
        vlat[:, :] = lat
        if 'gcp' in data.keys():
            vglon = fid.createVariable('lon_gcp', 'f4', (ndim_glat, ndim_glon))
            vglat = fid.createVariable('lat_gcp', 'f4', (ndim_glat, ndim_glon))
            vilon = fid.createVariable('index_lon_gcp', 'f4', (ndim_glat,
                                                               ndim_glon))
            vilat = fid.createVariable('index_lat_gcp', 'f4', (ndim_glat,
                                                               ndim_glon))
            vglon[:, :] = data['lon_gcp']
            vglat[:, :] = data['lat_gcp']
            vilon[:, :] = data['index_lon_gcp']
            vilat[:, :] = data['index_lat_gcp']
    vlon.units = unit['lon']
    vlat.units = unit['lat']
    vlon.long_name = long_name['lon']
    vlat.long_name = long_name['lat']
    if isidf:
        vglon.units = unit['lon']
        vglat.units = unit['lat']
        vglon.long_name = f'ground control point {long_name["lon"]}'
        vglat.long_name = f'ground control point {long_name["lat"]}'
        _text = 'index of ground control point in'
        vilon.long_name = f'{_text} {long_name["lon"]} dimension'
        vilat.long_name = f'{_text} {long_name["lat"]} dimension'
    for key, value in kwargs.items():
        if value.any():
            dim_value = len(value.shape)
            if isidf:
                bin_key = f'{key}_bin'
                bvalue, scale, offset = pack_as_ubytes(value, fill_value)
            if dim_value == 1:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ),
                                         fill_value=fill_value)
                var[:] = value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time, ),
                                              fill_value=numpy.ubyte(255))
                    bvar[:] = bvalue

            if dim_value == 2:
                var = fid.createVariable(str(key), 'f4', (ndim_time1, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[0, :, :] = value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time1,
                                              ndim_lat, ndim_lon),
                                              fill_value=numpy.ubyte(255))
                    bvar[0, :, :] = bvalue
            elif dim_value == 3:
                var = fid.createVariable(str(key), 'f4', (ndim_time, ndim_lat,
                                         ndim_lon), fill_value=fill_value)
                var[:, :, :] = value
                if isidf:
                    bvar = fid.createVariable(bin_key, 'u1', (ndim_time,
                                              ndim_lat, ndim_lon),
                                              fill_value=numpy.ubyte(255))
                    bvar[:, :, :] = bvalue
            if str(key) in unit.keys():
                var.units = unit[str(key)]
                if isidf:
                    bvar.units = unit[str(key)]
            if str(key) in long_name.keys():
                var.long_name = long_name[str(key)]
                if isidf:
                    bvar.long_name = long_name[str(key)]
            if isidf:
                bvar.scale_factor = scale
                bvar.add_offset = offset
                bvar.valid_min = 0
                bvar.valid_max = 254

    for key, value in meta.items():
        setattr(fid, key, value)
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
    exportables = [k for k in dir(p) if not k.startswith('__')]
    str_params = [f'{k}={getattr(p, k)}' for k in exportables]
    fid.comment = ' '.join(str_params)

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

        vtime[:] = T['time_hr']
        vtime.units = "days"
        vtime.long_name = ('High temporal resolution time from the reference'
                           'time')
        if p.save_U is True:
            vu = fid.createVariable('zonal_velocity', 'f4', (dim_time_hr,
                                    dim_part), fill_value=fill_value)
            vu[:, :] = T['u_hr']
            vu.units = 'm/s'
            vu.long_name = 'High temporal resolution zonal velocity'
        if p.save_V is True:
            vv = fid.createVariable('meridional_velocity', 'f4',
                                    (dim_time_hr, dim_part),
                                    fill_value=fill_value)
            vv[:, :] = T['v_hr']
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
    vtime[:] = VEL.time[t]
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


def pack_as_ubytes(var, fill_value):
    vmin = numpy.nanmin(var)
    vmax = numpy.nanmax(var)
    vmin = max(vmin, -100)
    vmax = min(vmax, 400)
    nan_mask = ((numpy.ma.getmaskarray(var)) | (numpy.isnan(var))
                 | (var == fill_value))

    offset, scale = vmin, (float(vmax) - float(vmin)) / 254.0
    if vmin == vmax:
        scale = 1.0

    numpy.clip(var, vmin, vmax, out=var)

    # Required to avoid runtime warnings on masked arrays wherein the
    # division of the _FillValue by the scale cannot be stored by the dtype
    # of the array
    if isinstance(var, numpy.ma.MaskedArray):
        mask = numpy.ma.getmaskarray(var).copy()
        var[numpy.where(mask)] = vmin
        _var = (numpy.ma.getdata(var) - offset) / scale
        var.mask = mask  # Restore mask to avoid side-effects
    else:
        _var = (var - offset) / scale

    result = numpy.round(_var).astype('ubyte')
    result[numpy.where(nan_mask)] = 255
    return result, scale, offset
