'''
Module to read and write data \n
Contains tracer and velocity classes to read  netcdf, cdf and hdf files
Classes are:

'''

import sys
import netCDF4
import numpy
import scipy
from scipy.ndimage import filters
from scipy.interpolate import griddata
import logging
logger = logging.getLogger(__name__)

# -------------------#
#      GENERAL       #
# -------------------#


def read_coordinates(filename, nlon, nlat, dd=True, subsample=1):
    ''' General routine to read coordinates in a netcdf file. \n
    Inputs are file name, longitude name, latitude name. \n
    Outputs are longitudes and latitudes (2D arrays).'''

    # - Open Netcdf file
    fid = netCDF4.Dataset(filename, 'r')

    # - Read 1d or 2d coordinates
    vartmp = fid.variables[nlat]
    if len(vartmp.shape) == 1:
        lon_tmp = numpy.array(fid.variables[nlon][:]).squeeze()
        lat_tmp = numpy.array(fid.variables[nlat][:]).squeeze()
        lon_tmp = lon_tmp[::subsample]
        lat_tmp = lat_tmp[::subsample]
        if dd is True:
            lon, lat = numpy.meshgrid(lon_tmp, lat_tmp)
        else:
            lon = lon_tmp
            lat = lat_tmp
    elif len(vartmp.shape) == 2:
        lon = numpy.array(fid.variables[nlon][:, :]).squeeze()
        lat = numpy.array(fid.variables[nlat][:, :]).squeeze()
        lon = lon[::subsample, ::subsample]
        lat = lat[::subsample, ::subsample]
    else:
        logger.error('unknown dimension for lon and lat')
        sys.exit(1)
    fid.close()
    return lon, lat


def read_var(filename, var, index=None, time=0, depth=0, subsample=1,
             missing_value=None):
    ''' General routine to read variables in a netcdf file. \n
    Inputs are file name, variable name, index=index to read part
    of the variables, time=time to read a specific time, depth=depth to read a
    specific depth, missing_value=nan value '''

    # - Open Netcdf file
    fid = netCDF4.Dataset(filename, 'r')

    # - Check dimension of variable
    vartmp = fid.variables[var]
    ndim = len(vartmp.shape)
    # - Exit if number of dimension is not supported
    if ndim > 4:
        logger.error(f'Variable {var} has a number of dimension not supported'
                     f'{ndim}')
        sys.exit(1)

    # - Read variable
    T = numpy.ma.array(fid.variables[var][:], dtype='float16')
    if ndim == 1:
        if index is None:
            T = T[index]
        T = T[::subsample]
    elif ndim == 2:
        if index is not None:
            T = T[index]
        T = T[::subsample, ::subsample]
    elif ndim == 3:
        T = T[:, index]
        if time is not None:
            T = T[time, :, :]
        T = T[:, ::subsample, ::subsample]
    elif ndim == 4:
        T = T[:, :, index]
        T = T[:, depth, :, :].squeeze()
        if time is not None:
            T = T[time, :, :]
        T = T[:, ::subsample, ::subsample]
    fid.close()

    # - Mask value that are NaN
    if missing_value is not None:
        T[numpy.where(T == missing_value)] = numpy.nan
    T._sharedmask = False
    return T


def read_time(filename, ntime, time=None):
    # - Open Netcdf file
    fid = netCDF4.Dataset(filename, 'r')

    # - Read time variable and units
    vtime = fid.variables[ntime][:]
    if time is not None:
        vtime = vtime[time]
    time_units = fid[ntime].units

    # - Try to read calendar
    try:
        time_cal = fid[ntime].calendar
    except AttributeError:
        logger.warn('No calendar in time attributes, standard calendar used')
        time_cal = u"gregorian"
    vtime = netCDF4.num2date(vtime, units=time_units, calendar=time_cal)
    fid.close()
    return vtime


class velocity_netcdf():
    ''' read and write velocity data that is on a regular netcdf grid. '''
    def __init__(self, filename=None, lon='lon', lat='lat', time='time',
                 varu='U', varv='V', var='H', missing_value=None, box=None):
        self.file = filename
        self.nlon = lon
        self.nlat = lat
        self.ntime = time
        self.nvaru = varu
        self.nvarv = varv
        self.nvar = var
        self.missing_value = missing_value
        self.box = box

    def read_coord(self):
        '''Read data coordinates'''
        self.lon0, self.lat0 = read_coordinates(self.file, self.nlon,
                                                self.nlat, dd=False)
        self.lon0 = numpy.mod(self.lon0 + 360, 360)
        if self.box is not None:
            box = self.box
            if len(box) == 4:
                indx = numpy.where((self.lon0[:] > box[0])
                                   & (self.lon0[:] < box[1]))[0]
                self.slice_x = slice(indx[0], indx[-1] + 1)
                indy = numpy.where((self.lat0[:] > box[2])
                                   & (self.lat0[:] < box[3]))[0]
                self.slice_y = slice(indy[0], indy[-1] + 1)
            else:
                logger.error('provide a valid box [lllon, urlon, lllat, urlat]'
                             )
                sys.exit(1)
        else:
            self.slice_x = None
            self.slice_y = None

        if self.box is not None:
            self.lon0 = self.lon0[self.slice_x]
            self.lat0 = self.lat0[self.slice_y]
        self.lon, self.lat = (self.lon0[:], self.lat0[:])
        return None

    def read_vel(self, index=None, time=None, depth=0, missing_value=None,
                 size_filter=None, slice_xy=None):
        '''Read data velocity'''
        if slice_xy is None:
            self.read_coord()
        else:
            self.slice_x = slice_xy[0]
            self.slice_y = slice_xy[1]
        varu = read_var(self.file, self.nvaru, index=None, time=time,
                        depth=depth, missing_value=missing_value)
        varv = read_var(self.file, self.nvarv, index=None, time=time,
                        depth=depth, missing_value=missing_value)
        if size_filter is not None:
            varu = filters.gaussian_filter(varu, sigma=size_filter)
            varv = filters.gaussian_filter(varv, sigma=size_filter)
        if self.box is not None:
            if slice_xy is None:
                self.read_coord()
            else:
                self.slice_x = slice_xy[0]
                self.slice_y = slice_xy[1]
            varu = varu[:, self.slice_y, self.slice_x]
            varv = varv[:, self.slice_y, self.slice_x]
        else:
            self.slice_x = None
            self.slice_y = None
        self.varu = varu
        self.varv = varv
        self.varu._sharedmask = False
        self.varv._sharedmask = False
        return None

    def read_var(self, index=None, depth=0, time=None, missing_value=None,
                 slice_xy=None, size_filter=None):
        '''Read data variable'''
        if slice_xy is None:
            self.slice_x, self.slice_y = self.read_coord()
        else:
            self.slice_x = slice_xy[0]
            self.slice_y = slice_xy[1]
        var = read_var(self.file, self.nvar, index=None, time=time,
                       depth=depth, missing_value=missing_value)
        if self.box is not None:
            if slice_xy is None:
                self.slice_x, self.slice_y = self.read_coord()
            else:
                self.slice_x = slice_xy[0]
                self.slice_y = slice_xy[1]
            var = var[self.slice_y, self.slice_x]
        self.var = var
        return None

    def read_time(self, time=None):
        '''Read data time'''
        self.time = read_time(self.file, self.ntime, time=time)
        return None


class nemo():
    ''' Nemo netcdf class: read and write NEMO data that
    are in netcdf.'''
    def __init__(self, filename=None, lon='nav_lon', lat='nav_lat',
                 time='time', varu='vozocrtx', varv='vomecrty',
                 var='sossheig', box=None, subsample=1):
        self.file = filename
        self.nlon = lon
        self.nlat = lat
        self.ntime = time
        self.nvaru = varu
        self.nvarv = varv
        self.nvar = var
        self.box = box
        self.ss = subsample

    def read_coord(self):
        '''Read data coordinates'''
        self.lon0, self.lat0 = read_coordinates(self.file,  self.nlon,
                                                self.nlat, dd=False,
                                                subsample=self.ss)
        self.lon0 = numpy.mod(self.lon0 + 360, 360)
        if self.box is not None:
            box = self.box
            if len(box) == 4:
                indx = numpy.where((self.lon0[0, :] > box[0])
                                   & (self.lon0[0, :] < box[1]))[0]
                self.slice_x = slice(indx[0], indx[-1] + 1)
                indy = numpy.where((self.lat0[:, 0] > box[2])
                                   & (self.lat0[:, 0] < box[3]))[0]
                self.slice_y = slice(indy[0], indy[-1] + 1)
            else:
                logger.error('provide a valid box [lllon, urlon, lllat, urlat]'
                             )
                sys.exit(1)
        else:
            self.slice_x = None
            self.slice_y = None

        if box is not None:
            self.lon0 = self.lon0[self.slice_y, self.slice_x]
            self.lat0 = self.lat0[self.slice_y, self.slice_x]
        self.lon2d, self.lat2d = numpy.meshgrid(self.lon0[0, :],
                                                self.lat0[:, 0])
        self.lon, self.lat = (self.lon0[0, :], self.lat0[:, 0])
        return None

    def read_vel(self, index=None, time=0, missing_value=None,
                 size_filter=None, slice_xy=None):
        '''Read data velocity'''
        #if slice_xy is None:
        self.read_coord()
        #else:
        #    self.slice_x = slice_xy[0]
        #    self.slice_y = slice_xy[1]
        varu = read_var(self.file, self.nvaru, index=index, time=0, depth=0,
                        missing_value=missing_value, subsample=self.ss)
        varv = read_var(self.file, self.nvarv, index=index, time=0, depth=0,
                        missing_value=missing_value, subsample=self.ss)
        if size_filter is not None:
            varu = filters.gaussian_filter(varu, sigma=size_filter)
            varv = filters.gaussian_filter(varv, sigma=size_filter)
        if self.box is not None:
            #if slice_xy is None:
            #    self.read_coord()
            #else:
            #    self.slice_x = slice_xy[0]
            #    self.slice_y = slice_xy[1]
            varu = varu[self.slice_y, self.slice_x]
            varv = varv[self.slice_y, self.slice_x]
        else:
            self.slice_x = None
            self.slice_y = None
        self.varu = griddata((self.lon0.ravel(), self.lat0.ravel()),
                             varu.ravel(),
                             (self.lon2d, self.lat2d), method='linear')
        self.varv = griddata((self.lon0.ravel(), self.lat0.ravel()),
                             varv.ravel(),
                             (self.lon2d, self.lat2d), method='linear')
        # # TODO: Implement more efficient gridding method
        self.varu[numpy.isnan(self.varu)] = 0
        self.varv[numpy.isnan(self.varv)] = 0
        return None

    def read_var(self, index=None, time=0, missing_value=None,
                 size_filter=None):
        '''Read data variable'''
        var = read_var(self.file, self.nvar, index=index, time=0, depth=0,
                       missing_value=missing_value, subsample=self.ss)
        if size_filter is not None:
            var = filters.gaussian_filter(var, sigma=size_filter)
        if self.box is not None:
            var = var[self.slice_y, self.slice_x]
        self.var = griddata((self.lon0.ravel(), self.lat0.ravel()), var,
                            (self.lon2d, self.lat2d), method='linear')
        return None

    def read_time(self, index=None, time=0, missing_value=None):
        '''Read data time'''
        self.time = read_time(self.file, self.ntime, itime=0)
        return None


class initclass():
    def __init__(self, ):
        pass


class tracer_netcdf():
    def __init__(self, filename=None, lon='lon', lat='lat', time='time',
                 var='sst', qual=None):
        self.file = filename
        self.nlon = lon
        self.nlat = lat
        self.ntime = time
        self.nvar = var
        if qual:
            self.nerror = qual
        else:
            self.nerror = None

    def read_var(self, index=None, time=0, missing_value=None):
        '''Read tracer variable'''
        self.var = read_var(self.file, self.nvar, index=index, time=time,
                            depth=0, missing_value=missing_value)
        return None

    def read_coord(self):
        '''Read tracer coordinates'''
        self.lon, self.lat = read_coordinates(self.file, self.nlon, self.nlat)
        self.lon = (self.lon + 360) % 360
        return None

    def read_mask(self, index=None, time=0, missing_value=None):
        '''Read tracer mask'''
        self.mask = read_var(self.file, self.nerror, index=index, time=time,
                             depth=0, missing_value=missing_value)
        return None

    def read_time(self):
        '''Read tracer trime'''
        self.time = read_time(self.file, self.ntime)
        return None


def read_trajectory(infile:str, list_var: list) -> dict:
    dict_var = {}
    fid = netCDF4.Dataset(infile, 'r')
    for key in list_var:
        if key in fid.variables.keys():
            dict_var[key] = fid.variables[key][:]
        else:
            logger.warn(f'variable {key} not found in file {infile}')
    fid.close()
    return dict_var


def interp_vel(VEL:dict, coord:dict) -> dict:
    interp2d = scipy.interpolate.interp2d
    _inte = {}
    _inte['time'] = coord['time']
    for key in VEL.keys():
        _inte[key] = None
        VEL[key]['array'][numpy.isnan(VEL[key]['array'])] = 0
    if len(numpy.shape(VEL['u']['array'])) > 2:
        for key in VEL.keys():
            _inte[key] = []
        for t in range(len(coord['time'])):
            for key in VEL.keys():
                nlon = VEL[key]['lon']
                nlat = VEL[key]['lat']
                _indlon = numpy.where(nlon == 170.125)
                _indlat = numpy.where(nlat == 25.875)

                _tmp = interp2d(nlon, nlat,
                                VEL[key]['array'][t, :, :])
                _inte[key].append(_tmp)
    else:
        for key in VEL.keys():
            nlon = VEL[key]['lon']
            nlat = VEL[key]['lat']
            _tmp = interp2d(nlon, nlat,
                            VEL[key]['array'][t, :, :])
            _inte[key] = list([_tmp, ])
    return _inte
