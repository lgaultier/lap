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


def geoloc_from_gcps(gcplon, gcplat, gcplin, gcppix, lin, pix):
    """"""
    import pyproj
    geod = pyproj.Geod(ellps='WGS84')
    fwd, bwd, dis = geod.inv(gcplon[:, :-1], gcplat[:, :-1],
                             gcplon[:, 1:], gcplat[:, 1:])

    # Find line and column for the top-left corner of the 4x4 GCPs cell which
    # contains the requested locations
    nlin, npix = gcplat.shape
    _gcplin = gcplin[:, 0]
    _gcppix = gcppix[0, :]
    top_line = numpy.searchsorted(_gcplin, lin, side='right') - 1
    left_column = numpy.searchsorted(_gcppix, pix, side='right') - 1

    # Make sure this line and column remain within the matrix and that there
    # are adjacent line and column to define the bottom-right corner of the 4x4
    # GCPs cell
    top_line = numpy.clip(top_line, 0, nlin - 2)
    bottom_line = top_line + 1
    left_column = numpy.clip(left_column, 0, npix - 2)
    right_column = left_column + 1

    # Compute coordinates of the requested locations in the 4x4 GCPs cell
    line_extent = _gcplin[bottom_line] - _gcplin[top_line]
    column_extent = _gcppix[right_column] - _gcppix[left_column]
    line_rel_pos = (lin - _gcplin[top_line]) / line_extent
    column_rel_pos = (pix - _gcppix[left_column]) / column_extent

    # Compute geographical coordinates of the requested locations projected on
    # the top and bottom lines
    lon1, lat1, _ = geod.fwd(gcplon[top_line, left_column],
                             gcplat[top_line, left_column],
                             fwd[top_line, left_column],
                             dis[top_line, left_column] * column_rel_pos)
    lon2, lat2, _ = geod.fwd(gcplon[bottom_line, left_column],
                             gcplat[bottom_line, left_column],
                             fwd[bottom_line, left_column],
                             dis[bottom_line, left_column] * column_rel_pos)

    # Compute the geographical coordinates of the requested locations projected
    # on a virtual column joining the projected points on the top and bottom
    # lines
    fwd12, bwd12, dis12 = geod.inv(lon1, lat1, lon2, lat2)
    lon, lat, _ = geod.fwd(lon1, lat1, fwd12, dis12 * line_rel_pos)

    return lon, lat


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


def read_idf_gcps(filename, nlon, nlat):
    # - Open Netcdf file
    fid = netCDF4.Dataset(filename, 'r')

    # - Read 1d or 2d coordinates
    # Extract data from the handler
    lon_gcp = numpy.array(handler['lon_gcp'][:])
    lat_gcp = numpy.array(handler['lat_gcp'][:])
    # Enforce longitude continuity (to be improved)
    if len(numpy.shape(lon_gcp)) == 2:
        regular = False
        i_gcp = numpy.array(handler['index_row_gcp'][:])
        j_gcp = numpy.array(handler['index_cell_gcp'][:])
    else:
        regular = True
        i_gcp = numpy.array(handler['index_lat_gcp'][:])
        j_gcp = numpy.array(handler['index_lon_gcp'][:])
    if regular is True:
        ind_lon = numpy.where((lon_gcp > box[0]-1) & (lon_gcp < box[1]+1))
        ind_lat = numpy.where((lat_gcp > box[2]-1) & (lat_gcp < box[3]+1))
        lon_gcp = lon_gcp[ind_lon]
        lat_gcp = lat_gcp[ind_lat]
        j_gcp = j_gcp[ind_lon]
        i_gcp = i_gcp[ind_lat]
        i0 = numpy.min(i_gcp)
        i1 = numpy.max(i_gcp) + 1
        j0 = numpy.min(j_gcp)
        j1 = numpy.max(j_gcp) + 1
        j_gcp = j_gcp - j_gcp[0]
        i_gcp = i_gcp - i_gcp[0]
    else:
        ind_lon_lat = numpy.where((lon_gcp > box[0]) & (lon_gcp < box[1])
                               & (lat_gcp > box[2]) & (lat_gcp < box[3]))
        i_gcp_0 = numpy.min(ind_lon_lat[0])
        i_gcp_1 = numpy.max(ind_lon_lat[0])
        j_gcp_0 = numpy.min(ind_lon_lat[1])
        j_gcp_1 = numpy.max(ind_lon_lat[1])
        lon_gcp = lon_gcp[i_gcp_0:i_gcp_1 + 1, j_gcp_0: j_gcp_1 + 1]
        lat_gcp = lat_gcp[i_gcp_0:i_gcp_1 + 1, j_gcp_0: j_gcp_1 + 1]
        i0 = i_gcp[i_gcp_0]
        i1 = i_gcp[i_gcp_1] + 1
        i0 = j_gcp[j_gcp_0]
        i1 = j_gcp[j_gcp_1] + 1
    if (lon_gcp[-1] - lon_gcp[0]) > 180.0:
        print('Difference between first and last longitude exceeds '
                    '180 degrees, assuming IDL crossing and remapping '
                    'longitudes in [0, 360]')
        box[0] = numpy.mod(box[0] + 360, 360)
        box[1] = numpy.mod(box[1] + 360, 360)
        lon_gcp = numpy.mod((lon_gcp + 360.0), 360.0)
    # Restore shape of the GCPs
    #gcps_shape = (8, 8)  # hardcoded in SEAScope
    #i_shaped = numpy.reshape(i_gcp, gcps_shape)
    #j_shaped = numpy.reshape(j_gcp, gcps_shape)
    #lon_shaped = numpy.reshape(lon_gcp, gcps_shape)
    #lat_shaped = numpy.reshape(lat_gcp, gcps_shape)
    if len(numpy.shape(i_gcp)) == 1:
        j_shaped, i_shaped = numpy.meshgrid(j_gcp, i_gcp)
    else:
        i_shaped = i_gcp
        j_shaped = j_gcp
    if len(numpy.shape(lon_gcp)) == 1:
        lon_shaped, lat_shaped = numpy.meshgrid(lon_gcp, lat_gcp[:])
    else:
        lon_shaped = lon_gcp[:, :]
        lat_shaped = lat_gcp[:, :]
    shape = numpy.shape(sst_reg)
    dst_lin = numpy.arange(0, shape[0])
    dst_pix = numpy.arange(0, shape[1])
    _dst_lin = numpy.tile(dst_lin[:, numpy.newaxis], (1, shape[1]))
    _dst_pix = numpy.tile(dst_pix[numpy.newaxis, :], (shape[0], 1))

    lon2d, lat2d = geoloc_from_gcps(lon_shaped, lat_shaped, i_shaped,
                                      j_shaped, _dst_lin, _dst_pix)



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
    if index is None:
        if ndim == 1:
            T = numpy.ma.array(fid.variables[var][:]).squeeze()
            T = T[::subsample]
        elif ndim == 2:
            T = numpy.ma.array(fid.variables[var][:, :]).squeeze()
            T = T[::subsample, ::subsample]
        elif ndim == 3:
            if time is None:
                T = numpy.ma.array(fid.variables[var][:, :, :]).squeeze()
            else:
                T = numpy.ma.array(fid.variables[var][time, :, :]).squeeze()
            T = T[::subsample, ::subsample]
        elif ndim == 4:
            if time is None:
                if depth is None:
                    T = numpy.ma.array(fid.variables[var][:, :, :, :]).squeeze()
                else:
                    T = numpy.ma.array(fid.variables[var][:, depth, :,
                                    :]).squeeze()
            elif depth is None:
                T = numpy.ma.array(fid.variables[var][time, :, :, :]).squeeze()
            else:
                T = numpy.ma.array(fid.variables[var][time, depth, :,
                                :]).squeeze()
            T = T[::subsample, ::subsample]
    else:
        if ndim == 1:
            T = numpy.ma.array(fid.variables[var][index]).squeeze()
            T = T[::subsample]
        elif ndim == 2:
            T = numpy.ma.array(fid.variables[var][index]).squeeze()
            T = T[::subsample, ::subsample]
        elif ndim == 3:
            if time is None:
                U = numpy.ma.array(fid.variables[var][:, :, :]).squeeze()
                T = U[:, index]
            else:
                U = numpy.ma.array(fid.variables[var][time, :, :]).squeeze()
                T = U[index]
            T = T[::subsample, ::subsample]
        elif ndim == 4:
            if time is None:
                if depth is None:
                    U = numpy.ma.array(fid.variables[var][:, :, :, :]).squeeze()
                    T = U[:, :, index]
                else:
                    U = numpy.ma.array(fid.variables[var][:, depth, :,
                                    :]).squeeze()
                    T = U[:, index]
            elif depth is None:
                U = numpy.ma.array(fid.variables[var][time, :, :, :]).squeeze()
                T = U[:, index]
            else:
                U = numpy.ma.array(fid.variables[var][time, depth, :,
                                :]).squeeze()
                T = U[index]
            T = T[::subsample, ::subsample]

    fid.close()

    # - Mask value that are NaN
    if missing_value is not None:
        T[numpy.where(T == missing_value)] = numpy.nan
    T._sharedmask = False
    return T


def read_time(filename, ntime, itime=None):
    # - Open Netcdf file
    fid = netCDF4.Dataset(filename, 'r')

    # - Read time variable and units
    time = fid.variables[ntime][:]
    if itime is not None:
        time = time[itime]
    time_units = fid[ntime].units

    # - Try to read calendar
    try:
        time_cal = fid[ntime].calendar
    except AttributeError:
        logger.warn('No calendar in time attributes, standard calendar used')
        time_cal = u"gregorian"
    time = netCDF4.num2date(time, units=time_units, calendar=time_cal)
    fid.close()
    return time


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

    def read_vel(self, index=None, time=0, missing_value=None,
                 size_filter=None, slice_xy=None):
        '''Read data velocity'''
        if slice_xy is None:
            self.read_coord()
        else:
            self.slice_x = slice_xy[0]
            self.slice_y = slice_xy[1]
        varu = read_var(self.file, self.nvaru, index=None, time=0,
                        depth=0, missing_value=missing_value)
        varv = read_var(self.file, self.nvarv, index=None, time=0,
                        depth=0, missing_value=missing_value)
        if size_filter is not None:
            varu = filters.gaussian_filter(varu, sigma=size_filter)
            varv = filters.gaussian_filter(varv, sigma=size_filter)
        if self.box is not None:
            if slice_xy is None:
                self.read_coord()
            else:
                self.slice_x = slice_xy[0]
                self.slice_y = slice_xy[1]
            varu = varu[self.slice_y, self.slice_x]
            varv = varv[self.slice_y, self.slice_x]
        else:
            self.slice_x = None
            self.slice_y = None
        self.varu = varu
        self.varv = varv
        self.varu._sharedmask = False
        self.varv._sharedmask = False
        return None

    def read_var(self, index=None, time=0, missing_value=None,
                 slice_xy=None, size_filter=None):
        '''Read data variable'''
        if slice_xy is None:
            self.slice_x, self.slice_y = self.read_coord()
        else:
            self.slice_x = slice_xy[0]
            self.slice_y = slice_xy[1]
        var = read_var(self.file, self.nvar, index=None, time=0, depth=0,
                            missing_value=missing_value)
        if self.box is not None:
            if slice_xy is None:
                self.slice_x, self.slice_y = self.read_coord()
            else:
                self.slice_x = slice_xy[0]
                self.slice_y = slice_xy[1]
            var = var[self.slice_y, self.slice_x]
        self.var = var
        return None

    def read_time(self, index=None, time=0):
        '''Read data time'''
        self.time = read_time(self.file, self.ntime)
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


def read_trajectory(infile, list_var):
    dict_var = {}
    fid = netCDF4.Dataset(infile, 'r')
    for key in list_var:
        if key in fid.variables.keys():
            dict_var[key] = fid.variables[key][:]
        else:
            logger.warn(f'variable {key} not found in file {infile}')
    fid.close()
    return dict_var


def interp_vel(VEL, save_s=False, save_ow=False, save_rv=False,) -> dict:
    if len(numpy.shape(VEL.Vlonu)) == 1:
        regular = True
        interp2d = scipy.interpolate.interp2d
        _lonu = + VEL.Vlonu
        _lonv = + VEL.Vlonv
        _latu = + VEL.Vlatu
        _latv = + VEL.Vlatv
    else:
        interp2d = scipy.interpolate.NearestNDInterpolator
        _lonu = + VEL.Vlonu.ravel()
        _lonv = + VEL.Vlonv.ravel()
        _latu = + VEL.Vlatu.ravel()
        _latv = + VEL.Vlatv.ravel()
        points = (_lonu, _latu) #list(zip((_lonu, _latu)))
        print(numpy.shape(_lonu), numpy.shape(_latu))
        regular = False
    _inte_u = None
    _inte_v = None
    _inte_h = None
    _inte_sn = None
    _inte_ss = None
    _inte_rv = None
    if len(numpy.shape(VEL.u)) > 2:
        _inte_u = []
        _inte_v = []
        _inte_h = []
        _inte_sn = []
        _inte_ss = []
        _inte_rv = []
        for t in range(numpy.shape(VEL.u)[0]):
            if regular:
                _u = + VEL.u[t, :, :]
                _v = + VEL.v[t, :, :]
                _h = + VEL.h[t, :, :]
                _inte_u.append(interp2d(_lonu, _latu, _u))
                _inte_v.append(interp2d(_lonv, _latv, _v))
                _inte_h.append(interp2d(_lonu, _latv, _h))
            else:
                _u = + VEL.u[t, :, :].ravel()
                _v = + VEL.v[t, :, :].ravel()
                _h = + VEL.h[t, :, :].ravel()
                _inte_u.append(interp2d(points, _u))
                _inte_v.append(interp2d(points, _v))
                _inte_h.append(interp2d(points, _h))
            if save_s is True or save_ow is True:
                if regular:
                    _Sn = + VEL.Sn[t, :, :]
                    _Ss = + VEL.Ss[t, :, :]
                    _inte_sn.append(interp2d(_lonu, _latv, _Sn))
                    _inte_ss.append(interp2d(_lonu, _latv, _Ss))
                else:
                    _Sn = + VEL.Sn[t, :, :].ravel()
                    _Ss = + VEL.Ss[t, :, :].ravel()

                    _inte_sn.append(interp2d(points, _Sn))
                    _inte_ss.append(interp2d(points, _Ss))
            if save_rv is True or save_ow is True:
                if regular:
                    _RV = + VEL.RV[t, :, :]
                    _inte_rv.append(interp2d(_lonu, _latv, _RV))
                else:
                    _RV = + VEL.RV[t, :, :].ravel()
                    _inte_rv.append(interp2d(points, _RV))
    else:
        if regular:
            _u =  VEL.u[:, :]
            _v =  VEL.v[:, :]
            _h = VEL.h[:, :]
            _inte_u = list([interp2d(_lonu, _latu, _u), ])
            _inte_v = list([interp2d(_lonv, _latv, _v), ])
            _inte_h = list([interp2d(_lonu, _latv, _h), ])
        else:
            _u =  VEL.u[:, :].ravel()
            print(numpy.shape(_u))
            _v =  VEL.v[:, :].ravel()
            _h = VEL.h[:, :].ravel()
            _inte_u = list([interp2d(points, _u), ])
            _inte_v = list([interp2d(points, _v), ])
            _inte_h = list([interp2d(points, _h), ])
        if save_s is True or save_ow is True:
            if regular:
                _Sn = VEL.Sn[:, :]
                _Ss = VEL.Ss[:, :]
                _inte_sn = list([interp2d(_lonu, _latv, _Sn), ])
                _inte_ss = list([interp2d(_lonu, _latv, _Ss), ])
            else:
                _Sn = VEL.Sn[:, :].ravel()
                _Ss = VEL.Ss[:, :].ravel()
                _inte_sn = list([interp2d(points, _Sn), ])
                _inte_ss = list([interp2d(points, _Ss), ])
        if save_rv is True or save_ow is True:
            if regular:
                _RV = + VEL.RV[:, :]
                _inte_rv = list([interp2d(_lonu, _latv, _RV), ])
            else:
                _RV = + VEL.RV[:, :].ravel()
                _inte_rv = list([interp2d(points, _RV), ])

    dic_interp = {'u': _inte_u, 'v': _inte_v, 'h': _inte_h,
                  'sn': _inte_sn, 'ss': _inte_ss, 'rv': _inte_rv}
    return dic_interp
