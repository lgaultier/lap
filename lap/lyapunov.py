'''
#-----------------------------------------------------------------------
#                       Additional Documentation
# Author: Lucile Gaultier
#
# Modification History:
# - Jan 2015:  Original by Lucile Gaultier
# - Feb 2015: Version 1.0
# - Dec 2015: Version 2.0
# - Dec 2017: Version 3.0
# - Jan 2021: Version 4.0
# Notes:
# - Written for Python 3.4, tested with Python 3.6
#
# Copyright (c)
#-----------------------------------------------------------------------
'''

import datetime
from typing import Optional, Tuple
import numpy
import sys
import lap.mod_advection as mod_advection
import lap.utils.tools as tools
import lap.mod_io as mod_io
import lap.utils.general_utils as utils
import logging
logger = logging.getLogger(__name__)


def init_particles(plon: float, plat: float, dx: float
                   ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    lonp = plon + dx
    lonm = plon - dx
    latp = plat + dx
    latm = plat - dx
    npa_lon = [lonm, lonp, plon, plon]
    npa_lat = [plat, plat, latm, latp]
    return npa_lon, npa_lat


def advection(p, npa_lon: numpy.ndarray, npa_lat: float, dic_vel: dict,
              store: Optional[bool] = True
              ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    num_pa = numpy.shape(npa_lon)
    mask = 0
    r = numpy.zeros((2, 1))
    tadvection = (p.first_date - p.last_date).total_seconds() / 86400
    for i in range(num_pa[0]):
        lonpa = + npa_lon[i]
        latpa = + npa_lat[i]
        if store is True:
            npa_lon[i] = []
            npa_lat[i] = []
        dt = 0
        while dt < abs(tadvection):
			# Index in velocity array, set to 0 if stationary
			curdate = p.first_date + datetime.timedelta(seconds=dt*86400)
			_diff = (numpy.datetime64(curdate) - dic_vel['time'])
			ind_t = numpy.argmin(abs(_diff), out=None)
			if dic_vel['time'][ind_t] > numpy.datetime64(curdate) :
				ind_t = max(0, ind_t - 1)
			if ind_t > len(dic_vel['time']) - 1:
				break
			dt_vel = ((numpy.datetime64(curdate) - dic_vel['time'][ind_t])
				      / (dic_vel['time'][ind_t + 1] - dic_vel['time'][ind_t]))
			if p.stationary:
				ind_t = 0
				dt_vel = 0

            # # TODO: retrieve index t in velocity
            _adv = mod_advection.advection_pa_timestep_np
            advect = _adv(p, lonpa, latpa, dt_vel, ind_t, mask, r,
                          dic_vel['u'], dic_vel['v'])
            lonpa, latpa, mask = advect
            dt += p.adv_time_step
            if store is True:
                npa_lon[i].append(lonpa)
                npa_lat[i].append(latpa)
        if store is False:
            npa_lon[i] = + lonpa
            npa_lat[i] = + latpa
    return npa_lon, npa_lat, mask


def deform_pa(npa_lon: numpy.ndarray, npa_lat: numpy.ndarray, dx: float
              ) -> Tuple[float, numpy.ndarray, numpy.ndarray]:
    deform = numpy.zeros((2, 2))
    deform[0, 0] = npa_lon[1] - npa_lon[0]
    deform[0, 1] = npa_lat[1] - npa_lat[0]
    deform[1, 0] = npa_lon[3] - npa_lon[2]
    deform[1, 1] = npa_lat[3] - npa_lat[2]
    deform = deform * 2 / dx

    # Compute jacobian M * M^T
    jacob = numpy.matmul(deform, numpy.transpose(deform))
    # jacob = numpy.transpose(deform) * deform

    # Compute eigenvalues l1 and l2 and eigenvectors v1 and v2
    lmean = (jacob[0, 0] + jacob[1, 1]) / 2.
    lprod = jacob[0, 0] * jacob[1, 1] - jacob[0, 1] * jacob[1, 0]
    dl = numpy.sqrt(lmean * lmean - lprod)
    l1 = lmean + dl
    l2 = lmean - dl
    # l1, l2 = numpy.linalg.eigvals(jacob)
    lmax = numpy.sqrt(max(abs(l1), abs(l2)))
    v1, v2 = numpy.linalg.eig(jacob)

    '''
    ! Compute eigenvalues l1 and l2 of matd
    lmean =  ( matd(1,1) + matd(2,2) ) * half
    lprod = matd(1,1) * matd(2,2) - matd(1,2) * matd(2,1)
    dl = SQRT ( lmean * lmean  - lprod )
    l1 = lmean + dl ; l2 = lmean - dl

    ! Compute Lyapunov exponent
    lmax = MAX(ABS(l1),ABS(l2))
    ftle = LOG(SQRT(lmax))/ABS(tend-tstart)
    '''
    return lmax, v1, v2


def ftle_pa(lon: numpy.ndarray, lat: numpy.ndarray, df:float, d0: float,
            dt:float, tf: float) -> Tuple[float, numpy.ndarray, numpy.ndarray]:
    # Compute eigenvalues and vector
    lmax, v1, v2 = deform_pa(lon, lat, d0)
    # Compute FTLE
    ftle = numpy.log(lmax) / abs(tf)
    return ftle, v1, v2


def fsle_pa(lon: numpy.ndarray, lat: numpy.ndarray, df: float, d0: float,
            dt: float, tf: float) -> Tuple[float, float, float]:
    haversine = False
    if haversine is True:
        dx1 = tools.haversine(lon[1][:], lon[0][:], lat[1][:], lat[0][:])
        dx2 = tools.haversine(lon[3][:], lon[3][:], lat[3][:], lat[2][:])
    else:
        dlon1 = numpy.array(lon[1][:]) - numpy.array(lon[0][:])
        dlat1 = numpy.array(lat[1][:]) - numpy.array(lat[0][:])
        dlon2 = numpy.array(lon[3][:]) - numpy.array(lon[2][:])
        dlat2 = numpy.array(lat[3][:]) - numpy.array(lat[2][:])
        dx1 = numpy.sqrt(dlon1**2 + dlat1**2)
        dx2 = numpy.sqrt(dlon2**2 + dlat2**2)
    _max = numpy.maximum(dx1, dx2)
    tau = 0
    istep = 0
    while istep < len(_max):
        if _max[istep] > df:
            tau = istep * dt
            break
        istep += 1
    if tau != 0:
        lonpa = [lon[0][istep], lon[1][istep], lon[2][istep], lon[3][istep]]
        latpa = [lat[0][istep], lat[1][istep], lat[2][istep], lat[3][istep]]
        deform = numpy.zeros((2, 2))
        deform[0, 0] = lon[1][istep] - lon[0][istep]
        deform[0, 1] = lon[3][istep] - lon[2][istep]
        deform[1, 0] = lat[1][istep] - lat[0][istep]
        deform[1, 1] = lat[3][istep] - lat[2][istep]
        lmax, _, _ = deform_pa(lonpa, latpa, d0)
    else:
        tau = tf
        lmax = df / d0
    fsle = numpy.log(lmax) / tau
    return fsle, 0, 0


def run_lyapunov(p):
    tools.make_default(p)
    logger.info('Loading Velocity')
    VEL, coord = rr.read_velocity(p)
    lyapunov(p, VEL, coord)


def lyapunov(p, VEL, coord):
    # - Initialize variables from parameter file
    # ------------------------------------------
    tools.make_default(p)
    comm = None
    p.parallelisation, size, rank, comm = utils.init_mpi(p.parallelisation)
    if rank == 0:
        data = {}
    if p.diagnostic == 'FSLE':
        isFSLE = True
        method = fsle_pa
        store = True
        name_var = 'fsle'
    elif p.diagnostic == 'FTLE':
        isFSLE = False
        method = ftle_pa
        store = False
        name_var = 'ftle'
    else:
        logger.error(f'Wrong diagnostic {p.diagnostics}, choice for lyapunov'
                     'are FSLE or FTLE')
        sys.exit(1)
    # Make Grid
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        grid = mod_io.make_grid(p)
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)
    else:
        grid = None
    if p.parallelisation is True:
        grid = comm.bcast(grid, root=0)

    # Read velocity
    dic_vel = None
    if rank == 0:
        dic_vel = utils.interp_vel(VEL, coord)
    if p.parallelisation is True:
        dic_vel = comm.bcast(dic_vel, root=0)

    # For each point in Grid
    grid_size = numpy.shape(grid.lon1d)[0]
    dim = (grid_size, )
    grid.fsle = numpy.zeros(dim)
    init = utils.init_empty_variables(p, grid, None, size, rank)
    _, _, grid_size, reducepart, i0, i1 = init
    all_lambda = []
    all_mask = []
    data_r = {}

    for pa in reducepart:
        lonpa = grid.lon1d[pa]
        latpa = grid.lat1d[pa]
        # advect four points around position
        _npalon, _npalat = init_particles(lonpa, latpa, p.delta0)
        npalon, npalat, mask = advection(p, _npalon, _npalat, dic_vel['u'],
                                         dic_vel['v'],
                                         store=store)
        if pa % 100 == 0:
            perc = float(pa - i0) / float(len(reducepart))
            tools.update_progress(perc, str(pa), str(rank))

        # Compute FTLE
        vlambda, _, _ = method(npalon, npalat, p.deltaf, p.delta0,
                               p.adv_time_step, abs(p.tadvection))
        all_lambda.append(vlambda)
        all_mask.append(mask)
    data_r[name_var] = all_lambda
    data_r['mask'] = all_mask
    if p.parallelisation is True:
        drifter = utils.gather_data_mpi(p, data_r, None, None, dim, dim,
                                        comm, rank, size, grid_size)

    else:
        drifter = utils.gather_data(p, data_r, None, None)
    if rank == 0:
        drifter[name_var] = numpy.array(drifter[name_var])
        drifter['mask'] = numpy.array(drifter['mask'])
        shape_grid = numpy.shape(grid.lon)
        data[name_var] = drifter[name_var].reshape(shape_grid)
        data['mask'] = drifter['mask'].reshape(shape_grid)
        # Write FTLE / FSLE
        data['lon'] = grid.lon
        data['lon2'] = grid.lon1d.reshape(shape_grid)
        data['lat2'] = grid.lat1d.reshape(shape_grid)
        data['lat'] = grid.lat
        _date = (p.first_day - p.reference).total_seconds() / 86400
        data['time'] = numpy.array([_date, ])
        if isFSLE is True:
            description = ("Finite-Size Lyapunov Exponent computed using"
                           "lap_toolbox")
            mod_io.write_diagnostic_2d(p, data, description=description,
                                       FSLE=data[name_var], time=data['time'])
        else:
            description = ("Finite-Time Lyapunov Exponent computed using"
                           "lap_toolbox")
            mod_io.write_diagnostic_2d(p, data, description=description,
                                       FTLE=data[name_var], time=data['time'],
                                       lon2=data['lon2'], lat2=data['lat2'])
        logger.info(f'Stop time {datetime.datetime.now()}')
