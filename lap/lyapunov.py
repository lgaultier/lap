import datetime
import numpy
import sys
import lap.mod_advection as mod_advection
import lap.mod_tools as mod_tools
import lap.mod_io as mod_io
import lap.utils.general_utils as utils
import lap.const as const
import logging
logger = logging.getLogger(__name__)


def init_particles(plon, plat, dx):
    lonp = plon + dx
    lonm = plon - dx
    latp = plat + dx
    latm = plat - dx
    npa_lon = [lonm, lonp, plon, plon]
    npa_lat = [plat, plat, latm, latp]
    return npa_lon, npa_lat


def advection(p, npa_lon, npa_lat, VEL, store=True):
    num_pa = numpy.shape(npa_lon)
    su = numpy.shape(VEL.u)
    sv = numpy.shape(VEL.v)
    svel = (su, sv)
    # npa_lon_end = numpy.empty(num_pa)
    # npa_lat_end = numpy.empty(num_pa)
    mask = 0
    r = numpy.zeros((2, 1))
    for i in range(num_pa[0]):
        lonpa = + npa_lon[i]
        latpa = + npa_lat[i]
        if store is True:
            npa_lon[i] = []
            npa_lat[i] = []
        init = mod_advection.init_velocity(VEL, lonpa, latpa, su, sv)
        [iu, ju], [iv, jv], dvcoord = init
        [dVlatu, dVlatv, dVlonu, dVlonv] = dvcoord
        vcoord = [iu, ju, iv, jv]
        dt = 0
        sizeadvection = p.tadvection / p.adv_time_step
        while dt < abs(p.tadvection):
            # # TODO: retrieve index t in velocity
            advect = mod_advection.advection_pa_timestep(p, lonpa, latpa,
                                                         int(dt), dt,
                                                         mask, r, VEL,
                                                         vcoord, dvcoord, svel,
                                                         sizeadvection)
            rcoord, vcoord, dvcoord, lonpa, latpa, mask = advect
            dt += p.adv_time_step
            if store is True:
                npa_lon[i].append(lonpa)
                npa_lat[i].append(latpa)
        if store is False:
            npa_lon[i] = lonpa
            npa_lat[i] = latpa

    return npa_lon, npa_lat, mask


def deform_pa(npa_lon, npa_lat, dx):
    # Compute deformation matrix
    deform = numpy.zeros((2, 2))
    deform[0, 0] = npa_lon[1] - npa_lon[0]
    deform[0, 1] = npa_lat[1] - npa_lat[0]
    deform[1, 0] = npa_lon[3] - npa_lon[2]
    deform[1, 1] = npa_lat[3] - npa_lat[2]
    deform = deform * 2 / dx

    # Compute jacobian M * M^T
    # jacob = deform * numpy.transpose(deform)
    jacob = numpy.transpose(deform) * deform

    # Compute eigenvalues l1 and l2 and eigenvectors v1 and v2
    l1, l2 = numpy.linalg.eigvals(jacob)
    lmax = numpy.sqrt(max(abs(l1), abs(l2)))
    v1, v2 = numpy.linalg.eig(jacob)
    return lmax, v1, v2


def ftle_pa(lon, lat, df, dt, d0, tf):
    # Compute eigenvalues and vector
    lmax, v1, v2 = deform_pa(lon, lat, d0)
    # Compute FTLE
    ftle = numpy.log(lmax) / abs(tf)
    return ftle, v1, v2


def fsle_pa(lon, lat, df, dt, d0, tf):
    haversine = False
    if haversine is True:
        dx1 = mod_tools.haversine(lon[1][:], lon[0][:], lat[1][:], lat[0][:])
        dx2 = mod_tools.haversine(lon[3][:], lon[3][:], lat[3][:], lat[2][:])
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


def lyapunov(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    mod_tools.make_default(p)
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
    if rank == 0:
        logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p)

    # For each point in Grid
    grid_size = numpy.shape(grid.lon1d)[0]
    dim = (grid_size, )
    grid.fsle = numpy.zeros(dim)
    num_pa = numpy.shape(grid.lon1d)
    init = utils.init_empty_variables(p, grid, None, size, rank)
    _, _, grid_size, reducepart, i0, i1 = init
    all_lambda = []
    all_mask = []
    data_r = {}
    for pa in reducepart:
        lonpa = grid.lon1d[pa]
        latpa = grid.lat1d[pa]
        # advect four points around position
        npalon, npalat = init_particles(lonpa, latpa, p.delta0)
        npalon, npalat, mask = advection(p, npalon, npalat, VEL, store=store)
        if pa % 100 == 0:
            perc = float(pa - i0) / float(len(reducepart))
            mod_tools.update_progress(perc, str(pa), str(rank))

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
        data['lat'] = grid.lat
        data['time'] = numpy.array([p.first_day, ])
        if isFSLE is True:
            description = ("Finite-Size Lyapunov Exponent computed using"
                           "lap_toolbox")
            mod_io.write_diagnostic_2d(p, data, description=description,
                                       FSLE=data[name_var], time=data['time'])
        else:
            description = ("Finite-Time Lyapunov Exponent computed using"
                           "lap_toolbox")
            mod_io.write_diagnostic_2d(p, data, description=description,
                                       FTLE=data[name_var], time=data['time'])
        logger.info(f'Stop time {datetime.datetime.now()}')
