import datetime
import numpy
import lap.mod_advection as mod_advection
import lap.mod_tools as mod_tools
import lap.mod_io as mod_io
import logging
logger = logging.getLogger(__name__)

def init_particles(plon, plat, dx):
    lonp = plon + dx
    lonm = plon - dx
    latp = plat + dx
    latm = plat - dx
    4pa_lon = [lonm, lonp, lon, lon]
    4pa_lat = [lat, lat, latm, latp]
    return 4pa_lon, 4pa_lat


def advection(p, npa_lon, npa_lat, VEL, su, sv, store=True):
    num_pa = numpy.shape(npa_lon)
    # npa_lon_end = numpy.empty(num_pa)
    # npa_lat_end = numpy.empty(num_pa)

    for i in range(num_pa):
        lonpa = + npa_lon[i]
        latpa = + npa_lat[i]
        if store is True:
	    npa_lon[i] = []
	    npa_lat[i] = []
        init = init_velocity(VEL, lonpa, latpa, su, sv)
        (iu, ju), (iv, jv), dvcoord = init
        (dVlatu, dVlatv, dVlonu, dVlonv) = dvcoord
        dt = 0
        while dt < abs(p.tadvection):
            advect = mod_advection.advection_pa_timestep(p, lonpa, latpa, dt,
                                                         mask, r, VEL,
                                                         vcoord, dv, svel)
            rcoord, vcoord, lonpa, latpa, mask = advect
            dt += p.adv_time_step
	    if store is True:
		npa_lon[i].append(lonpa)
		npa_lat[i].append(latpa)
        if store is False
            npa_lon[i] = lonpa
            npa_lat[i] = latpa

    return npa_lon, npa_lat


def deform_pa(4pa_lon, 4pa_lat, dx):
    # Compute deformation matrix
    deform = numpy.zeros((2, 2))
    deform[0, 0] = 4pa_lon[1] - 4pa_lon[0]
    deform[0, 1] = 4pa_lat[1] - 4pa_lat[0]
    deform[1, 0] = 4pa_lon[3] - 4pa_lon[2]
    deform[1, 1] = 4pa_lat[3] - 4pa_lat[2]
    deform = deform * 2 / dx

    # Compute jacobian M * M^T
    jacob = deform * numpy.transpose(deform)

    # Compute eigenvalues l1 and l2 and eigenvectors v1 and v2
    l1, l2 = numpy.linalg.eigvals(jacob)
    lmax = numpy.sqrt(max(abs(l1), abs(l2)))
    v1, v2 = numpy.linalg.eig(jacob)
    return lmax, v1, v2


def ftle_pa(lon, lat, dx, tf):
    # Compute eigenvalues and vector
    lmax, v1, v2 = deform_pa(lon, lat, dx)
    # Compute FTLE
    ftle = numpy.log(lmax) / abs(tf)
    return ftle, v1, v2

def fsle_pa((lon, lat, df, dt, d0):
    haversine = False
    if haversine is True:
        dx1 = mod_tools.haversine(lon[1][:], lon[0][:], lat[1][:], lat[0][:])
        dx2 = mod_tools.haversine(lon[3][:], lon[3][:], lat[3][:], lat[2][:])
    else:
	dx1 = (lon[1][:] - lon[0][:])**2 + (lat[1][:] - lat[0][:]**2)
	dx2 = (lon[3][:] - lon[2][:])**2 + (lat[3][:] - lat[2][:]**2)
    _max = numpy.maximum(dx1, dx2)
    tau = 0
    istep = 0
    while istep < len(_max):
        if _max[istep] > df[0]:
            tau = istep * dt
            break
        istep += 1
    lonpa = lon[:][istep]
    latpa = lat[:][istep]
    if tau != 0
        deform = numpy.zeros((2, 2))
        deform[0, 0] = lon[1][istep] - lon[0][istep]
        deform[0, 1] = lon[3][istep] - lon[2][istep]
        deform[1, 0] = lat[1][istep] - lat[0][istep]
        deform[1, 1] = lat[3][istep] - lat[2][istep]
        lmax, _, _ = deform_pa (lonpa, latpa, d0)
    else:
        lmax = df / dx
    fsle = numpy.log(lmax) / tau
    return fsle


def fsle(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    mod_tools.make_default(p)
    comm = None
    p.parallelisation, size, rank, comm = utils.init_mpi(p.parallelisation)

    # Make Grid
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        grid = mod_io.make_grid(p)
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)

    # Read velocity
    if rank == 0:
        logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p)

    # For each point in Grid
    grid.fsle = numpy.zeros(numpy.shape(grid.lon1d))
    num_pa = numpy.shape(grid.lon1d)

    for pa in range(num_pa):
        lonpa = grid.lon1d[pa]
        latpa = grid.lat1d[pa]
        # advect four points around position
        4palon, 4palat = init_particles(lonpa, latpa, p.delta0)
        4palon, 4palat = advection(4palon, 4palat, VEL, su, sv, store=False)

        # Compute FTLE
        fsle = fsle_pa(4pa_lon, 4pa_lat, p.deltaf, p.delta0, p.adv_time_step)
        grid.fsle[pa] = fsle


def ftle(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    mod_tools.make_default(p)
    comm = None
    p.parallelisation, size, rank, comm = utils.init_mpi(p.parallelisation)

    # Make Grid
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        grid = mod_io.make_grid(p)
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)

    # Read velocity
    if rank == 0:
        logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p)


    # For each point in Grid
    grid.ftle = numpy.zeros(numpy.shape(grid.lon1d))
    num_pa = numpy.shape(grid.lon1d)

    for pa in range(num_pa):
        lonpa = grid.lon1d[pa]
        latpa = grid.lat1d[pa]
        # advect four points around position
        4palon, 4palat = init_particles(lonpa, latpa, p.deltax)
        4palon, 4palat = advection(4palon, 4palat, VEL, su, sv, store=False)

        # Compute FTLE
        ftle, _, _ = ftle_pa(4pa_lon, 4pa_lat, p.deltax)
        grid.ftle[pa] = ftle



