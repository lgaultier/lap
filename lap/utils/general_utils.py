import numpy
import lap.mod_advection as mod_advection
import logging
logger = logging.getLogger(__name__)


# - Initialization
def init_mpi(isMPI):
    if isMPI is True:
        try:
            from mpi4py import MPI
        except ImportError:
            logger.warn("Module MPI not installed, no parallelisation")
            isMPI = False
    if isMPI is True:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # name = MPI.Get_processor_name()
    else:
        comm = None
        size = 1
        rank = 0
    return isMPI, size, rank, comm


def init_empty_variables(p, grid, listTr, size, rank):
    # - Initialize new tracer variables
    if rank == 0:
        logger.info('Advecting variables')
    Gridsize = numpy.shape(grid.lon1d)[0]
    # Tsize=numpy.shape(Tr.lon)
    npixel = int((Gridsize - 1)/size) + 1
    i0 = npixel * rank
    i1 = min(npixel*(rank + 1), Gridsize)
    # List of particles
    # particules = numpy.arange(0, Gridsize)
    # Sublist of particles when parallelised
    reducepart = numpy.arange(i0, i1)
    # Number of Timesteps to store
    Timesize_lr = int(abs(p.tadvection + 1) / p.output_step)
    # initialization + advection for t in [0, tadvection] with adv_time_step
    # time step
    Timesize_hr = int(abs(p.tadvection + 1) / p.output_step / p.adv_time_step
                      + 1)
    dim_hr = (Timesize_hr, Gridsize)
    dim_lr = (Timesize_lr, Gridsize)
    # Initialise number of tracer in list
    nlist = 0
    if rank == 0:
        # Low resolution variables
        grid.lon_lr = numpy.empty(dim_lr)
        grid.lat_lr = numpy.empty(dim_lr)
        grid.mask_lr = numpy.empty(dim_lr)
        grid.iit_lr = numpy.empty((2, dim_lr[0], dim_lr[1]))
        grid.ijt_lr = numpy.empty((2, dim_lr[0], dim_lr[1]))
        # High resolution variables
        grid.lon_hr = numpy.empty(dim_hr)
        grid.lat_hr = numpy.empty(dim_hr)
        grid.mask_hr = numpy.empty(dim_hr)
        grid.vel_v_hr = numpy.empty(dim_hr)
        grid.vel_u_hr = numpy.empty(dim_hr)
        grid.hei_hr = numpy.empty(dim_hr)
        if p.save_S is True:
            grid.S_hr = numpy.empty(dim_hr)
        if p.save_RV is True:
            grid.RV_hr = numpy.empty(dim_hr)
        if p.save_OW is True:
            grid.OW_hr = numpy.empty(dim_hr)
        if p.list_tracer is not None:
            nlist = len(listTr)
        for i in range(nlist):
            Tr = listTr[i]
            Tr.newvar = numpy.empty(dim_lr)
            Tr.newi = numpy.empty(dim_lr)
            Tr.newj = numpy.empty(dim_lr)
    return dim_hr, dim_lr, Gridsize, reducepart, i0, i1


def gather_data_mpi(p, list_var_adv, listGr, listTr, dim_lr, dim_hr,
                    comm, rank, size, grid_size):

    # Define local empty variables with the correct size
    if p.list_tracer is not None:
        nlist = len(listTr)
        for i in range(nlist):
            Tr = listTr[i]
            Tr.newvarlocal = numpy.empty(dim_lr)
            Tr.newilocal = numpy.empty(dim_lr)
            Tr.newjlocal = numpy.empty(dim_lr)

    # - Gather data in processor 0 and save them
    if listTr is not None:
        mod_advection.reordering1dmpi(p, listTr, listGr)
    comm.barrier()
    local = {}
    for key, value in list_var_adv.items():
        local[key] = comm.gather(value, root=0)
    if p.list_tracer is not None:
        nlist = len(listTr)
        for i in range(nlist):
            Tr = listTr[i]
            Tr.newvarlocal = comm.gather(Tr.newvarloc,  root=0)
            Tr.newilocal = comm.gather(Tr.newiloc,  root=0)
            Tr.newjlocal = comm.gather(Tr.newjloc, root=0)
    if rank == 0:
        data = {}
        if 'time_hr' in list_var_adv.keys():
            data['time_hr'] = list_var_adv['time_hr']
            del list_var_adv['time_hr']
        tstep = p.tadvection / p.output_step / abs(p.tadvection)
        tstop = p.first_day + p.tadvection + tstep
        data['time'] = numpy.arange(p.first_day, tstop, tstep)
        for irank in range(0, size):

            npixel = int((grid_size - 1)/size) + 1
            i0 = npixel * irank
            i1 = min(npixel*(irank + 1), grid_size)
            for key, value in list_var_adv.items():
                if irank == 0:
                    if 'lr' in key:
                        dim = dim_lr
                    else:
                        dim = dim_hr
                    data[key] = numpy.empty(dim)
                ndim = len(dim)
                if ndim == 1:
                    data[key][i0:i1] = local[key][irank][:]
                elif ndim == 2:
                    try:
                        data[key][:, i0:i1] = local[key][irank][:, :]
                    except:
                        import pdb ; pdb.set_trace()
                else:
                    logger.error(f'Wrong dimension for variable {key}: {ndim}')
            if p.list_tracer is not None:
                nlist = len(listTr)
                for i in range(nlist):
                    Tr = listTr[i]
                    Tr.newvar[:, i0:i1] = Tr.newvarlocal[irank][:, :]
                    Tr.newi[:, i0:i1] = Tr.newilocal[irank][:, :]
                    Tr.newj[:, i0:i1] = Tr.newjlocal[irank][:, :]
        return data


def gather_data(p, list_var_adv, listGr, listTr):
    logger.info('No parallelisation')
    if listTr is not None:
        mod_advection.reordering1d(p, listTr, listGr)
    data = {}
    for key, value in list_var_adv.items():
        data[key] = list_var_adv[key]
    tstep = p.tadvection / p.output_step / abs(p.tadvection)
    tstop = p.first_day + p.tadvection + tstep
    data['time'] = numpy.arange(p.first_day, tstop, tstep)
    return data


def make_list_particles(grid):
    grid.lon1d = + grid.lon.ravel()
    grid.lat1d = + grid.lat.ravel()
    grid.mask1d = + grid.mask.ravel()
    grid.lon1d = grid.lon1d[~numpy.isnan(grid.mask1d)]
    grid.lat1d = grid.lat1d[~numpy.isnan(grid.mask1d)]
    grid.mask1d = grid.mask1d[~numpy.isnan(grid.mask1d)]
