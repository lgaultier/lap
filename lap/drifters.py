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
#
# Notes:
# - Written for Python 3.4, tested with Python 3.6
#
# Copyright (c)
#-----------------------------------------------------------------------
'''

import datetime
import lap.mod_tools as mod_tools
import lap.mod_io as mod_io
import lap.mod_advection as mod_advection
import lap.utils.general_utils as utils
import logging
logger = logging.getLogger(__name__)


def run_drifter(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    mod_tools.make_default(p)
    comm = None
    p.parallelisation, size, rank, comm = utils.init_mpi(p.parallelisation)

    # - Load tracer and make or read output grid
    # ------------------------------------------
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        # - Read or make advection grid
        if p.make_grid is False:
            grid = mod_io.read_grid_netcdf(p)
        else:
            grid = mod_io.make_grid(p)
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)

        # - Read tracer to collocate
        if p.list_tracer is not None:
            dict_tracer = mod_tools.available_tracer_collocation()
            logger.info('Loading tracer')
            listTr = list(mod_io.read_list_tracer(p, dict_tracer))
            logger.info('Loading tracer grid')
            listGr = list(mod_io.read_list_grid(p, dict_tracer))
        else:
            listTr = None
            listGr = None
    else:
        grid = None
        listTr = None
        listGr = None
    if p.parallelisation is True:
        grid = comm.bcast(grid, root=0)
        listTr = comm.bcast(listTr, root=0)
        listGr = comm.bcast(listGr, root=0)

    # - Read velocity
    if rank == 0:
        logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p)

    # - Initialise empty variables and particles
    init = utils.init_empty_variables(p, grid, listTr, size, rank)
    dim_hr, dim_lr, grid_size, reducepart, i0, i1 = init

    # - Perform advection
    list_var_adv = mod_advection.advection(reducepart, VEL, p, i0, i1, listGr,
                                           grid, rank=rank, size=size)
    import numpy
    print(dim_hr)
    dim_hr = numpy.shape(list_var_adv['lon_hr'])
    dim_lr = numpy.shape(list_var_adv['lon_lr'])
    print(dim_hr)
    # - Save output in netcdf file
    if p.parallelisation is True:
        drifter = utils.gather_data_mpi(p, list_var_adv, listGr, listTr,
                                        dim_lr, dim_hr, comm, rank, size,
                                        grid_size)
    else:
        drifter = utils.gather_data(p, list_var_adv, listGr, listTr)

    if rank == 0:
        mod_io.write_drifter(p, drifter, listTr)
        logger.info(f'End time {datetime.datetime.now()}')
