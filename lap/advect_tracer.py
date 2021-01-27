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
from typing import Optional, Tuple
import sys
import lap.utils.tools as tools
import lap.mod_io as mod_io
import lap.mod_advection as mod_advection
import lap.utils.general_utils as utils
import logging
logger = logging.getLogger(__name__)

# TODO check code
def tracer_advection(p, Tr, VEL, AMSRgrid=None):
    tools.make_default(p)
    if p.make_grid is False:
        grid = mod_io.read_grid_netcdf(p)
    else:
        grid = mod_io.make_grid(p)
    # Make a list of particles out of the previous grid
    utils.make_list_particles(grid)
    # - Initialise empty variables and particle
    init = utils.init_empty_variables(p, grid, listTr, size, rank)
    dim_hr, dim_lr, grid_size, reducepart, i0, i1 = init

    # - Advect particles on each processor
    list_var_adv = mod_advection.advection(reducepart, VEL, p, i0, i1, listTr,
                                           grid, rank=rank, size=size)

    # - Save output in netcdf file
    if p.parallelisation is True:
        data_out = utils.gather_data_mpi(p, grid, list_var_adv, listGr, listTr,
                                         dim_lr, dim_hr, comm, rank, size,
                                         grid_size)
    else:
        data_out = utils.gather_data(p, grid, list_var_adv, listGr, listTr)

    if rank == 0:
        mod_advection.reordering(Tr, data_out, AMSR, p)
    return data_out


def run_tracer_advection(p):
    tools.make_default(p)
    logger.info('Loading Velocity')
    VEL, coord = mod_io.read_velocity(p)
    tracer_advection(p, VEL, coord)


def tracer_advection(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    tools.make_default(p)
    comm = None
    listTr = None
    listGr = None
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

        # TODO: read a mask or a grid?
        # else:
        #   if p.TRATYPE=='AMSR':
        #     grid=mod_io.Read_grid_HDF(p)
        #   elif p.TRATYPE=='MODIS':
        #     j0=7900 ; j1=8150 ; i0=1250; i1=1550
        #     grid=mod_io.Read_grid_modis_hr(p, i0, i1, j0, j1)
        #   elif p.TRATYPE=='MODEL':
        #     grid=mod_io.Read_grid_model(p)
        # elif p.type_grid=='make':
        #     lon0, lon1, dlon, lat0, lat1, dlat=list( p.par_grid )
        # grid=mod_io.make_grid(p)
        # else:
        # 'Umknown type of grid, please choose between CDF, AMSR or make'
        # - Read initial tracer


        # #TODO Adapt with read_list_tracer
        # - Read tracer to collocate
        if p.list_tracer is not None:
            dict_tracer = tools.available_tracer_collocation()
            logger.info('Loading tracer')
            listTr = list(mod_io.read_list_tracer(p, dict_tracer))
            logger.info('Loading tracer grid')
            listGr = list(mod_io.read_list_grid(p, dict_tracer))
        else:
            listTr = None
            listGr = None
    else:
        Tr = None
    if p.parallelisation is True:
        grid = comm.bcast(grid, root=0)
        listTr = comm.bcast(listTr, root=0)
        listGr = comm.bcast(listGr, root=0)

    # - Read velocity
    if rank == 0:
        VEL = utils.interp_vel(VEL, coord)
        logger.info('Loading Velocity')
    if p.parallelisation is True:
        VEL = comm.bcast(VEL, root=0)
        dic_vel = comm.bcast(dic_vel, root=0)


    # - Read AMSR if nudging
    # TODO check loading of AMSR data
    if p.gamma:
        AMSR = mod_io.Read_amsr_t(p)
    else:
        AMSR = None

    # - Initialise empty variables and particle
    init = utils.init_empty_variables(p, grid, listTr, size, rank)
    dim_hr, dim_lr, grid_size, reducepart, i0, i1 = init

    # - Advect particles on each processor
    list_var_adv = mod_advection.advection(reducepart, VEL, p, i0, i1, listTr,
                                           grid, rank=rank, size=size)

    # - Save output in netcdf file
    if p.parallelisation is True:
        data_out = utils.gather_data_mpi(p, grid, list_var_adv, listGr, listTr,
                                         dim_lr, dim_hr, comm, rank, size,
                                         grid_size)
    else:
        data_out = utils.gather_data(p, grid, list_var_adv, listGr, listTr)

    if rank == 0:
        mod_advection.reordering(Tr, data_out, AMSR, p)
        mod_io.write_advected_tracer(p, data_out)
        logger.info(f'End time {datetime.datetime.now()}')
