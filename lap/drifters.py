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
# - Written for Python 3.4, tested with Python 3.11
#
Copyright (C) 2015-2024 OceanDataLab
This file is part of lap_toolbox.

lap_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

lap_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with lap_toolbox.  If not, see <http://www.gnu.org/licenses/>.
#-----------------------------------------------------------------------
'''

import datetime
import numpy
import lap.utils.tools as tools
import lap.mod_io as mod_io
import lap.mod_advection as mod_advection
import lap.utils.general_utils as utils
import lap.utils.read_utils as uread
import logging
logger = logging.getLogger(__name__)


def run_drifter(p) -> None:
    tools.make_default(p)
    logger.info('Loading Velocity')
    #VEL, coord = mod_io.read_velocity(p)
    from . import read_utils_xr as rr
    VEL, coord = rr.read_velocity(p)
    _ = drifter(p, VEL, coord, save_netcdf=True)


def drifter(p, VEL: dict, coord: dict, save_netcdf=False) -> dict:
    # - Initialize variables from parameter file
    # ------------------------------------------
    tools.make_default(p)
    comm = None
    p.parallelisation, size, rank, comm = utils.init_mpi(p.parallelisation)

    # - Load tracer and make or read output grid
    # ------------------------------------------
    if rank == 0:
        logger.info(f'Start time {datetime.datetime.now()}')
        logger.info(f'Loading grid for advection for processor {rank}')
        # - Read or make advection grid
        if p.make_grid is False:
            grid = mod_io.read_points(p)
        else:
            grid = mod_io.make_grid(p, VEL, coord)
        # Make a list of particles out of the previous grid
        utils.make_list_particles(grid)

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
        grid = None
        listTr = None
        listGr = None
    if p.parallelisation is True:
        grid = comm.bcast(grid, root=0)
        listTr = comm.bcast(listTr, root=0)
        listGr = comm.bcast(listGr, root=0)

    # - Read velocity
    dic_vel = None
    if rank == 0:
        logger.info('Loading Velocity')
        dic_vel = uread.interp_vel(VEL, coord)
    if p.parallelisation is True:
        dic_vel = comm.bcast(dic_vel, root=0)

    # - Initialise empty variables and particles
    init = utils.init_empty_variables(p, grid, listTr, size, rank)
    dim_hr, dim_lr, grid_size, reducepart, i0, i1 = init

    # - Perform advection
    list_var_adv = mod_advection.advection(reducepart, dic_vel, p,
                                           i0, i1, listGr,
                                           grid, rank=rank, size=size)
    dim_hr[0] = numpy.shape(list_var_adv['lon_hr'])[0]
    dim_lr[0] = numpy.shape(list_var_adv['lon_lr'])[0]
    # - Save output in netcdf file
    if p.parallelisation is True:
        drifter = utils.gather_data_mpi(p, list_var_adv, listGr, listTr,
                                        dim_lr, dim_hr, comm, rank, size,
                                        grid_size)
    else:
        drifter = utils.gather_data(p, list_var_adv, listGr, listTr)

    if rank == 0:
        if save_netcdf is True:
            mod_io.write_drifter(p, drifter, listTr)
        end_time = datetime.datetime.now()
        logger.info(f'End time {end_time}')
        mod_io.write_params(p, end_time)
        return drifter
