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


def run_tracer_advection(p):
    # - Initialize variables from parameter file
    # ------------------------------------------
    mod_tools.make_default(p)
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
        if p.tracer_type == 'netcdf':
            Tr = mod_io.Read_tracer_hr(p)
        elif p.tracer_type == 'AMSR':
            Tr = mod_io.Read_amsr(p)
        elif p.tracer_type == 'MODIS':
            j0 = 7900
            j1 = 8150
            i0 = 1250
            i1 = 1550
            Tr = mod_io.Read_modis_hr(p, i0, i1, j0, j1)
        elif p.tracer_type == 'MODEL':
            Tr = mod_io.Read_tracer_model(p)
    else:
        Tr = None
    if p.parallelisation is True:
        grid = comm.bcast(grid, root=0)
        listTr = comm.bcast(listTr, root=0)
        listGr = comm.bcast(listGr, root=0)

    # - Read velocity
    if rank == 0:
        logger.info('Loading Velocity')
    VEL = mod_io.read_velocity(p)

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


# - Gather data in processor 0 and save them
#  if MPI:
#    comm.barrier()
#    newilocal=comm.gather(newi, newi, root=0)
#    newjlocal=comm.gather(newj, newj, root=0)
#    if rank==0:
#        grid.iit[:,:,i0:i1]=newilocal[irank][:,:,:]
#        grid.ijt[:,:,i0:i1]=newjlocal[irank][:,:,:]

#  else:
#    print('no MPI')
#    grid.iit[:,:,:]=newi
#    grid.ijt[:,:,:]=newj
#    grid.i=newi
#    grid.j=newj
#    #grid.var2=grid.var[grid.i,grid.j]
#
#    rw_data.write_cdf2d(output, grid)
