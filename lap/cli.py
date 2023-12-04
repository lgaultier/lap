'''
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
'''

import sys
import lap.utils.tools as tools
import logging
logger = logging.getLogger()
handler = logging.StreamHandler()


def run_drifters_script() -> None:
    ''' Advect fictive particles, possibility to collocated advected position
    with tracer observation. '''
    import lap.drifters as drifters
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if len(sys.argv) < 2:
        logger.error('Please specify a parameter file')
        sys.exit(1)
    else:
        file_param = str(sys.argv[1])

    p = tools.load_python_file(file_param)
    drifters.run_drifter(p)


def run_eulerian_diags_script() -> None:
    ''' Compute Eulerian diagnostics on a 2D grid such as the Relative
    Vorticity, the Strain and Okubo Weiss. '''
    import lap.eulerian_tools as eulerian
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if len(sys.argv) < 2:
        logger.error('Please specify a parameter file')
        sys.exit(1)
    else:
        file_param = str(sys.argv[1])

    p = tools.load_python_file(file_param)
    eulerian.compute_eulerian_diags(p)


def run_lyapunov_script() -> None:
    ''' Compute FSLE or FTLE. '''
    import lap.lyapunov as lyapunov
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if len(sys.argv) < 2:
        logger.error('Please specify a parameter file')
        sys.exit(1)
    else:
        file_param = str(sys.argv[1])
    print('Compute_fsle')
    p = tools.load_python_file(file_param)
    lyapunov.run_lyapunov(p)


def run_lagrangian_script() -> None:
    ''' Compute Lagrangian diagnostics. '''
    import lap.lag_diag as lag_diag
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    if len(sys.argv) < 2:
        logger.error('Please specify a parameter file')
        sys.exit(1)
    else:
        file_param = str(sys.argv[1])

    p = tools.load_python_file(file_param)
    lag_diag.lagrangian_diag(p)
