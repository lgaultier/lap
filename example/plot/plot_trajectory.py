# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plot/plot_trajectory.py
#        /mnt/data/project/dimup/natl60_inst_advection_21215_21265.nc
#        /mnt/data/project/dimup --box 290 325 34 55 --subsampling 11



import os
import numpy
import logging
import argparse
import lap.utils.read_utils as read_utils
import shapely.geometry as geoms
import shapely
import matplotlib
import plot_tools
matplotlib.use('Agg')
from matplotlib import pyplot




if '__main__' == __name__:
    # Setup logging
    main_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    main_logger.addHandler(handler)
    main_logger.setLevel(logging.INFO)

    # Parse input
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('input', help='Input file')
    parser.add_argument('output_path', help='Output path. Use "-" for stdout')
    parser.add_argument('--box', nargs=4, type=float,
                        help=f'Bounding box (lllon, urlon, lllat, urlat)')
    parser.add_argument('--subsampling', type=int,
                        help=f'Subsample particles', default=25)
    parser.add_argument('--list_var', nargs='+', type=str,
                        default=['time_hr'],
                        help='List all variables to plot on trajectory')
    #args, inputs = parser.parse_known_args()
    args = parser.parse_args()
    print(args)
    infile = args.input
    output_path = args.output_path
    box = list(args.box)

    # Read netcdf trajectory file
    list_var = args.list_var
    list_var.append('lon_hr')
    list_var.append('lat_hr')
    dict_var = read_utils.read_trajectory(infile, list_var)

    # Plot variables
    basename = os.path.splitext(os.path.basename(infile))[0]
    output = os.path.join(output_path, f'{basename}.png')
    lon = numpy.mod(dict_var['lon_hr'] + 360, 360)
    plot_tools.plot_trajectory(lon, dict_var['lat_hr'],
                    dict_var[list_var[0]], output, box=box,
                    subsampling=args.subsampling)


