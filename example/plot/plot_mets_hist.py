# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plot/plot_mets_hist.py
#        /mnt/data/project/dimup/natl60_inst_advection_21215_21265.nc
#        -o /mnt/data/project/dimup -t 500 --box 290 325 34 55 --subsampling 11


import numpy
import os
import netCDF4
import logging
import argparse
import plot_tools
import shapely.geometry as geoms
import shapely
import matplotlib
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
    parser.add_argument('-o', '--output_path',
                        help='Output path. Use "-" for stdout')
    parser.add_argument('-b', '--box', nargs=4, type=float,
                        help=f'Bounding box (lllon, urlon, lllat, urlat)',
                        default=(295., 315., 35., 42.))
    parser.add_argument('-t', '--time', help='Time in netcdf file', type=int,
                        default=400)
    args, inputs = parser.parse_known_args()
    infile = args.input
    output_path = args.output_path
    extent = args.box
    t = args.time

    # Read netcdf file
    fid = netCDF4.Dataset(infile, 'r')
    lon = fid.variables['lon'][:]
    lat = fid.variables['lat'][:]
    metx = fid.variables['METX'][:]
    mety = fid.variables['METY'][:]

    # Plot variables
    basename = os.path.splitext(os.path.basename(infile))[0]
    output = os.path.join(output_path, f'{basename}.png')
    lon = numpy.mod(lon + 360, 360)
    output = os.path.join(output_path, f'metx_{t:03d}_{basename}.png')
    plot_tools.plot_2dfields(lon, lat, metx[t, :, :], output, extent)
    output = os.path.join(output_path, f'mety_{t:03d}_{basename}.png')
    plot_tools.plot_2dfields(lon, lat, mety[t, :, :], output, extent)
    output = os.path.join(output_path, f'hist_{t:03d}_{basename}.png')
    pother = (-0.20523, 5.15571, 3.01864, 0.04769, 3.19413, 1.51592)
    pother = (-0.20199, 5.83660, 3.36631, 0.12884, 3.15708, 1.21617)
    plot_tools.plot_histogram(metx[t, :, :], mety[t, :, :], output, pother=pother)
