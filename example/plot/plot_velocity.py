# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plot/plot_quiver.py
#        /mnt/data/project/dimup/
#        -o /mnt/data/project/dimup --box 290 325 34 55 --subsampling 11


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
    parser.add_argument('-s', '--subsampling',
                        help='Subsampling in netcdf file', type=int,
                        default=1)
    args, inputs = parser.parse_known_args()
    infile = args.input
    output_path = args.output_path
    extent = args.box
    s = args.subsampling

    # Read netcdf file
    fid = netCDF4.Dataset(infile, 'r')
    lon = fid.variables['lon'][:]
    lat = fid.variables['lat'][:]
    u = fid.variables['U'][:]
    v = fid.variables['V'][:]

    # Plot variables
    basename = os.path.splitext(os.path.basename(infile))[0]
    output = os.path.join(output_path, f'vel_{basename}.png')
    lon = numpy.mod(lon + 360, 360)
    scale=1
    norm = numpy.sqrt(u**2 + v**2)
    u = numpy.ma.masked_where(norm > 1.6, u)
    v = numpy.ma.masked_where(norm > 1.6, v)
    u = numpy.ma.masked_less(u, -1.5)
    v = numpy.ma.masked_less(v, -1.5)
    u = numpy.ma.masked_greater(u, 1.5)
    v = numpy.ma.masked_greater(v, 1.5)
    plot_tools.plot_quiver(lon[::s], lat[::s], u[0, ::s, ::s], v[0, ::s, ::s],
                           output, extent, scale)
