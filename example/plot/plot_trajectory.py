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
matplotlib.use('Agg')
from matplotlib import pyplot


def init_cartopy(projection, box=[-180, 180, -90, 90]):
    import cartopy
    # projection = cartopy.crs.Mercator()
    ax = pyplot.axes(projection=projection)
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)
    ax.set_extent([box[0], box[1], box[2], box[3]])
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray')
    gl.xlabels_top = False
    gl.ylabels_left = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    return ax, gl


def plot_trajectory(lon, lat, var, output, box, subsampling=25,
                    is_cartopy=True):
    from matplotlib import pyplot
    import shapely
    pyplot.figure(figsize=(20, 20))
    if is_cartopy is True:
        try:
            import cartopy
        except ImportError:
            logger.warn('Cartopy is not available on this machine')
            is_cartopy = False

    if is_cartopy is True:
        map_proj = cartopy.crs.Mercator()
        data_proj = cartopy.crs.Geodetic()
        ax, gl = init_cartopy(map_proj, box=box)
    else:
        ax = pyplot.axes()
    for pa in range(0, numpy.shape(lon)[1], subsampling):
        if is_cartopy is True:
            track = shapely.geometry.LineString(zip(lon[:, pa], lat[:, pa]))
            #track = geoms.LineString(zip(lon[:, pa], lat[:, pa]))
            # Buffer linestring by two degrees
            # track_buffer = track.buffer(0.5)
            # ax.add_geometries([track], map_proj)
            pyplot.plot(lon[:, pa], lat[:, pa], linewidth=0.5,
                        transform=data_proj)
        else:
            ax.plot(lon[:, pa], lat[:, pa], linewidth=0.5)
    pyplot.savefig(output)


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
    plot_trajectory(lon, dict_var['lat_hr'],
                    dict_var[list_var[0]], output, box=box,
                    subsampling=args.subsampling)


