# author Lucile Gaultier
# python plot_lavd_ow.py -o /mnt/data/project/dimup -t 500
# /mnt/data/project/dimup/aviso_lag_diag_20819_20898.nc
# ./eulerian_aviso/eulerian_020819.nc

import numpy
import os
import netCDF4
import matplotlib
from matplotlib import pyplot
import cartopy
import argparse
from scipy.ndimage import filters

def read_data(filepath, nvar):
    dset = netCDF4.Dataset(filepath, 'r')
    lon = dset.variables['lon'][:]
    lat = dset.variables['lat'][:]
    var = dset.variables[nvar][:]
    return lon, lat, var


def plot_cartopy(lon, lat, data, extent, output, lons=None, lats=None,
                 vrange=None):
    pyplot.figure(figsize=(22, 8))
    ax = pyplot.axes(projection=cartopy.crs.PlateCarree())
    ax.set_extent(extent)
    ax.add_feature(cartopy.feature.LAND, zorder=3)
    #ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=3)
    ax.gridlines(draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.stock_img()
    if vrange is None:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet',
                          transform=cartopy.crs.PlateCarree())
    else:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet', vmin=-vrange,
                          vmax=vrange, transform=cartopy.crs.PlateCarree())
    pyplot.colorbar()
    if lons is not None and lats is not None:
        pyplot.scatter(lons, lats, c='w', transform=cartopy.crs.PlateCarree())
    pyplot.savefig(output)


def max_lavd(lon, lat, data):
    neighborhood_size = 20
    threshold = numpy.mean(data)
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima = numpy.where((data==data_max) & ((data_max - data_min) > threshold))
    return lon[maxima], lat[maxima]


if '__main__' == __name__:
    # Parse inputs
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('lavd', help='LAVD file.')
    parser.add_argument('ow', help='OW file.')
    parser.add_argument('-o', '--output_path',
                        help='Output path. Use "-" for stdout')
    parser.add_argument('-b', '--box', nargs=4, type=float,
                        help='Domain for plots',
                        default=(295., 315., 35., 42.))
    parser.add_argument('-t', '--time', help='Time in netcdf file', type=int,
                        default=400)
    args, inputs = parser.parse_known_args()

    t = args.time
    extent = args.box
    lon, lat, lavd = read_data(args.lavd, 'LAVD')
    lon2, lat2, ow = read_data(args.ow, 'OW')
    lon2, lat2, u = read_data(args.ow, 'U')
    lon2, lat2, v = read_data(args.ow, 'V')
    file_split = os.path.splitext(os.path.basename(args.lavd))[0]
    output = os.path.join(args.output_path,
                          f'lavd_{t}_{file_split}.png')
    data = lavd[t, :, :]
    data = filters.gaussian_filter(data, sigma=3)
    plot_cartopy(lon, lat, data, extent, output, lons=None, lats=None)
    max_lon, max_lat = max_lavd(lon, lat, data)
    print(max_lon, max_lat)
    output = os.path.join(args.output_path,
                          f'lavd_min_max_{t}_{file_split}.png')
    plot_cartopy(lon, lat, data, extent, output, lons=max_lon, lats=max_lat)
    output = os.path.join(args.output_path,
                          f'ow_min_max_{t}_{file_split}.png')
    if 'natl' in file_split:
        plot_cartopy(lon2, lat2, ow[0, :, :] ,extent, output, lons=max_lon,
                     lats=max_lat, vrange=5e-2)
    else:
        plot_cartopy(lon2, lat2, ow[0, :, :] ,extent, output, lons=max_lon,
                     lats=max_lat)

    output = os.path.join(args.output_path,
                          f'vel_min_max_{t}_{file_split}.png')
    vel = numpy.sqrt(u**2 + v**2)
    plot_cartopy(lon2, lat2, vel[0, :, :] ,extent, output, lons=max_lon,
                 lats=max_lat, vrange=1)
