# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plotplot_lavd_ow.py -o /mnt/data/project/dimup -t 500
# /mnt/data/project/dimup/aviso_lag_diag_20819_20898.nc
# ./eulerian_aviso/eulerian_020819.nc

import numpy
import os
import netCDF4
import matplotlib
import plot_tools
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
                 vrange=None, noocean=False):
    pyplot.figure(figsize=(15, 4.5))
    map_proj = cartopy.crs.Mercator()
    data_proj = cartopy.crs.Geodetic()
    ax, gl = plot_tools.init_cartopy(map_proj, box=extent)
    if noocean is False:
        ax.stock_img()
    if vrange is None:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet',
                          transform=cartopy.crs.PlateCarree())
    else:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet', vmin=vrange[0],
                          vmax=vrange[1], transform=cartopy.crs.PlateCarree())
    pyplot.colorbar()
    if lons is not None and lats is not None:
        pyplot.scatter(lons, lats, c='w', transform=cartopy.crs.PlateCarree())
        pyplot.scatter(lons, lats, c='k', marker='+',
                       transform=cartopy.crs.PlateCarree())
    pyplot.savefig(output)


def max_lavd(lon, lat, data, threshold):
    neighborhood_size = int(1.5 / numpy.nanmean(abs(lon[:, 1:]-lon[:, :-1])))
    # threshold = numpy.mean(data)
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima = numpy.where((data==data_max) & ((data_max - data_min) > threshold))
    return lon[maxima], lat[maxima], maxima


def threshold_ow(lon, lat, ow, extent, threshold):
    ind_lon = numpy.where((lon >= extent[0]) & (lon <= extent[1]))
    ow_out = ow[:, ind_lon]
    ind_lat = numpy.where((lat >= extent[2]) & (lat <= extent[3]))
    ow_out = ow[ind_lat, :]
    ow_out[abs(ow_out) > 1] = numpy.nan
    std_ow = numpy.nanstd(ow_out)
    print(std_ow)
    mask = numpy.where((ow > -threshold*std_ow) & (ow < threshold*std_ow))
    ow_out = + ow
    ow_out[mask] = numpy.nan
    ow_out = numpy.ma.masked_invalid(ow_out)
    return ow_out

def plot_trajectory(lon, lat, var, output, box, contours, subsampling=1,
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
        ax, gl = plot_tools.init_cartopy(map_proj, box=box)
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
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    pyplot.savefig(output)


if '__main__' == __name__:
    # Parse inputs
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('lavd', help='LAVD file.')
    parser.add_argument('--ow', help='OW file.', default=None)
    parser.add_argument('-o', '--output_path',
                        help='Output path. Use "-" for stdout')
    parser.add_argument('--trajectory', default=None,
                        help='Lagrangian trajectory file')
    parser.add_argument('-b', '--box', nargs=4, type=float,
                        help='Domain for plots',
                        default=(295., 315., 35., 42.))
    parser.add_argument('-t', '--time', help='Time in netcdf file', type=int,
                        default=400)
    args, inputs = parser.parse_known_args()

    t = args.time
    extent = args.box
    lon, lat, lavd = read_data(args.lavd, 'LAVD')
    if args.ow is not None:
        lon2, lat2, ow = read_data(args.ow, 'OW')
        lon2, lat2, u = read_data(args.ow, 'U')
        lon2, lat2, v = read_data(args.ow, 'V')
        _, _, rv = read_data(args.ow, 'Vorticity')
    file_split = os.path.splitext(os.path.basename(args.lavd))[0]
    output = os.path.join(args.output_path,
                          f'lavd_{t}_{file_split}.png')
    data = lavd[t, :, :]
    threshold = 0.5* numpy.nanmean(data)
    data = filters.gaussian_filter(data, sigma=1)
    plot_cartopy(lon, lat, data, extent, output, lons=None, lats=None, vrange=[0, 2])
    max_lon, max_lat, maxima = max_lavd(lon, lat, data, threshold)
    output = os.path.join(args.output_path,
                          f'lavd_min_max_{t}_{file_split}.png')
    if 'natl' in file_split:
        plot_cartopy(lon, lat, data, extent, output, lons=max_lon,
                     lats=max_lat, vrange=[0, 2])
    else:
        plot_cartopy(lon, lat, data, extent, output, lons=max_lon, lats=max_lat)
    if args.ow is not None:
        thres_ow = 0.08
        ow_th = threshold_ow(lon2, lat2, ow[0, :, :], extent, thres_ow)
        output = os.path.join(args.output_path,
                              f'ow_min_max_{t}_{file_split}.png')
        if 'natl' in file_split:
            plot_cartopy(lon2, lat2, ow_th[:, :] ,extent, output, lons=max_lon,
                         lats=max_lat, vrange=[-5e-2, 5e-2], noocean=True)
        else:
            plot_cartopy(lon2, lat2, ow_th[:, :] ,extent, output, lons=max_lon,
                         lats=max_lat, noocean=True)

        output = os.path.join(args.output_path,
                              f'vorticity_min_max_{t}_{file_split}.png')
        if 'natl' in file_split:
            plot_cartopy(lon2, lat2, rv[0, :, :] ,extent, output, lons=max_lon,
                         lats=max_lat, vrange=[-0.3, 0.3])
        else:
            plot_cartopy(lon2, lat2, rv[0, :, :] ,extent, output, lons=max_lon,
                         lats=max_lat)
        output = os.path.join(args.output_path,
                              f'vel_min_max_{t}_{file_split}.png')
        vel = numpy.sqrt(u**2 + v**2)
        plot_cartopy(lon2, lat2, vel[0, :, :] ,extent, output, lons=max_lon,
                     lats=max_lat, vrange=[0, 1])
    if args.trajectory is not None:
        __, __, vort = read_data(args.trajectory, 'Vorticity')
        __, __, hlon = read_data(args.trajectory, 'lon_hr')
        __, __, hlat = read_data(args.trajectory, 'lat_hr')
        print(maxima)
        output = os.path.join(args.output_path, f'traj_{file_split}.png')
        # flat_index = col + row * num_cols
        num_cols = numpy.shape(lavd)[2]
        flat_index = maxima[1] + maxima[0] * num_cols
        print(flat_index, numpy.shape(hlon)) #[:, flat_index]))
    #    plot_trajectory()
        from skimage import measure

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(lavd[t, :, :], 0.8)

        plot_trajectory(hlon[:, flat_index], hlat[:, flat_index],
                        vort[:, flat_index], output, extent, contours,
                        subsampling=1)
