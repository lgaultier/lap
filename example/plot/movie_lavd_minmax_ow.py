# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plot/movie_lavd_minmax_ow.py
# /mnt/data/project/dimup/gc_geos_lag_diag_20819_20898.nc
# -o /mnt/data/project/dimup -t 500 --trajectory
# /mnt/data/project/dimup/gc_geos_inst_heavy_advection_21184_21274.nc
#--ow ~/test/dimup/test_lap/eulerian_globcurrent_geos
# -b 295 315 35 43


import numpy
import os
import netCDF4
import matplotlib
import plot_tools
from matplotlib import pyplot
import cartopy
import argparse
from scipy.ndimage import filters
import plot_lavd_minmax_ow as lavd_tools


def plot_cartopy(lon, lat, data, extent, output, lons=None, lats=None,
                 lonsnf=None, latsnf=None, vrange=None, noocean=True,
                 cs_lon=None, cs_lat=None, poly=None, title=None):
    pyplot.figure(figsize=(15, 4.5))
    map_proj = cartopy.crs.PlateCarree()
    data_proj = cartopy.crs.PlateCarree()
    ax, gl = plot_tools.init_cartopy(map_proj, box=extent)
    if noocean is False:
        ax.stock_img()
    if vrange is None:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet',
                          transform=data_proj)
    else:
        pyplot.pcolormesh(lon, lat, data, cmap = 'jet', vmin=vrange[0],
                          vmax=vrange[1], transform=data_proj)
    pyplot.colorbar()
    if cs_lon is not None and cs_lat is not None:
        for i in range(numpy.shape(cs_lon)[1]):
            pyplot.plot(cs_lon[:, i], cs_lat[:,i], linewidth=1.5,
                        transform=data_proj)
    if poly is not None:
        for i in range(len(poly)):
            pyplot.plot(poly[i][:, 0], poly[i][:, 1], 'k', linewidth=2,
                                    transform=data_proj)
    if lons is not None and lats is not None and len(lons) > 0:
        pyplot.scatter(lons, lats, c='k', marker='D',
                       transform=data_proj)
    if lonsnf is not None and latsnf is not None and len(lonsnf) > 0:
        #pyplot.scatter(lons, lats, c='w', transform=cartopy.crs.PlateCarree())
        pyplot.scatter(lonsnf, latsnf, c='r', marker='*',
                       transform=data_proj)
    if title is not None:
        pyplot.title(title)
    pyplot.savefig(output)
    pyplot.close()


if '__main__' == __name__:
    # Parse inputs
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('lavd', help='LAVD file.')
    parser.add_argument('--ow', help='OW path.', default=None)
    parser.add_argument('-o', '--output_path',
                        help='Output path. Use "-" for stdout')
    parser.add_argument('--trajectory', default=None,
                        help='Lagrangian trajectory file')
    parser.add_argument('-b', '--box', nargs=4, type=float,
                        help='Domain for plots',
                        default=(295., 315., 35., 42.))
    parser.add_argument('--eulerian_time', help='Time in netcdf file',
                        type=int, nargs=3, default=(21184, 21265, 1))
    parser.add_argument('-t', '--time', help='Time in netcdf file', type=int,
                        default=400)

    args, inputs = parser.parse_known_args()

    os.makedirs(args.output_path, exist_ok=True)
    t = args.time
    extent = args.box
    lon, lat, lavd = lavd_tools.read_data(args.lavd, 'LAVD')
    data = lavd[t, :, :]
    threshold = 0.5* numpy.nanmean(data)
    data = filters.gaussian_filter(data, sigma=1)
    __, __, vort = lavd_tools.read_data(args.trajectory, 'Vorticity')
    __, __, hlon = lavd_tools.read_data(args.trajectory, 'lon_hr')
    __, __, hlat = lavd_tools.read_data(args.trajectory, 'lat_hr')
    __, __, htime = lavd_tools.read_data(args.trajectory, 'time_hr')
    istart = 0
    istop = len(htime)
    max_lon, max_lat, maxima = lavd_tools.max_lavd(lon, lat, data, threshold)
    cs_lon, cs_lat, polys = lavd_tools.get_contours(lon, lat, data,
                                                    numpy.nanmean(data))
    max_lon2, max_lat2, maxima2 = lavd_tools.filter_max(max_lon, max_lat,
                                                        maxima, polys)
    num_cols = numpy.shape(lavd)[2]
    flat_index = numpy.array(maxima2[1]) + numpy.array(maxima2[0]) * num_cols
    tlon = hlon[:, flat_index]
    tlat = hlat[:, flat_index]
    file_split = os.path.splitext(os.path.basename(args.lavd))[0]
    vrange_ow = [-0.0015, 0.001]
    vrange_vort = [-0.04, 0.04]
    vel_step = 1
    thres_ow = 0.4
    shift = 0
    if 'natl' in file_split:
        thres_ow = 0.08
        vrange_ow = [-5e-2, 5e-2]
        vrange_vort = [-0.3, 0.3]
        vel_step = 5
        shift = 1
    for ind in range(istart, istop):
        jj = int((htime[ind]+shift) / vel_step) * vel_step - shift
        ifile = f'eulerian_{jj:06d}.nc'
        ipath = os.path.join(args.ow, ifile)
        lon2, lat2, ow = lavd_tools.read_data(ipath, 'OW')
        __, __, u = lavd_tools.read_data(ipath, 'U')
        __, __, v = lavd_tools.read_data(ipath, 'V')
        __, __, rv = lavd_tools.read_data(ipath, 'Vorticity')
        diam_lon = tlon[ind, :]
        diam_lat = tlat[ind, :]
        output = os.path.join(args.output_path,
                              f'ow_{t}_{file_split}_{ind:04d}.png')
        ow_th = lavd_tools.threshold_ow(lon2, lat2, ow[0, :, :], extent, thres_ow)
        tindays = int(htime[t] - htime[0])
        adv = htime[ind] - htime[0]
        title=f'{adv:.2f} days of advection, eddies from {tindays} days'
        plot_cartopy(lon2, lat2, ow_th, extent, output, lons=diam_lon,
                     lats=diam_lat, vrange=vrange_ow, cs_lon=tlon, cs_lat=tlat,
                     title=title)
        output = os.path.join(args.output_path,
                              f'vort_{t}_{file_split}_{ind:04d}.png')
        plot_cartopy(lon2, lat2, rv[0, :, :], extent, output, lons=diam_lon,
                     lats=diam_lat, vrange=vrange_vort, cs_lon=tlon, cs_lat=tlat,
                     title=title)


