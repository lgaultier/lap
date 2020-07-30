# author Lucile Gaultier
# python ~/src/lap_toolbox/example/plot/plot_lavd_minmax_ow.py
# /mnt/data/project/dimup/gc_geos_lag_diag_20819_20898.nc
# -o /mnt/data/project/dimup -t 500 --trajectory
# /mnt/data/project/dimup/gc_geos_inst_heavy_advection_21184_21274.nc
#--ow ~/test/dimup/test_lap/eulerian_globcurrent_geos/eulerian_021184.nc
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

def read_data(filepath, nvar):
    dset = netCDF4.Dataset(filepath, 'r')
    lon = dset.variables['lon'][:]
    lat = dset.variables['lat'][:]
    var = dset.variables[nvar][:]
    return lon, lat, var


def plot_cartopy(lon, lat, data, extent, output, lons=None, lats=None, lonsnf=None, latsnf=None, 
                 vrange=None, noocean=False, cs_lon=None, cs_lat=None, poly=None):
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
        for i in range(len(cs_lon)):
            pyplot.plot(cs_lon[i], cs_lat[i], linewidth=2,
                        transform=data_proj)
    if poly is not None:
        for i in range(len(poly)):
            pyplot.plot(poly[i][:, 0], poly[i][:, 1], 'k', linewidth=2,
                                    transform=data_proj)
    if lons is not None and lats is not None and len(lons) > 0:
        pyplot.scatter(lons, lats, c='w', transform=data_proj)
        pyplot.scatter(lons, lats, c='k', marker='+',
                       transform=data_proj)
    if lonsnf is not None and latsnf is not None and len(lonsnf) > 0:
        #pyplot.scatter(lons, lats, c='w', transform=cartopy.crs.PlateCarree())
        pyplot.scatter(lonsnf, latsnf, c='r', marker='*',
                       transform=data_proj)
    pyplot.savefig(output)
    pyplot.close()


def max_lavd(lon, lat, data, threshold):
    data = filters.gaussian_filter(data, sigma=2)
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


def get_contours(lon, lat, data, threshold):
    from skimage import measure
    # Find contours at a constant value of 0.8
    data = filters.gaussian_filter(data, sigma=4)
    cs_lon = []
    cs_lat = []
    all_polys = []
    for perc in range(50, 100, 4):
        threshold = numpy.percentile(data, perc)
        contours = measure.find_contours(data, threshold)
        # import pdb ; pdb.set_trace()
        for i in range(len(contours)):
             ## Append if contour is convex
             coords = measure.approximate_polygon(contours[i], tolerance=4.5)
             c = numpy.zeros((numpy.shape(coords)[0] - 2))
             for k in range(1, numpy.shape(coords)[0] - 1):
                 c[k-1] = ((coords[k, 0] - coords[k-1, 0])
                         / (coords[k+1, 1] - coords[k, 1])
                         - (coords[k, 1] - coords[k-1, 1])
                         / (coords[k+1, 0] - coords[k, 0]))
             if not numpy.all(c >= 0) and not numpy.all(c <= 0):
                 continue
             coords = measure.approximate_polygon(contours[i], tolerance=0)
             cs_lon.append(lon[numpy.array(contours[i][:, 0], dtype=int),
                               numpy.array(contours[i][:, 1], dtype=int)])
             cs_lat.append(lat[numpy.array(contours[i][:, 0], dtype=int),
                               numpy.array(contours[i][:, 1], dtype=int)])

             polygon = [[lon[int(x[0]), int(x[1])], lat[int(x[0]), int(x[1])]]
                        for x in coords]
             all_polys.append(numpy.array(polygon))
    print(numpy.shape(cs_lon), 'get_contour: cs_lon')
    return cs_lon, cs_lat, all_polys


def filter_max(lon, lat, maxima, polys):
    import shapely

    polygons = []
    for i in range(len(polys)):
        if numpy.shape(polys[i])[0] > 2:
            linear_ring = shapely.geometry.polygon.LinearRing(polys[i])
            polygon = shapely.geometry.polygon.asPolygon(linear_ring)
            polygons.append(polygon)

    out_lon = []
    out_lat = []
    out_i = []
    out_j = []
    for i in range(len(lon)):
        point = shapely.geometry.point.Point(lon[i], lat[i])
        for j in range(len(polygons)):
            point_in_polygon = polygons[j].contains(point)
            if point_in_polygon is True:
                out_lon.append(lon[i])
                out_lat.append(lat[i])
                out_i.append(maxima[0][i])
                out_j.append(maxima[1][i])
                break
    return out_lon, out_lat, [out_i, out_j]


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
    cs_lon, cs_lat, polys = get_contours(lon, lat, data, numpy.nanmean(data))
    max_lon2, max_lat2, maxima2 = filter_max(max_lon, max_lat, maxima, polys)
    if 'natl' in file_split:
        #plot_cartopy(lon, lat, data, extent, output, lons=max_lon,
        #            lats=max_lat, vrange=[0, 2], cs_lon=cs_lon, cs_lat=cs_lat)
        plot_cartopy(lon, lat, data, extent, output, lons=max_lon2,
                     lats=max_lat2, vrange=[0, 2], poly=polys)
    else:
        plot_cartopy(lon, lat, data, extent, output, lons=max_lon,
                     lats=max_lat, cs_lon=cs_lon, cs_lat=cs_lat)
    if args.ow is not None:
        thres_ow = 0.7
        ow_th = threshold_ow(lon2, lat2, ow[0, :, :], extent, thres_ow)
        output = os.path.join(args.output_path,
                              f'ow_min_max_{t}_{file_split}.png')
        if 'natl' in file_split:
            plot_cartopy(lon2, lat2, ow_th[:, :] ,extent, output, lons=max_lon2,
                         lats=max_lat2, vrange=[-5e-2, 5e-2], noocean=True,
                         cs_lon=cs_lon, cs_lat=cs_lat)
        else:
            plot_cartopy(lon2, lat2, ow_th[:, :] ,extent, output, lons=max_lon2,
                         lats=max_lat2, noocean=True,
                         cs_lon=cs_lon, cs_lat=cs_lat)

        output = os.path.join(args.output_path,
                              f'vorticity_min_max_{t}_{file_split}.png')
        if 'natl' in file_split:
            plot_cartopy(lon2, lat2, rv[0, :, :] ,extent, output, lons=max_lon2,
                         lats=max_lat2, vrange=[-0.3, 0.3],
                         cs_lon=cs_lon, cs_lat=cs_lat)
        else:
            plot_cartopy(lon2, lat2, rv[0, :, :] ,extent, output, lons=max_lon2,
                         lats=max_lat2,
                         cs_lon=cs_lon, cs_lat=cs_lat)
        output = os.path.join(args.output_path,
                              f'vel_min_max_{t}_{file_split}.png')
        vel = numpy.sqrt(u**2 + v**2)
        plot_cartopy(lon2, lat2, vel[0, :, :] ,extent, output, lons=max_lon,
                     lats=max_lat, vrange=[0, 1])
    if args.trajectory is not None:
        __, __, vort = read_data(args.trajectory, 'Vorticity')
        __, __, hlon = read_data(args.trajectory, 'lon_hr')
        __, __, hlat = read_data(args.trajectory, 'lat_hr')
        print(maxima2, maxima)
        output = os.path.join(args.output_path, f'traj_{t}_{file_split}.png')
        # flat_index = col + row * num_cols
        num_cols = numpy.shape(lavd)[2]
        flat_index = maxima[1] + maxima[0] * num_cols
    #    plot_trajectory()

        plot_tools.plot_trajectory(hlon[:, flat_index], hlat[:, flat_index],
                                   vort[:, flat_index], output, extent,
                                   subsampling=1)

        output = os.path.join(args.output_path, f'traj2_{t}_{file_split}.png')
        # flat_index = col + row * num_cols
        num_cols = numpy.shape(lavd)[2]
        flat_index = numpy.array(maxima2[1]) + numpy.array(maxima2[0]) * num_cols
        print(flat_index)
    #    plot_trajectory()

        plot_tools.plot_trajectory(hlon[:, flat_index], hlat[:, flat_index],
                        vort[:, flat_index], output, extent,
                        subsampling=1)

