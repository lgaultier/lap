# author Lucile Gaultier


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
    ax.add_feature(cartopy.feature.LAND, zorder=3)
    ax.add_feature(cartopy.feature.COASTLINE, zorder=3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5, zorder=3)
    ax.add_feature(cartopy.feature.RIVERS, zorder=3)
    ax.set_extent([box[0], box[1], box[2], box[3]])
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=2,
                      alpha=0.5, color='gray')
    gl.xlabels_top = False
    gl.ylabels_left = False
    # gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    return ax, gl


def plot_trajectory(lon, lat, var, output, box, subsampling=25, noocean=False,
                    is_cartopy=True):
    from matplotlib import pyplot
    import shapely
    pyplot.figure(figsize=(15, 7))
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
        if noocean is False:
            ax.add_feature(cartopy.feature.OCEAN, zorder=3)
    else:
        ax = pyplot.axes()
    for pa in range(0, numpy.shape(lon)[1], subsampling):
        if is_cartopy is True:
            track = shapely.geometry.LineString(zip(lon[:, pa], lat[:, pa]))
            pyplot.plot(lon[:, pa], lat[:, pa], linewidth=2,
                        transform=data_proj)
        else:
            ax.plot(lon[:, pa], lat[:, pa], linewidth=2)
    pyplot.savefig(output)


def plot_2dfields(lon, lat, var, output, box, is_cartopy=True):
    from matplotlib import pyplot
    import shapely
    pyplot.figure(figsize=(15, 4.5))
    if is_cartopy is True:
        try:
            import cartopy
        except ImportError:
            logger.warn('Cartopy is not available on this machine')
            is_cartopy = False

    if is_cartopy is True:
        map_proj = cartopy.crs.PlateCarree()
        data_proj = cartopy.crs.PlateCarree()
        ax, gl = init_cartopy(map_proj, box=box)
    else:
        ax = pyplot.axes()
    if is_cartopy is True:
        pyplot.pcolormesh(lon, lat, var, transform=data_proj, cmap='jet')
    else:
        pyplot.pcolormesh(lon, lat, var, cmap='jet')
    pyplot.colorbar()
    pyplot.savefig(output)


def plot_quiver(lon, lat, uvar, vvar, output, box, scale, is_cartopy=True):
    from matplotlib import pyplot
    import shapely
    pyplot.figure(figsize=(15, 4.5))
    if is_cartopy is True:
        try:
            import cartopy
        except ImportError:
            logger.warn('Cartopy is not available on this machine')
            is_cartopy = False

    if is_cartopy is True:
        map_proj = cartopy.crs.PlateCarree()
        data_proj = cartopy.crs.PlateCarree()
        ax, gl = init_cartopy(map_proj, box=box)
    else:
        ax = pyplot.axes()
    norm = numpy.sqrt(uvar **2 + vvar **2)
    if is_cartopy is True:
        pyplot.quiver(lon, lat, uvar, vvar, norm, units='xy', scale=scale,
                      transform=data_proj)
    else:
        pyplot.quiver(lon, lat, uvar, vvar, norm, units='xy', scale=scale)
    pyplot.colorbar()
    pyplot.savefig(output)


def plot_histogram(metx, mety, output, extrem_fit=True, pother=None):
    from scipy.stats import genextreme
    #pyplot.figure(figsize=(7, 15))
    f, ax = pyplot.subplots(1, 2, sharey=True)
    n1, bins1, patches1 = ax[0].hist((metx.ravel()), 100, normed=1, facecolor='g',
                                      alpha=0.75, edgecolor="none")
    if extrem_fit is True:
        p_fit = genextreme.fit(metx)
        y_fit = genextreme.pdf(bins1, p_fit[0], p_fit[1], p_fit[2])
        ax[0].plot(bins1, y_fit, 'b')
        # ax[0].set_title(f'{p_fit[0]:.5f} {p_fit[1]:.5f} {p_fit[2]:.5f}')
        if pother is not None:
            y_fit = genextreme.pdf(bins1, pother[0], pother[1], pother[2])
            ax[0].plot(bins1, y_fit, 'k')
    ax[0].set_ylabel('Probability')
    ax[0].set_xlabel('Longitudinal MET')
    #ax[0].axis([0, 14, 0, 0.45])
    n2, bins2, patches2 = ax[1].hist((mety.ravel()), 100, normed=1, facecolor='r',
                                      alpha=0.75, edgecolor="none")
    ax[1].set_xlabel('Latitudinal MET')
    if extrem_fit is True:
        p_fit = genextreme.fit(mety)
        y_fit = genextreme.pdf(bins2, p_fit[0], p_fit[1], p_fit[2])
        ax[1].plot(bins2, y_fit, 'b')
        # ax[1].set_title(f'{p_fit[0]:.5f} {p_fit[1]:.5f} {p_fit[2]:.5f}')
        if pother is not None:
            y_fit = genextreme.pdf(bins2, pother[3], pother[4], pother[5])
            ax[1].plot(bins2, y_fit, 'k')
    #pyplot.axis([0, 14, 0, 0.45])
    pyplot.savefig(output)
