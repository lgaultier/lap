import datetime
import numpy
import lap.utils.read_utils as read_utils


def compute_mets(plon, plat):
    xmax = numpy.nanmax(plon, axis=0)
    xmin = numpy.nanmin(plon, axis=0)
    ymax = numpy.nanmax(plat, axis=0)
    ymin = numpy.nanmin(plat, axis=0)
    METx = (xmax-xmin)
    METy = (ymax-ymin)
    return METx, METy


def compute_mezic(plon, plat, t0=0):
    ntime, npa = numpy.shape(plon)
    xt0 = numpy.zeros((ntime - t0, npa))
    yt0 = numpy.zeros((ntime - t0, npa))
    for t in range(ntime - t0):
        xt0[t, :] = (plon[t, :] - plon[t0, :])
        yt0[t, :] = plat[t, :] - plat[t0, :]
    return xt0, yt0


def compute_lavd(plon, plat, ptime, pvort, mean_vort, t0=0):
    ntime, npa = numpy.shape(plon)
    lavd = numpy.zeros((ntime - t0, npa))
    lavd_time = numpy.zeros((ntime - t0))
    mean_vort = numpy.mean(pvort, axes=2)
    tini = t0 + 1
    for t in range(ntime - t0):
        tf = t + t0 + 2
        index_f = floor(tf * p.adv_time_step)
        index_i = floor(t0 * p.adv_time_step)
        lavd_tmp = numpy.sum(numpy.abs(pvort[t0 + 1: tf, :]
                             - mean_vort[index_i: index_f, numpy.newaxis]),
                             axis=0)
        lavd_t0 = numpy.abs(pvort[pvort[t0, :] - mean_vort[index_i]) / 2
        lavd_t = numpy.abs(pvort[tinipvort[tf, :] - mean_vort[index_f]) / 2
        lavd_tmp = lavd_tmp + lavd_t0 + lavd_t
        lavd_time[t] = ptime[tf] - ptime[t0]
        lavd[t, :] = lavd_tmp
    return lavd, lavd_time


def lagrangian_diag(p):
    logger.info(f'Start time {datetime.datetime.now()}')
    list_var = ['lon_hr', 'lat_hr', 'time_hr', 'u_hr', 'Vorticity']
    dict_var = read_utils.read_trajectory(p.output, list_var)
    lon = dict_var['lon_hr']
    lat = dict_var['lat_hr']
    vort = dict_var['Vorticity']
    ptime = dict_var['time_hr']
    num_t, num_pa = numpy.shape(lon)
    if p.make_grid is False:
        grid = mod_io.read_grid_netcdf(p)
    else:
        grid = mod_io.make_grid(p)
    shape_grid = numpy.shape(grid.lon)
    metx_all = []
    mety_all = []
    mezic_all = []
    for pa in num_pa:
        if 'MET' in p.diagnostic:
            metx, mety = compute_mets(lon[:, pa], lat[:, pa])
            metx_all.append(metx)
            mety_all.append(mety)
        if 'Mezic' in p.diagnostic:
            mezic_strain = compute_mezic(lon[:, pa], lat[:, pa])
            mezic_all.append(mezic_strain)
    if 'MET' in p.diagnostic:
        metx_2d = numpy.array(metx_all).reshape(grid_shape)
        mety_2d = numpy.array(mety_all).reshape(grid_shape)
    else:
        metx_2d = None
        mety_2d = None
    if 'Mezic' in p.diagnostic:
        mezic_2D = numpy.array(mezic_all).reshape(grid_shape)
    else:
        mezic_2D = None

    if 'LAVD' in p.diagnostic:
        p.save_RV = True
        VEL = mod_io.read_velocity(p)
        mean_vort = numpy.mean(numpy.mean(VEL.RV, axis=2), axis=1)
        lavd, lavd_t = compute_lavd(lon, lat, ptime, pvort, mean_vort)
        lavd_2D = numpy.zeros((numpy.shape(lavd)[0], grid_shape[0],
                               grid_shape[1]))
        for t in range(numpy.shape(lavd)[0]):
            lavd_2D[t, :, :] = lavd[t, :].reshape(grid_shape)
    else
        lavd_2D = None
    time = numpy.arange(t0, ntime)
    data={}
    data['lon'] = grid.lon
    data['lat'] = grid.lat
    description = 'lagrangian diagnostics'
    mod_io.write_diagnostic_2d(p, data, description=description, METX=metx_2D,
                               METY=mety_2D, MEZIC=mezic_2D, LAVD=lavd_2D,
                               time=time)
    logger.info(f'Stop time {datetime.datetime.now()}')
