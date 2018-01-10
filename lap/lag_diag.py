import datetime
import numpy
import lap.utils.read_utils as read_utils
import lap.mod_io as mod_io
import lap.mod_tools as mod_tools
import logging
logger = logging.getLogger(__name__)


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
    for t in range(1, ntime - t0):
        xt0[t, :] = (plon[t + t0, :] - plon[t0, :]) / t
        yt0[t, :] = (plat[t + t0, :] - plat[t0, :]) / t
    return xt0, yt0


def compute_lavd(p, plon, plat, ptime, pvort, mean_vort, t0=0):
    ntime, npa = numpy.shape(plon)
    lavd = numpy.zeros((ntime - t0, npa))
    lavd_time = numpy.zeros((ntime - t0))
    ntime_vel = numpy.shape(mean_vort)[0]
    #mean_vort = numpy.mean(pvort, axis=2)
    tini = t0 + 1
    tvel = numpy.arange(0, numpy.shape(mean_vort)[0])
    tpa = numpy.arange(0, ntime)

    mvort_t = numpy.interp(tpa, tvel, mean_vort)
    for t in range(ntime - t0 - 2):
        tf = t + t0 + 2
        index_f = min(int(tf * p.adv_time_step / p.vel_step), ntime_vel - 2)
        lavd_tmp = numpy.sum(numpy.abs(pvort[t0 + 1: tf, :]
                             - mvort_t[t0 + 1: tf, numpy.newaxis]),
                             axis=0)
        lavd_t0 = numpy.abs(pvort[t0, :] - mvort_t[t0]) / 2
        lavd_t = numpy.abs(pvort[tf, :] - mvort_t[tf]) / 2
        lavd_tmp = lavd_tmp + lavd_t0 + lavd_t
        lavd_time[t] = ptime[tf] - ptime[t0]
        lavd[t, :] = lavd_tmp / lavd_time[t]
    return lavd, lavd_time


def lagrangian_diag(p):
    logger.info(f'Start time {datetime.datetime.now()}')
    mod_tools.make_default(p)
    list_var = ['lon_hr', 'lat_hr', 'time_hr', 'zonal_velocity',
                'meridional_velocity', 'Vorticity']
    t0 = 0
    dict_var = read_utils.read_trajectory(p.input_path, list_var)
    plon = dict_var['lon_hr']
    plat = dict_var['lat_hr']
    pvort = dict_var['Vorticity']
    ptime = dict_var['time_hr']
    num_t, num_pa = numpy.shape(plon)
    if p.make_grid is False:
        grid = mod_io.read_grid_netcdf(p)
    else:
        grid = mod_io.make_grid(p)
    shape_grid = numpy.shape(grid.lon)
    print(shape_grid)
    mezic_all = []
    if 'MET' in p.diagnostic:
        logger.info('compute MET')
        metx_2d = numpy.empty((num_t - t0, shape_grid[0], shape_grid[1]))
        mety_2d = numpy.empty((num_t - t0, shape_grid[0], shape_grid[1]))
        for t in range(t0 + 1, num_t):
            #metx_all = []
            #mety_all = []
            #for pa in range(num_pa):
            #    metx, mety = compute_mets(plon[t0: t, pa], plat[t0: t, pa])
            #    metx_all.append(metx)
            #    mety_all.append(mety)
            metx_all, mety_all = compute_mets(plon[t0: t, :], plat[t0: t, :])
            metx_2d[t - t0, :, :] = numpy.array(metx_all).reshape(shape_grid)
            mety_2d[t - t0, :, :] = numpy.array(mety_all).reshape(shape_grid)
    else:
        metx_2d = numpy.empty(shape_grid)
        mety_2d = numpy.empty(shape_grid)
    if 'Mezic' in p.diagnostic:
        logger.info('compute Mezic')
        xt0, yt0 = compute_mezic(plon[:, :], plat[:, :])
        xt0_2d = numpy.zeros((numpy.shape(xt0)[0], shape_grid[0],
                             shape_grid[1]))
        yt0_2d = numpy.zeros((numpy.shape(yt0)[0], shape_grid[0],
                             shape_grid[1]))
        mezic_strain = numpy.zeros((numpy.shape(yt0)[0], shape_grid[0],
                                   shape_grid[1]))
        for t in range(numpy.shape(lavd)[0]):
            xt0_2d[t, :, :] = numpy.array(xt0[t, :]).reshape()
            yt0_2d[t, :, :] = numpy.array(yt0[t, :]).reshape()
            # mezic_strain[t, :, :] = det(
    #    mezic_all.append(mezic_strain)
    if 'Mezic' in p.diagnostic:
        mezic_2d = numpy.array(mezic_all).reshape(shape_grid)
    else:
        mezic_2d = numpy.empty(shape_grid)

    if 'LAVD' in p.diagnostic:
        logger.info('compute LAVD')
        p.save_RV = True
        logger.info('load vorticity')
        VEL = mod_io.read_velocity(p)
        abox = (p.parameter_grid[0], p.parameter_grid[1], p.parameter_grid[3],
                p.parameter_grid[4])
        index_lon = numpy.where((VEL.lon >= abox[0]) & (VEL.lon <= abox[1]))[0]
        index_lat = numpy.where((VEL.lat >= abox[2]) & (VEL.lat <= abox[3]))[0]
        vort_small = VEL.RV[:, numpy.min(index_lat): numpy.max(index_lat) + 1,
                            numpy.min(index_lon): numpy.max(index_lon) + 1]
        mean_vort = numpy.mean(numpy.mean(vort_small, axis=2), axis=1)
        logger.info('compute lavd')
        try:
            lavd, lavd_t = compute_lavd(p, plon, plat, ptime, pvort, mean_vort,
                                    t0=t0)
        except:
            import pdb ; pdb.set_trace()
        lavd_2d = numpy.zeros((numpy.shape(lavd)[0], shape_grid[0],
                               shape_grid[1]))
        for t in range(numpy.shape(lavd)[0]):
            lavd_2d[t, :, :] = lavd[t, :].reshape(shape_grid)
    else:
        lavd_2d = numpy.empty(shape_grid)
    time = numpy.arange(t0, (numpy.shape(lavd_2d)[0] + 1) * p.adv_time_step-1,
                        p.adv_time_step)
    print(numpy.shape(time), numpy.shape(metx_2d), numpy.shape(lavd_2d))
    logger.info('Write data')
    data={}
    data['lon'] = grid.lon
    data['lat'] = grid.lat
    data['time'] = time
    description = 'lagrangian diagnostics'
    mod_io.write_diagnostic_2d(p, data, description=description, METX=metx_2d,
                               METY=mety_2d, MEZIC=mezic_2d, LAVD=lavd_2d,
                               time=time)
    logger.info(f'Stop time {datetime.datetime.now()}')
