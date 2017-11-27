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
        lavd_tmp = numpy.sum(numpy.abs(pvort[tini: tf, :]
                             - mean_vort[tini: tf, numpy.newaxis]), axis=0)
        lavd_t0 = numpy.abs(pvort[tinipvort[t0, :] - mean_vort[t0]) / 2
        lavd_t = numpy.abs(pvort[tinipvort[tf + 1, :] - mean_vort[tf + 1]) / 2
        lavd_tmp = lavd_tmp + lavd_t0 + lavd_t
        lavd_time[t] = ptime[tf + 1] - ptime[t0]
        lavd[t, :] = lavd_tmp
    return lavd, lavd_time


def lagragian_diag(p):
    logger.info(f'Start time {datetime.datetime.now()}')
    list_var = ['lon_hr', 'lat_hr', 'time_hr', 'u_hr', 'Vorticity']
    dict_var = read_utils.read_trajectory(input_file, list_var)
    lon = dict_var['lon_hr']
    lat = dict_var['lat_hr']
    vort = dict_var['Vorticity']
    time_part = dict_var['time_hr']
    num_t, num_pa = numpy.shape(lon)
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
    if 'LAVD' in p.diagnostic
        lavd, lavd_t = compute_lavd(lon, lat, ptime, pvort, mean_vort)
    logger.info(f'Stop time {datetime.datetime.now()}')
