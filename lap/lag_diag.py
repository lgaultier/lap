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


def compute_lavd(plon, plat, pvort, t0=0):
    ntime, npa = numpy.shape(plon)
    lavd = numpy.zeros((ntime - t0, npa))
    mean_vort = numpy.mean(pvort, axes=2)
    for t in range(ntime - t0):
        mvort_t = pvort[t, :] - mean_vort[t]
        mvort_t0 = pvort[t0, :] - mean_vort[t0]
        lavd[t, :] = mvort_t - mvort_t0


def lagragian_diag(p):
    logger.info(f'Start time {datetime.datetime.now()}')
    list_var = ['lon_hr', 'lat_hr', 'u_hr', 'RV_hr']
    dict_var = read_utils.read_trajectory(input_file, list_var)
    lon = dict_var['lon_hr']
    lat = dict_var['lat_hr']
    vort = dict_var['RV_hr']
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
    logger.info(f'Stop time {datetime.datetime.now()}')
