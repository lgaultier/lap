import datetime
import os
import lap.read_write_utils as rw_data


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
    for t in range(ntime -t0):
        mvort_t = pvort[t, :] - mean_vort[t]
        mvort_t0 = pvort[t0, :] - mean_vort[t0]
        lavd[t, :] = mvort_t - mvort_t0

     

def lagragian_diag(p)
    dict_var = rw_data.read_trajectory_1d(input_file):
