import numpy
from typing import Optional, Tuple
from scipy.stats import norm
import math
import lap.const as const
import lap.utils.tools as tools
import datetime


def init_random_walk(sizeadvection: int, time_step: float, scale: float
                     ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    ''' Intialize random coefficient if a random walk scheme is used to
    simulate diffusion. '''
    r = numpy.zeros((3, int(sizeadvection / time_step) + 1))
    rt = numpy.zeros((1, int(sizeadvection) + 1))
    r = norm.rvs(size=r.shape, scale=scale)
    # TODO remove rt??
    rt = norm.rvs(size=rt.shape, scale=scale)
    return r, rt


def find_indice_tracer(listGr: list, newlon: float, newlat: float,
                       timetmp: int, num_pa: int) -> None:
    '''find indices corresponding to position of particle at time t'''
    for Trac in listGr:
        newi0 = Trac.newi0
        newj0 = Trac.newj0
        dlon = (Trac.lon[newi0, newj0] - newlon)
        ind = 0
        while True:
            if ((abs(dlon) <= Trac.dlon / 2.) or (newj0 <= 0)
                    or (newj0 >= (numpy.shape(Trac.lon)[1] - 1))):
                break
            if dlon < 0.:
                newj0 = int(min(newj0 + 1, numpy.shape(Trac.lon)[1] - 1))
            else:
                newj0 = int(max(newj0-1, 0))
            dlon = (Trac.lon[newi0, newj0] - newlon)
            ind += 1
            # TODO find less costly alternative
            if ind > 1000:
                break
        ind = 0
        dlat = Trac.lat[newi0, newj0] - newlat
        while True:
            if ((abs(dlat) <= Trac.dlat / 2.) or (newi0 <= 0)
                    or (newi0 >= (numpy.shape(Trac.lon)[0] - 1))):
                break
            if dlat < 0.:
                newi0 = int(min(newi0 + 1, numpy.shape(Trac.lon)[0] - 1))
            else:
                newi0 = int(max(newi0 - 1, 0))
            dlat = (Trac.lat[newi0, newj0] - newlat)
            ind += 1
            if ind > 1000:
                break
        Trac.newi[0, timetmp, num_pa] = newi0
        Trac.newj[0, timetmp, num_pa] = newj0
        Trac.newj[1, timetmp, num_pa] = dlon
        Trac.newi[1, timetmp, num_pa] = dlat
    return None


def init_full_traj(p, s0: int, s1: int):
    shape = (s0, s1)
    lon_hr = numpy.zeros(shape)
    lat_hr = numpy.zeros(shape)
    mask_hr = numpy.zeros(shape)
    if p.save_S:
        S_hr = numpy.zeros(shape)
    else:
        S_hr = 0
    if p.save_RV:
        RV_hr = numpy.zeros(shape)
    else:
        RV_hr = 0
    if p.save_OW:
        OW_hr = numpy.zeros(shape)
    else:
        OW_hr = 0
    if p.save_U:
        u_hr = numpy.zeros(shape)
    else:
        u_hr = 0
    if p.save_U:
        v_hr = numpy.zeros(shape)
    else:
        v_hr = 0
    if p.save_U:
        h_hr = numpy.zeros(shape)
    else:
        h_hr = 0
    return lon_hr, lat_hr, mask_hr, S_hr, RV_hr, OW_hr, u_hr, v_hr, h_hr


def advection_pa_timestep_np(p, lonpa: float, latpa: float, dt: float,
                             interp_dt:int,
                             mask: bool, rk: float, _interp_u: list,
                             _interp_v: list
                             ) -> Tuple[float, float, float, float, float]:

    # TODO boundary condition
    # Temporal interpolation  for velocity
    tadvection = (p.last_date - p.first_date).total_seconds() / 86400
    if type(_interp_u) is list and len(_interp_u) > 1:
        if p.stationary is True:
            _interp_ut = _interp_u[interp_dt](lonpa, latpa)
            _interp_vt = _interp_v[interp_dt](lonpa, latpa)
        else:
            _interp_ut = (_interp_u[interp_dt](lonpa, latpa) * (1 - dt)
                          + _interp_u[interp_dt+1](lonpa, latpa) * dt)
            _interp_vt = (_interp_v[interp_dt](lonpa, latpa) * (1 - dt)
                          + _interp_v[interp_dt+1](lonpa, latpa) * dt)
    else:
        _interp_ut = _interp_u[0](lonpa, latpa)
        _interp_vt = _interp_v[0](lonpa, latpa)
    dlondt = numpy.sign(tadvection) * _interp_ut
    dlatdt = numpy.sign(tadvection) * _interp_vt
    if dlondt == 0 or dlatdt == 0:
        mask = True
    # TODO
    # Set velocity to 0 if particle is outside domain
    # if (rlonu < 0 or rlonu > 1 or rlatu < 0 or rlatu > 1
    #      or rlonv < 0 or rlonv > 1 or rlatv < 0 or rlatv > 1):
    #    dlondt = 0
    #    dlatdt = 0
    #    mask = 1
    # Propagate position of particle with velocity
    deltat = (p.adv_time_step * const.day2sec)  # * p.tadvection
    #  / float(sizeadvection))
    # compute pure advection transport
    transportu = dlondt * deltat
    transportv = dlatdt * deltat
    # Bera vera motion of a particule with a weight different from sea water
    fo = numpy.cos(numpy.deg2rad(latpa)) * 2 * const.omega
    tau = 2 * p.radius_part**2 / (9 * const.visc * p.weight_part)
    inertial_partu = - tau * (p.weight_part - 1) * fo * dlatdt
    inertial_partv = tau * (p.weight_part - 1) * fo * dlondt
    # compute turbulence if there is diffusion
    turbulence = p.B * rk + p.sigma * rk * deltat
    lonpa = lonpa + transportu + inertial_partu + turbulence[0]
    latpa = latpa + transportv + inertial_partv + turbulence[1]
    return lonpa, latpa, mask, dlondt, dlatdt


def advection(part: numpy.ndarray, dic: dict, p, i0: int, i1: int,
              listGr: list, grid, rank: Optional[int] = 0,
              size: Optional[int] = 1, AMSR=None):
    # # Initialize listGrid and step
    if listGr is None:
        listGr = []
        grid.dlon = 10.
        grid.dlat = 10.
    # # Initialize empty matrices
    tadvection = (p.last_date - p.first_date).total_seconds() / 86400
    sizeadvection = int(abs(tadvection) / p.output_step)
    shape_lr = (sizeadvection + 1, i1 - i0)
    shape_hr = int(numpy.ceil(sizeadvection / p.adv_time_step) + 1)
    lon_lr = numpy.empty(shape_lr)
    lat_lr = numpy.empty(shape_lr)
    time_lr = numpy.empty((shape_lr[0]))
    mask_lr = numpy.full(shape_lr, True, dtype=bool)
    for Trac in listGr:
        Trac.newi = numpy.zeros((2, shape_lr[0], shape_lr[1]))
        Trac.newj = numpy.zeros((2, shape_lr[0], shape_lr[1]))
    # # Loop on particles
    first_day = (p.first_date - p.reference).total_seconds() / 86400.
    for pa in part:
        # # - Initialize final variables
        lonpa = + grid.lon1d[pa]
        latpa = + grid.lat1d[pa]
        lon_lr[0, pa - i0] = + lonpa
        lat_lr[0, pa - i0] = + latpa
        time_lr[0] = + 0
        mask_lr[0, pa - i0] = grid.mask1d[pa - i0]
        if p.save_traj is True:
            vlonpa = [lonpa, ]
            vlatpa = [latpa, ]
            vu = [0, ]
            vh = [0, ]
            vv = [0, ]
            vS = [0, ]
            vOW = [0, ]
            vRV = [0, ]
            vmask = [grid.mask1d[pa - i0], ]
        for Trac in listGr:
            Trac.newj0 = numpy.argmin(abs(Trac.lon - lonpa), axis=1)[0]
            Trac.newi0 = numpy.argmin(abs(Trac.lat - latpa), axis=0)[0]
        # # if non regular grid
        # newjlon=numpy.argmin(numpy.argmin(Trac.lon-lonpa, axis=1))
        # newilon=numpy.argmin(Trac.lon - lonpa, axis=1)[newjlon]
        # newilat=numpy.argmin(numpy.argmin(Trac.lat-latpa, axis=0))
        # newjlat=numpy.argmin(Trac.lat-latpa, axis=0)[newilat]
        # newi0=i_ini ; newj0=j_ini
        # newi[0,rank,0,i_ini, i_ini]=newi0
        # newj[0,rank,0,j_ini, j_ini]=newj0
        for Trac in listGr:
            Trac.newi[0, 0, pa - i0] = Trac.newi0
            Trac.newj[0, 0, pa - i0] = Trac.newj0

        tstop = int(abs(tadvection) / p.output_step) + p.adv_time_step
        if not grid.mask1d[pa - i0]:
            vtime = [0, ]
        # # - Compute trajectory and tracer value if tracer is not NAN
        # if not math.isnan(grid.mask1d[pa - i0]):
            # # - Compute initial location and grid size
            # Define random_walk
            if p.scale is not None:
                r, rt = init_random_walk(sizeadvection, p.adv_time_step,
                                         p.scale)
            if p.save_S or p.save_OW:
                if p.stationary is True:
                    Stmp = math.sqrt(dic['sn'](lonpa, latpa)**2
                                     + dic['ss'](lonpa, latpa)**2)
                else:
                    Stmp = math.sqrt(dic['sn'][0](lonpa, latpa)**2
                                     + dic['ss'][0](lonpa, latpa)**2)

            if p.save_RV or p.save_OW:
                if p.stationary is True:
                    RVtmp = dic['rv'](lonpa, latpa)
                else:
                    RVtmp = dic['rv'][0](lonpa, latpa)

            if p.save_OW:
                OWtmp = Stmp**2 - RVtmp**2

            # # - Loop on the number of advection days
            # # Change the output step ?
            mask = False
            tmod_lr = 1
            tout = 1
            if p.stationary:
                _diff = (numpy.datetime64(p.first_date) - dic['time'])
                ind_t = numpy.argmin(abs(_diff), out=None)
                dt = 0
            for t in numpy.arange(p.adv_time_step, tstop, p.adv_time_step):
                # dt = t % p.vel_step
                # Index for random walk
                k = int(t)
                # Index in velocity array, set to 0 if stationary
                if not p.stationary:
                    curdate = p.first_date + datetime.timedelta(seconds=t*86400)
                    _diff = (numpy.datetime64(curdate) - dic['time'])
                    ind_t = numpy.argmin(abs(_diff), out=None)
                    if dic['time'][ind_t] > numpy.datetime64(curdate) :
                        ind_t = max(0, ind_t - 1)
                    if ind_t > len(dic['time']) - 2:
                        ind_t = max(0, ind_t - 1)
                        #break
                    dt = ((numpy.datetime64(curdate) - dic['time'][ind_t])
                           / (dic['time'][ind_t + 1] - dic['time'][ind_t]))
                # while dt < p.output_step:
                rk = r[:, k]
                advect = advection_pa_timestep_np(p, lonpa, latpa, dt, ind_t, mask,
                                                  rk, dic['u'], dic['v'])
                lonpa, latpa, mask, dlondt, dlatdt = advect
                if p.stationary is True:
                    if p.save_U is True:
                        ums = dic['u'][0](lonpa, latpa)
                    if p.save_V is True:
                        vms = dic['v'][0](lonpa, latpa)
                    if 'h' in dic.keys():
                        H = dic['h'][0](lonpa, latpa)
                    if p.save_S is True or p.save_OW is True:
                        Sn = dic['sn'][0](lonpa, latpa)
                        Ss = dic['ss'][0](lonpa, latpa)
                    if p.save_RV is True or p.save_OW is True:
                        RVtmp = dic['rv'][0](lonpa, latpa)
                else:
                    if p.save_U is True:
                        ums = dic['ums'][ind_t](lonpa, latpa)
                    if p.save_V is True:
                        vms = dic['vms'][ind_t](lonpa, latpa)
                    # TODO temporal interpolation
                    if 'h' in dic.keys():
                        H = dic['h'][ind_t](lonpa, latpa)
                    if p.save_S is True or p.save_OW is True:
                        Sn = dic['sn'][ind_t](lonpa, latpa)
                        Ss = dic['ss'][ind_t](lonpa, latpa)
                    if p.save_RV is True or p.save_OW is True:
                        RVtmp = dic['rv'][ind_t](lonpa, latpa)
                if p.save_S is True or p.save_OW is True:
                    Stmp = numpy.sqrt((Sn**2 + Ss**2))
                if p.save_OW is True:
                    OWtmp = Stmp**2 - RVtmp**2

                # Store coordinates and physical variables at high
                # temporal resolution
                if p.save_traj is True:
                    vlonpa.append(lonpa)
                    vlatpa.append(latpa)
                    vtime.append(t)
                    vu.append(ums)
                    vv.append(vms)
                    #vh.append(H)
                    vmask.append(mask)
                    if p.save_S is True:
                        vS.append(Stmp)
                    else:
                        vS = 0
                    if p.save_RV is True:
                        vRV.append(RVtmp)
                    else:
                        vRV = 0
                    if p.save_OW is True:
                        vOW.append(OWtmp)
                    else:
                        vOW = 0
                # k += 1
                # -- Store new longitude and new latitude at each output_step
                if (t > tout):
                    lon_lr[tmod_lr, pa - i0] = lonpa
                    lat_lr[tmod_lr, pa - i0] = latpa
                    mask_lr[tmod_lr, pa - i0] = mask
                    time_lr[tmod_lr] = t

                    if listGr is not None:
                        find_indice_tracer(listGr, lon_lr[tmod_lr, pa - i0],
                                           lat_lr[tmod_lr, pa - i0],
                                           tmod_lr, pa - i0)
                    tmod_lr += 1
                    tout += p.output_step
        else:
            # If particle does not exist, set all values to fill_value
            timetmp = 1
            lon_lr[:, pa - i0] = [p.fill_value, ] * shape_lr[0]
            lat_lr[:, pa - i0] = [p.fill_value, ] * shape_lr[0]
            mask_lr[:, pa - i0] = [True, ] * shape_lr[0]
            #time_lr[:] =  [p.fill_value, ] * shape_lr[0]
            if listGr is not None:
                for Trac in listGr:
                    Trac.newi[0, :, pa - i0] = p.fill_value
                    Trac.newj[0, :, pa - i0] = p.fill_value

        if pa % 200 == 0:
            perc = float(pa - i0) / float(numpy.shape(part)[0])
            tools.update_progress(perc, f'{pa} particles', f'{rank} node')
        if p.save_traj:
            if pa == i0:
                #init = init_full_traj(p, numpy.shape(vlonpa)[0], i1 - i0)
                init = init_full_traj(p, shape_hr, i1 - i0)

                lon_hr, lat_hr, mask_hr, S_hr, RV_hr, OW_hr, u_hr, v_hr, h_hr = init
            #_cond = (vlonpa[0] != vlonpa[1] or vlatpa[0] != vlatpa[1]
            #         or (not numpy.array(vmask).all()))
            _cond =  (not numpy.array(vmask).all())
            if _cond:
                lon_hr[:, pa - i0] = numpy.transpose(vlonpa)
                lat_hr[:, pa - i0] = numpy.transpose(vlatpa)
                mask_hr[:, pa - i0] = numpy.transpose(vmask)
            else:
                lon_hr[:, pa - i0] = [p.fill_value, ] * shape_hr
                lat_hr[:, pa - i0] = [p.fill_value, ] * shape_hr
                mask_hr[:, pa - i0] = [True, ] * shape_hr
            if p.save_U is True:
                u_hr[:, pa - i0] = numpy.transpose(vu)
                v_hr[:, pa - i0] = numpy.transpose(vv)
                #h_hr[:, pa - i0] = numpy.transpose(vh)
            if p.save_S is True:
                S_hr[:, pa - i0] = numpy.transpose(vS)
            if p.save_RV is True:
                RV_hr[:, pa - i0] = numpy.transpose(vRV)
            if p.save_OW is True:
                OW_hr[:, pa - i0] = numpy.transpose(vOW)
            # mask_hr[:, pa - i0] = numpy.transpose(vmask)
    dict_output = {}
    tools.update_progress(1, f'{len(part)} particles', f'{rank} node')
    if p.save_traj is True:
        dict_output['lon_hr'] = lon_hr
        dict_output['lat_hr'] = lat_hr
        dict_output['S_hr'] = S_hr
        dict_output['u_hr'] = u_hr
        dict_output['v_hr'] = v_hr
        dict_output['h_hr'] = h_hr
        dict_output['OW_hr'] = OW_hr
        dict_output['RV_hr'] = RV_hr
        dict_output['time_hr'] = numpy.arange(0, tstop, p.adv_time_step)
        dict_output['mask_hr'] = mask_hr
    dict_output['lon_lr'] = lon_lr
    dict_output['lat_lr'] = lat_lr
    dict_output['mask_lr'] = mask_lr
    dict_output['time_lr'] = time_lr

    return dict_output


# TO CHECK
def interpolate_tracer(p, shape_lon, shape_tra, grid, Tr, AMSR, t):
    '''2D linear interpolation of tracer for each particle'''
    var = Tr.var
    for i in range(shape_lon[0]):
        for j in range(shape_lon[1]):
            itra = int(grid.ii[i, j])
            jtra = int(grid.ij[i, j])
            if (itra < shape_tra[0]) and (jtra < shape_tra[1]):
                # rlat = grid.ri[i, j]
                # rlon = grid.rj[i, j]
                # if rlon != 0:
                #    signlon = int(rlon / abs(rlon))
                # else:
                #     signlon = 0
                # if rlat != 0:
                #     signlat = int(rlat / abs(rlat))
                # else:
                #     signlat = 0
                _cond_inside = ((itra > 0) and (jtra > 0)
                                and (itra < shape_tra[0] - 1)
                                and (jtra < shape_tra[1] - 1))
                if _cond_inside:
                    _cond_masked = ((numpy.isnan(var[itra, jtra]))
                                    or ((var[itra, jtra]) < -10.))
                    if _cond_masked:
                        # or numpy.isnan(Tr.var[int(Tr.ii[i,j+signlon])
                        # , int(Tr.ij[i,j+signlon])])
                        # or numpy.isnan(Tr.var[int(Tr.ii[i+signlat,j])
                        # , int(Tr.ij[i+signlat,j])]):
                        grid.var2[t, i, j] = numpy.nan
                    else:
                        # if signlat > 0:
                        #     slice_i = slice(i, i + signlat)
                        # else:
                        #     slice_i = slice(i + signlat, i)
                        # if signlon > 0:
                        #     slice_j = slice(j, j + signlon)
                        # else:
                        #     slice_j = slice(j + signlon, j)

                        # tra = int(grid.ii[slice_i, slice_j])
                        # a = var[itra: int(Tr.ii[i,j + signlon],
                        # int(Tr.ij[i,j])]
                        # interp = lin_2Dinterp(a, abs(rlon), abs(rlat)))
                        # grid.var2[t, i, j] = interp
                        # # TODO code that
                        # tmpvar1=(1-abs(rlon))*Tr.var[int(Tr.ii[i,j]),
                        #   int(Tr.ij[i,j])]+abs(rlon)*Tr.var[int(Tr.ii[i,j+
                        #   signlon]),int(Tr.ij[i,j+signlon])]
                        # tmpvar2=(1-abs(rlon))*
                        #   Tr.var[int(Tr.ii[i+signlat,j]),
                        #   int(Tr.ij[i+signlat,j])]+abs(rlon)*
                        #   Tr.var[int(Tr.ii[i+signlat,j+signlon]),
                        #   int(Tr.ij[i+signlat,j+signlon])]
                        # Tr.var2[i,j]=((1-abs(rlon))*
                        #   Tr.var[int(Tr.ii[i,j]),int(Tr.ij[i,j])]+
                        #   abs(rlon)*Tr.var[int(Tr.ii[i,j+signlon]),
                        #   int(Tr.ij[i,j+signlon])]+(1-abs(rlat))*
                        #   Tr.var[int(Tr.ii[i,j]),int(Tr.ij[i,j])]+
                        #   abs(rlat)*Tr.var[int(Tr.ii[i+signlat,j]),
                        #   int(Tr.ij[i+signlat,j])])/2.
                        # Tr.var2[t,i,j]=(1-abs(rlat))*tmpvar1+
                        #   abs(rlat)*tmpvar2
                        grid.var2[t, i, j] = Tr.var[itra, jtra]
                else:
                    grid.var2[t, i, j] = Tr.var[itra, jtra]
            else:
                Tr.var2[t, i, j] = p.fill_value
            if p.gamma and AMSR is not None and t:
                grid.var2[t, i, j] = nudging_AMSR(AMSR, grid.lon[i, j],
                                                  grid.lat[i, j],
                                                  grid.var2[t, i, j], t, p)
                # grid.var2[t,:,:]= grid.var2[t,i,j]
                # numpy.where(grid.mask, numpy.nan, grid.var2[t,:,:])
                # ]=numpy.nan


def reordering(Tr, grid, AMSR, p):
    sizeadvection = numpy.shape(grid.vtime)[0]
    shape_lon = numpy.shape(grid.lon)
    shape_tra = numpy.shape(Tr.var)
    shape = (sizeadvection + 1, shape_lon[0], shape_lon[1])
    grid.i = numpy.zeros(shape)
    grid.j = numpy.zeros(shape)
    grid.var2 = numpy.ones(shape)
    for t in range(sizeadvection+1):
        # Distance to indexes i, j
        grid.ri = grid.iit[1, t, :].reshape(shape_lon)
        grid.rj = grid.ijt[1, t, :].reshape(shape_lon)
        # indexes i, j
        grid.ii = grid.iit[0, t, :].reshape(shape_lon)
        grid.ij = grid.ijt[0, t, :].reshape(shape_lon)
        grid.ii.astype(int)
        grid.ij.astype(int)
        grid.i[t, :, :] = grid.ii[:, :]
        grid.j[t, :, :] = grid.ij[:, :]
        for i in range(shape_lon[0]):
            for j in range(shape_lon[1]):
                itra = int(grid.ii[i, j])
                jtra = int(grid.ij[i, j])
                if (itra < shape_tra[0]) and (jtra < shape_tra[1]):
                    _cond_inside = ((itra > 0) and (jtra > 0)
                                    and (itra < shape_tra[0] - 1)
                                    and (jtra < shape_tra[1] - 1))
                    if _cond_inside:
                        if ((numpy.isnan(Tr.var[itra, jtra]))
                                or ((Tr.var[itra, jtra]) < -10.)):
                            grid.var2[t, i, j] = -999.
                        else:
                            grid.var2[t, i, j] = Tr.var[itra, jtra]
                    else:
                        grid.var2[t, i, j] = Tr.var[itra, jtra]
                else:
                    Tr.var2[t, i, j] = -999.
                if p.gamma and AMSR is not None and t:
                    grid.var2[t, i, j] = nudging_AMSR(AMSR, grid.lon[i, j],
                                                      grid.lat[i, j],
                                                      grid.var2[t, i, j], t, p)
    grid.var2[numpy.isnan(grid.var2)] = p.fill_value
    grid.var2[abs(grid.var2) > 50.] = p.fill_value
    grid.var2 = numpy.ma.array(grid.var2, mask=(grid.var2 == p.fill_value))
    grid.var = grid.var2


def reordering1d(p, listTr: list, listGr: list):
    first_day = (p.first_date - p.reference).total_seconds() / 86400
    tadvection = (p.last_date - p.first_date).total_seconds() / 86400
    if listGr:
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[p.listnum[i]]
            for time in range(nt):
                realtime = (first_day + time * numpy.sign(tadvection))
                tratime = numpy.argmin(abs(Tr.time - realtime))
                for pa in range(npa):
                    tra[time, pa, i] = Tr.var[tratime,
                                              int(Gr.newi[0, time, pa]),
                                              int(Gr.newj[0, time, pa])]
            Tr.newvar = tra[:, :, i]
            Tr.newi = Gr.newi[0, :, :]
            Tr.newj = Gr.newj[0, :, :]
        return tra
    else:
        return None


def reordering1dmpi(p, listTr: list, listGr: list):
    tra = None
    first_day = (p.first_date - p.reference).total_seconds() / 86400
    tadvection = (p.last_date - p.first_date).total_seconds() / 86400
    if listGr:
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[p.listnum[i]]
            for time in range(nt):
                realtime = (first_day + time * numpy.sign(tadvection))
                tratime = numpy.argmin(abs(Tr.time - realtime))
                for pa in range(npa):
                    tra[time, pa, i] = Tr.var[tratime,
                                              int(Gr.newi[0, time, pa]),
                                              int(Gr.newj[0, time, pa])]
            Tr.newvarloc = tra[:, :, i]
            Tr.newiloc = Gr.newi[0, :, :]
            Tr.newjloc = Gr.newj[0, :, :]
    return tra


def nudging_AMSR(AMSR, lon, lat, tra, t, p):
    iamsr = numpy.argmin(abs(AMSR.latt - lat))
    jamsr = numpy.argmin(abs(AMSR.lont - lon))
    if not numpy.isnan(AMSR.tra[int(t * p.output_step), iamsr, jamsr]):
        tra = tra - p.gamma*(tra - AMSR.tra[int(t * p.output_step), iamsr,
                                            jamsr])
        # +p.tsigma*r[2,k] '''
    return tra
