import numpy
from scipy.stats import norm
import math
import lap.const as const
import lap.mod_tools as mod_tools


def init_random_walk(sizeadvection, time_step, scale):
    ''' Intialize random coefficient if a random walk scheme is used to
    simulate diffusion. '''
    r = numpy.zeros((3, int(sizeadvection / time_step) + 1))
    rt = numpy.zeros((1, int(sizeadvection) + 1))
    r = norm.rvs(size=r.shape, scale=scale)
    # TODO remove rt??
    rt = norm.rvs(size=rt.shape, scale=scale)
    return r, rt


def init_velocity(VEL, lonpa, latpa, su, sv):
    '''Initialize particule position in velocity matrix.'''
    iu = numpy.argmin(abs(VEL.Vlatu - latpa))
    iv = numpy.argmin(abs(VEL.Vlatv - latpa))
    ju = numpy.argmin(abs(VEL.Vlonu - lonpa))
    jv = numpy.argmin(abs(VEL.Vlonv - lonpa))
    if (iv + 1) >= sv[0]:
        dVlatv = VEL.Vlatv[iv] - VEL.Vlatv[iv - 1]
    else:
        dVlatv = VEL.Vlatv[iv + 1] - VEL.Vlatv[iv]
    if (iu + 1) >= su[0]:
        dVlatu = VEL.Vlatu[iu] - VEL.Vlatu[iu - 1]
    else:
        dVlatu = VEL.Vlatu[iu + 1] - VEL.Vlatu[iu]
    if (ju + 1) >= su[1]:
        dVlonu = VEL.Vlonu[ju] - VEL.Vlonu[ju - 1]
    else:
        dVlonu = VEL.Vlonu[ju + 1] - VEL.Vlonu[ju]
    if (jv + 1) >= sv[1]:
        dVlonv = VEL.Vlonv[jv] - VEL.Vlonv[jv - 1]
    else:
        dVlonv = VEL.Vlonv[jv + 1] - VEL.Vlonv[jv]
    # index_vel = (iu, ju, iv, jv)
    dvcoord = (dVlatu, dVlatv, dVlonu, dVlonv)
    return (iu, ju), (iv, jv), dvcoord


def interpol_intime(arr, t, index_vel, interp_dt, su):
    '''Extract 4 neighbors and interpolate in time between two time steps'''
    (iu, ju) = index_vel
    result = numpy.zeros((2, 2))
    slice_t = slice(int(t), int(t+2))
    result[0, 0] = mod_tools.lin_1Dinterp(arr[slice_t, iu, ju], interp_dt)
    result[0, 1] = mod_tools.lin_1Dinterp(arr[slice_t, iu, min(ju + 1,
                                              su[1] - 1)], interp_dt)
    result[1, 0] = mod_tools.lin_1Dinterp(arr[slice_t, min(iu + 1, su[0] - 1),
                                          ju], interp_dt)
    result[1, 1] = mod_tools.lin_1Dinterp(arr[slice_t, min(iu + 1, su[0] - 1),
                                          min(ju + 1, su[1] - 1)], interp_dt)
    return result


def no_interpol_intime(arr, index_vel, su):
    '''Extract 4 neighbors, no interpolation in time (used for
    stationary fields).'''
    (iu, ju) = index_vel
    result = numpy.zeros((2, 2))
    result[0, 0] = arr[0, iu, ju]
    result[0, 1] = arr[0, iu, min(ju + 1, su[1] - 1)]
    result[1, 0] = arr[0, min(iu + 1, su[0] - 1), ju]
    result[1, 1] = arr[0, min(iu + 1, su[0] - 1), min(ju + 1, su[1] - 1)]
    return result


def dist_topoints(lon, lat, lonpa, latpa, dvcoord, index, su):
    '''Interpolate linearly the velocity between two grid points'''
    (iu, ju) = index
    (dVlon, dVlat) = dvcoord
    rlon = (lonpa - lon[ju]) / (dVlon)
    rlat = (latpa - lat[iu]) / (dVlat)
    while rlat > 1. and iu < (su[0] - 1):
        iu = min(iu + 1, su[0] - 1)
        if (iu + 1) >= su[0]:
            dVlat = lat[iu] - lat[iu - 1]
        else:
            dVlat = lat[iu + 1] - lat[iu]
        rlat = (latpa - lat[iu]) / (dVlat)
    while rlat < 0. and iu > 0:
        iu = max(iu - 1, 0)
        dVlat = lat[iu + 1] - lat[iu]
        rlat = (latpa - lat[iu]) / (dVlat)
    while rlon > 1. and ju < (su[1] - 1):
        ju = min(ju + 1, su[1] - 1)
        if (ju + 1) >= su[1]:
            dVlon = lon[ju] - lon[ju - 1]
        else:
            dVlon = lon[ju + 1] - lon[ju]
        rlon = (lonpa - lon[ju]) / (dVlon)
    while rlon < 0 and ju > 0:
        ju = max(ju - 1, 0)
        dVlon = lon[ju + 1] - lon[ju]
        rlon = (lonpa - lon[ju]) / (dVlon)
    return rlon, rlat, iu, ju, dVlon, dVlat


def find_indice_tracer(listGr, newlon, newlat, timetmp, num_pa):
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


def init_full_traj(p, s0, s1):
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


def advection_pa_timestep(p, lonpa, latpa, t, dt, mask, rk, VEL, vcoord, dv,
                          svel, sizeadvection):
    dVlonu, dVlatu, dVlonv, dVlatv = dv
    iu, ju, iv, jv = vcoord
    su, sv = svel
    # Temporal interpolation  for velocity
    interp_dt = dt / p.vel_step
    if p.stationary is False:
        VEL.ut = interpol_intime(VEL.u, t, [iu, ju], interp_dt, su)
        VEL.vt = interpol_intime(VEL.v, t, [iv, jv], interp_dt, sv)
    else:
        VEL.ut = no_interpol_intime(VEL.u, [iu, ju], su)
        VEL.vt = no_interpol_intime(VEL.v, [iv, jv], sv)
    # 2D Spatial interpolation for velocity
    dist = dist_topoints(VEL.Vlonu, VEL.Vlatu, lonpa, latpa,
                         [dVlonu, dVlatu], [iu, ju], su)
    rlonu, rlatu, iu, ju, dVlonu, dVlatu = dist
    dist = dist_topoints(VEL.Vlonv, VEL.Vlatv, lonpa, latpa,
                         [dVlonv, dVlatv], [iv, jv], sv)
    rlonv, rlatv, iv, jv, dVlonv, dVlatv = dist
    dlondt = mod_tools.lin_2Dinterp(VEL.ut, rlonu, rlatu)
    dlatdt = mod_tools.lin_2Dinterp(VEL.vt, rlonv, rlatv)
    # Set velocity to 0 if particle is outside domain
    if (rlonu < 0 or rlonu > 1 or rlatu < 0 or rlatu > 1
          or rlonv < 0 or rlonv > 1 or rlatv < 0 or rlatv > 1):
        dlondt = 0
        dlatdt = 0
        mask = 1
    rcoord = [rlonu, rlatu, rlonv, rlatv]
    vcoord = [iu, ju, iv, jv]
    dvcoord= [dVlonu, dVlatu, dVlonv, dVlatv]
    # Propagate position of particle with velocity
    deltat = (p.adv_time_step * const.day2sec) # * p.tadvection
              #/ float(sizeadvection))
    transport = dlondt * deltat
    turbulence = p.B * rk + p.sigma * rk * deltat
    lonpa = lonpa + transport + turbulence[0]
    transport = dlatdt * deltat
    latpa = latpa + transport + turbulence[1]
    return rcoord, vcoord, dvcoord, lonpa, latpa, mask


def advection(part, VEL, p, i0, i1, listGr, grid, rank=0, size=1, AMSR=None):
    # # Initialize listGrid and step
    if listGr is None:
        listGr = [grid, ]
        grid.dlon = 10.
        grid.dlat = 10.
    # # Initialize empty matrices
    sizeadvection = int(abs(p.tadvection) / p.output_step)
    shape_lr = (sizeadvection + 2, i1 - i0)
    lon_lr = numpy.empty(shape_lr)
    lat_lr = numpy.empty(shape_lr)
    mask_lr = numpy.ones(shape_lr, dtype=bool)
    for Trac in listGr:
        Trac.newi = numpy.zeros((2, shape_lr[0], shape_lr[1]))
        Trac.newj = numpy.zeros((2, shape_lr[0], shape_lr[1]))
    su = (numpy.shape(VEL.Vlatu)[0], numpy.shape(VEL.Vlonu)[0])
    sv = (numpy.shape(VEL.Vlatv)[0], numpy.shape(VEL.Vlonv)[0])
    # # Loop on particles
    for pa in part:
        # # - Initialize final variables
        lonpa = grid.lon1d[pa]
        latpa = grid.lat1d[pa]
        if p.save_traj is True:
            vlonpa = [lonpa, ]
            vlatpa = [latpa, ]
            vtime = [p.first_day, ]
            vu = [0, ]
            vh = [0, ]
            vv = [0, ]
            vS = [0, ]
            vOW = [0, ]
            vRV = [0, ]
            vmask = [1, ]
        lon_lr[0, pa - i0] = lonpa
        lat_lr[0, pa - i0] = latpa
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

        # # - Compute trajectory and tracer value if tracer is not NAN
        if not math.isnan(grid.mask1d[pa - i0]):
            # # - Compute initial location and grid size
            # Define random_walk
            if p.scale is not None:
                r, rt = init_random_walk(sizeadvection, p.adv_time_step,
                                         p.scale)
            init = init_velocity(VEL, lonpa, latpa, su, sv)
            (iu, ju), (iv, jv), dvcoord = init
            (dVlatu, dVlatv, dVlonu, dVlonv) = dvcoord
            vcoord = (iu, ju, iv, jv)
            if p.save_S or p.save_OW:
                Stmp = math.sqrt(VEL.Sn[0, iv, ju]**2 + VEL.Ss[0, iu, jv]**2)
            if p.save_RV or p.save_OW:
                RVtmp = VEL.RV[0, iu, jv]
            if p.save_OW:
                OWtmp = Stmp**2 - RVtmp**2

            # # - Loop on the number of advection days
            # # Change the output step ?
            mask = 0
            for t in numpy.arange(0, int(abs(p.tadvection) / p.output_step + 1), p.adv_time_step):
                dt = t - t%p.vel_step
                k = int(t)
                ind_t = + int(t)
                if p.stationary:
                    ind_t = 0
                #while dt < p.output_step:
                rk = r[:, k]
                advect = advection_pa_timestep(p, lonpa, latpa, t, dt,
                                               mask, rk, VEL, vcoord,
                                               dvcoord, (su, sv),
                                               sizeadvection)
                rcoord, vcoord, dvcoord, lonpa, latpa, mask = advect
                rlonu, rlatu, rlonv, rlatv = rcoord
                iu, ju, iv, jv = vcoord
                # 2D interpolation of physical variable
                # TODO handle enpty or 0d slice
                slice_iu = slice(iu, min(iu + 2, su[0] - 1))
                #if len(slice_iu) < 2:
                #    slice_iu = slice(su[0] - 2, su[0])
                slice_iv = slice(iv, min(iv + 2, sv[0] - 1))
                #if len(slice_iv) < 2:
                #    slice_iv = slice(sv[0] - 2, sv[0])
                slice_ju = slice(ju, min(ju + 2, su[1] - 1))
                #if len(slice_ju) < 2:
                #    slice_ju = slice(su[1] - 2, su[1])
                slice_jv = slice(jv, min(jv + 2, sv[1] - 1))
                #if len(slice_jv) < 2:
                #    slice_jv = slice(sv[1] - 2, sv[1])
                if p.save_U is True:
                    try:
                        ums = mod_tools.lin_2Dinterp(VEL.us[ind_t, slice_iu,
                                                 slice_ju], rlonu, rlatu)
                    except: import pdb ; pdb.set_trace()
                if p.save_V is True:
                    vms = mod_tools.lin_2Dinterp(VEL.vs[ind_t, slice_iv,
                                                 slice_jv], rlonv, rlatv)
                if p.stationary is False:
                    VEL.ht = interpol_intime(VEL.h, ind_t, (iu, jv),
                                             dt / p.vel_step, (su[0],
                                             sv[1]))
                else:
                    VEL.ht = no_interpol_intime(VEL.h, (iu, jv), (su[0],
                                                sv[1]))
                H = mod_tools.lin_2Dinterp(VEL.ht, rlonv, rlatu)
                if p.save_S is True or p.save_OW is True:
                    Sn = mod_tools.lin_2Dinterp(VEL.Sn[ind_t, slice_iv,
                                                slice_ju], rlonu, rlatv)
                    Ss = mod_tools.lin_2Dinterp(VEL.Ss[ind_t, slice_iu,
                                                slice_jv], rlonv, rlatu)
                    Stmp = numpy.sqrt((Sn**2 + Ss**2))
                if p.save_RV is True or p.save_OW is True:
                    RVtmp = mod_tools.lin_2Dinterp(VEL.RV[ind_t, slice_iu,
                                                slice_jv], rlonv, rlatu)
                if p.save_OW is True:
                    OWtmp = Stmp**2 - RVtmp**2

                # Store coordinates and physical variables at high
                # temporal resolution
                if p.save_traj is True:
                    vlonpa.append(lonpa)
                    vlatpa.append(latpa)
                    time = (p.first_day + (t * p.output_step)
                            * p.tadvection / float(sizeadvection))
                    vtime.append(time)
                    vu.append(ums)
                    vv.append(vms)
                    vh.append(H)
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
                #dt += p.adv_time_step
                #k += 1
                # -- Store new longitude and new latitude at each output_step
                if (t % p.output_step + dt > 1):
                    lon_lr[int(t) + 1, pa - i0] = lonpa
                    lat_lr[int(t) + 1, pa - i0] = latpa
                    mask_lr[int(t) + 1, pa - i0] = mask
                    timetmp = int(t) + 1
                    if listGr is not None:
                        find_indice_tracer(listGr, lon_lr[timetmp, pa - i0],
                                           lat_lr[timetmp, pa - i0],
                                           timetmp, pa - i0)
        else:
            # If particle does not exist, set all values to fill_value
            timetmp = 1
            lon_lr[:, pa - i0] = p.fill_value
            lat_lr[:, pa - i0] = p.fill_value
            mask_lr[:, pa - i0] = 1
            if listGr is not None:
                for Trac in listGr:
                    Trac.newi[0, :, pa - i0] = p.fill_value
                    Trac.newj[0, :, pa - i0] = p.fill_value

        if pa % 200 == 0:
            perc = float(pa - i0) / float(numpy.shape(part)[0])
            mod_tools.update_progress(perc, f'{pa} particles', f'{rank} node')
        if p.save_traj:
            if pa == i0:
                init = init_full_traj(p, numpy.shape(vlonpa)[0], i1 - i0)

                lon_hr, lat_hr, mask_hr, S_hr, RV_hr, OW_hr, u_hr, v_hr, h_hr = init
            if (vlonpa[0] != vlonpa[1] or vlatpa[0] != vlatpa[1]
                 or (not numpy.array(vmask).all())):
                lon_hr[:, pa - i0] = numpy.transpose(vlonpa)
                lat_hr[:, pa - i0] = numpy.transpose(vlatpa)
                mask_hr[:, pa - i0] = numpy.transpose(vmask)
            else:
                lon_hr[:, pa - i0] = p.fill_value
                lat_hr[:, pa - i0] = p.fill_value
                mask_hr[:, pa - i0] = 1
            if p.save_U is True:
                u_hr[:, pa - i0] = numpy.transpose(vu)
                v_hr[:, pa - i0] = numpy.transpose(vv)
                h_hr[:, pa - i0] = numpy.transpose(vh)
            if p.save_S is True:
                S_hr[:, pa - i0] = numpy.transpose(vS)
            if p.save_RV is True:
                RV_hr[:, pa - i0] = numpy.transpose(vRV)
            if p.save_OW is True:
                OW_hr[:, pa - i0] = numpy.transpose(vOW)
            # mask_hr[:, pa - i0] = numpy.transpose(vmask)
    dict_output = {}
    mod_tools.update_progress(1, f'{len(part)} particles', f'{rank} node')
    if p.save_traj is True:
        dict_output['lon_hr'] = lon_hr
        dict_output['lat_hr'] = lat_hr
        dict_output['S_hr'] = S_hr
        dict_output['u_hr'] = u_hr
        dict_output['v_hr'] = v_hr
        dict_output['h_hr'] = h_hr
        dict_output['OW_hr'] = OW_hr
        dict_output['RV_hr'] = RV_hr
        dict_output['lon_lr'] = lon_lr
        dict_output['lat_lr'] = lat_lr
        dict_output['time_hr'] = vtime
        dict_output['mask_hr'] = mask_hr
    dict_output['lon_lr'] = lon_lr
    dict_output['lat_lr'] = lat_lr
    dict_output['mask_lr'] = mask_lr

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
                rlat = grid.ri[i, j]
                rlon = grid.rj[i, j]
                if rlon != 0:
                    signlon = int(rlon / abs(rlon))
                else:
                    signlon = 0
                if rlat != 0:
                    signlat = int(rlat / abs(rlat))
                else:
                    signlat = 0
                if ((itra > 0) and (jtra > 0) and (itra < shape_tra[0] - 1)
                     and (jtra < shape_tra[1] - 1)):
                    if ((numpy.isnan(var[itra, jtra]))
                            or ((var[itra, jtra]) < -10.)):
                            # or numpy.isnan(Tr.var[int(Tr.ii[i,j+signlon])
                            # , int(Tr.ij[i,j+signlon])])
                            # or numpy.isnan(Tr.var[int(Tr.ii[i+signlat,j])
                            # , int(Tr.ij[i+signlat,j])]):
                        grid.var2[t, i, j] = numpy.nan
                    else:
                        if signlat > 0:
                            slice_i = slice(i, i + signlat)
                        else:
                            slice_i = slice(i + signlat, i)
                        if signlon > 0:
                            slice_j = slice(j, j + signlon)
                        else:
                            slice_j = slice(j + signlon, j)

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
                    rlat = grid.ri[i, j]
                    rlon = grid.rj[i, j]
                    if rlon != 0:
                        signlon = rlon / abs(rlon)
                    else:
                        signlon = 0
                    if rlat != 0:
                        signlat = rlat / abs(rlat)
                    else:
                        signlat = 0
                    if ((itra > 0) and (jtra > 0) and (itra < shape_tra[0] - 1)
                          and (jtra < shape_tra[1] - 1)):
                        if ((numpy.isnan(Tr.var[itra, jtra]))
                                or ((Tr.var[itra, jtra]) < -10.)):
                                # or numpy.isnan(Tr.var[int(Tr.ii[i,j+signlon])
                                # , int(Tr.ij[i,j+signlon])])
                                # or numpy.isnan(Tr.var[int(Tr.ii[i+signlat,j])
                                # , int(Tr.ij[i+signlat,j])]):
                            grid.var2[t, i, j] = -999.
                            # Tr.var[int(Tr.ii[i,j]),int(Tr.ij[i,j])]
                        else:
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
                    Tr.var2[t, i, j] = -999.
                if p.gamma and AMSR is not None and t:
                    grid.var2[t, i, j] = nudging_AMSR(AMSR, grid.lon[i, j],
                                                      grid.lat[i, j],
                                                      grid.var2[t, i, j], t, p)
                    # grid.var2[t,:,:]= grid.var2[t,i,j]
                    # numpy.where(grid.mask, numpy.nan, grid.var2[t,:,:])
                    # ]=numpy.nan
    grid.var2[numpy.isnan(grid.var2)] = p.fill_value
    grid.var2[abs(grid.var2) > 50.] = p.fill_value
    grid.var2 = numpy.ma.array(grid.var2, mask=(grid.var2 == p.fill_value))
    # grid.mask)
    grid.var = grid.var2
    # Tr.var2[0,:,:]=numpy.ma.array(Tr.var2[0,:,:], mask=Tr.mask)
    # Tr.var2[1,:,:]=numpy.ma.array(Tr.var2[1,:,:], mask=Tr.mask)
    # Tr.var=numpy.ma.array(Tr.var, mask=numpy.isnan(Tr.var))
    # Tr.var2=numpy.ma.array(Tr.var2, mask=numpy.isnan(Tr.var2))

# def interp(Tr, grid, ):
# interp2d


def reordering1d(p, listTr, listGr):
    if listGr:
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[p.listnum[i]]
            for time in range(nt):
                realtime = (p.first_day + time * numpy.sign(p.tadvection))
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


def reordering1dmpi(p, listTr, listGr):
    tra = None
    if listGr:
        n2, nt, npa = numpy.shape(listGr[0].newi)
        nlist = len(listTr)
        tra = numpy.zeros((nt, npa, nlist))
        for i in range(nlist):
            Tr = listTr[i]
            Gr = listGr[p.listnum[i]]
            for time in range(nt):
                realtime = (p.first_day + time * numpy.sign(p.tadvection))
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
