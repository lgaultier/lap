import numpy

def test_interpol_intime(dt):
    import lap.mod_advection as mod_advection
    array = numpy.zeros((2, 3, 3))
    array[0, :, :] = 1
    array[1, :, :] = 2
    slice_t = slice(0, 2)

    result = mod_advection.interpol_intime(array, 0, (0, 0), dt, (2,2), (2, 2))
    if (result == (dt + 1)).all():
        test = True
    else:
        test = False
    return test


def test_dist_topoints(loc):
    import lap.mod_advection as mod_advection
    lonpa = 5
    latpa = 5
    nstep = 2
    lon = numpy.arange(0, 10, nstep)
    lat = numpy.arange(0, 10, nstep)
    su = (len(lat), len(lon))
    dvcoord = (nstep, nstep)
    rlon, rlat, iu, ju = mod_advection.dist_topoints(lon, lat, lonpa, latpa, dvcoord, loc, su)
    test = True
    if ju != numpy.argmin(abs(lonpa - lon)):
        test = False
    if iu != numpy.argmin(abs(latpa - lat)):
        test=False
    if (lonpa - lon[ju])/dvcoord[1] != rlon:
        test = False
    if (latpa - lat[iu])/dvcoord[0] != rlat:
        test = False
    return test


def test_make_default(file_path):
    import lap.mod_tools as mod_tools
    p = mod_tools.load_python_file(file_path)
    mod_tools.make_default(p)
    return p


def test_lin1Dinterp(delta):
    import lap.mod_tools as mod_tools
    a = [1, 2]
    y = mod_tools.lin_1Dinterp(a, delta)
    test = False
    if y == (1 + delta):
        test = True
    return test


def test_lin_2Dinterp(delta, delta):
    import lap.mod_tools as mod_tools
    a = [[1, 2], [1, 2]]
    y =  mod_tools.lin_2Dinterp(a, delta, delta)
    test = False
    if y == (1 + delta):
        test = True
    return test


def test_bissextile(bissextile_year):
    import lap.mod_tools as mod_tools
    biss =  mod_tools.bissextile(bissextile_year)
    test = False
    if biss == 1:
        test = True
    return test


def test_dinm():
    import lap.mod_tools as mod_tools
    dinm = mod_tools.dinm(2000, 2)
    test = False
    if dinm == 29:
        test = True
    return test


def test_jj2date():
    import lap.mod_tools as mod_tools
    year, month. day = mod_tools.jj2date(24000)
    test = False
    if year == 2015 and month ==9 and day == 17:
        test = True
    return test


def test_haversine():
    import lap.mod_tools as mod_tools
    d =  haversine(0, 180, 0, 0)
    test = False
    if d == const.Rearth * numpy.pi:
        test = True
    return test



test = test_interpol_intime(0.5)
print(test)
loc = (0, 1)
test = test_dist_topoints(loc)
print(test)
test = test_lin1Dinterp(0.1)
print(test)
test = test_lin2Dinterp(0.1)
print(test)
test = test_lin2Dinterp(2000)
print(test)
test_dinm()
print(test)
test_jj2date()
