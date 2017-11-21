import math
import numpy
from shapely import geometry



def boundary_weight(r, xcp, bbox, polybox):
    ''' Set up a weith to points near the border proportional to the inside
    radius, bbox corresponds to [lon0, lon1, lat0, lat1].
    '''
    bboxr = [bbox[0] + r, bbox[1]-r, bbox[2] +r, bbox[3] -r]
    if ((xcp[0] >= bboxr[0]) and (xcp[0] <= bboxr[1])
         and (xcp[1] >= bboxr[2]) and (xcp[1] <= bboxr[3]):
         wp=1
    else:
         circle = geometry.Point(xcp[0], xcp[1]).buffer(r)
         wp = c.intersection(polybox)/ (2 * math.pi * r)
    return wp


def compute_ripley(x, p, xp, r, deltar, rmax, wp):
    gtmp=numpy.zeros(rmax)
    # compute distance from reference point, add haversine function ??
    d = numpy.sqrt((xc[:, 0] - xcp[0])**2+(yc[:, 1] - ycp[1])**2)
    gtmp[numpy.where((d > r) & (d <= r + deltar))] = wp / r
    return gtmp[r]


def default_parameters(p):
    # Maximum radius in km
    p.rmax = getattr(p, 'rmax', 200)
    # Incremental distance for radius in km
    p.deltar = getattr(p, 'deltar', 2)


def ripley(p):
    default_parameters(p)
    bbox = p.bbox
    polybox = geometry.Polygon([(0, 0), (latdist, 0), (latdist, londist),
                               (0, londist)])
    area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]

    gvec = numpy.zeros((len(x))
    # Loop on radius
    for r in numpy.range(0, rmax, deltar):
        npoints = 0
        # Loop on vortices
        for p in list(x):
            wp = boundary_weight(r, xcp, bbox, polybox)
            gtmp = ripley(x, p, x[p], r, deltar, rmax, wp)
        #gvec = gvec * area / (2 * math.pi * deltar)
        gvec[r] /= (len(x) ) ** 2
    gvec = gvec * area / (2 * math.pi * deltar)
    
