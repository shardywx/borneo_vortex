# Functions to calculate useful dynamical quantities 
# https://github.com/apaloczy/ap_tools/blob/b9ff0120665da7236511bc929f14e0ed695330df/dyn.py

import numpy as np
import math 

def deg2m_dist(lon, lat):
    """
    USAGE
    -----
    dx, dy = deg2m_dist(lon, lat)
    Calculates zonal and meridional grid spacing 'dx' and 'dy' (in meters)
    from the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees), using centered
    (forward/backward) finite-differences for the interior (edge) points. Assumes
    a locally rectangular cartesian on the scales of 'dx' and 'dy'.
    """
    lon, lat = map(np.asanyarray, (lon, lat))
    dlat  = np.gradient(lat)        # [deg]
    dlon  = np.gradient(lon)        # [deg]
    deg2m = 111120                  # [m/deg]
    # Account for divergence of meridians in zonal distance.
    dx = dlon * deg2m * math.cos(lat*np.pi/180.) # [m]
    dy = dlat * deg2m               # [m]

    return dx, dy

def divergence(lon, lat, u, v):
    """
    USAGE
    -----
    div = divergence(lon, lat, u, v)
    Calculates horizontal divergence 'div' (du/dx + dv/dy, in 1/s) from
    the 'u' and 'v' velocity arrays (in m/s) specified in spherical
    coordinates by the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees).
    """
    lon, lat, u, v = map(np.asanyarray, (lon, lat, u, v))

    dx, dy = deg2m_dist(lon, lat) # [m]
    _, dux = np.gradient(u)
    dvy, _ = np.gradient(v)

    dudx = dux/dx
    dvdy = dvy/dy
    div = dudx + dvdy # [1/s]

    return div

def strain(lon, lat, u, v):
    """
    USAGE
    -----
    alpha = strain(lon, lat, u, v)
    Calculates lateral rate of strain 'alpha' = sqrt[(du/dx - dv/dy)^2 + (du/dy + dv/dx)^2],
    in 1/s, from the 'u' and 'v' velocity arrays (in m/s) specified in spherical coordinates
    by the 'lon' and 'lat' 2D meshgrid-type arrays (in degrees).
    """
    lon, lat, u, v = map(np.asanyarray, (lon, lat, u, v))

    dx, dy = deg2m_dist(lon, lat) # [m]
    duy, dux = np.gradient(u)
    dvy, dvx = np.gradient(v)

    dudx = dux/dx
    dvdy = dvy/dy
    dudy = duy/dy
    dvdx = dvx/dx
    alpha = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2) # [1/s]

    return alpha

def vorticity(x, y, u, v, coord_type='geographic'):
    """
    USAGE
    -----
    zeta = vorticity(x, y, u, v, coord_type='geographic')
    Calculates the vertical component 'zeta' (dv/dx - du/dy, in 1/s) of the
    relative vorticity vector from the 'u' and 'v' velocity arrays (in m/s)
    specified in spherical coordinates by the 'lon' and 'lat' 2D meshgrid-type
    arrays (in degrees).
    """
    x, y, u, v = map(np.array, (x, y, u, v))

    if coord_type=='geographic':
        dx, dy = deg2m_dist(x, y)
    elif coord_type=='cartesian':
        dy, _ = np.gradient(y)
        _, dx = np.gradient(x)
    elif coord_type=='dxdy':
        dx, dy = x, y
        pass

    duy, _ = np.gradient(u)
    _, dvx = np.gradient(v)

    dvdx = dvx/dx
    dudy = duy/dy
    vrt = dvdx - dudy # [1/s]

    return vrt
