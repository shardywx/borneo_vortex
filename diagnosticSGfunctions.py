"""
(C) British Crown Copyright 2017 Met Office.
"""

import numpy as np
from scipy import signal
import scipy.stats as st

def thetaEstar(exner, theta, q, rh):

    '''
    Calculates equivalent potential temperature assuming saturation

    Parameters
    ----------
    exner :
        3 dimensional np arrays, first dimension assumed to be vertical,
        then northward and then eastward. Held at pressure levels.
    theta,q,rh :
        3 dimensional np arrays, first dimension assumed to be vertical,
        then northward and then eastward. Held at theta levels.
        3 dimensional np array, first dimension assumed to be vertical,
        then northward and then eastward. Held at pressure levels.
    Returns
    -------
    : thetEstar :
         np array with same dimensions as exner, held at pressure levels.

    '''

    nlevels = exner.shape[0]
    # calculate equivalent potential temperature at theta levels using (68).
    exneru = np.roll(exner, -1, 0)
    exneru[nlevels-1, ] = exner[nlevels-1, ] - .0001
    T = 0.5 * (exner + exneru) * theta
    # Calculate thetE assuming saturation
    thetav = theta * (1. + 0.61 * q)
    qsat = 100. * q / rh
    qsat= np.where(qsat > 1., 1., qsat)
    thetEstar = thetav * np.exp((3036. / T - 1.78) * qsat * (1. + 0.448 * qsat))
    
    return thetEstar


def fieldshift(field):
    '''
    Creates four new arrays obtained by shifting array by one point
    respectively to east, west, north, south. Periodic data assumed in esat
    west direction. Boundary values propagated inwards from north and south
    boundary.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.

    Returns
    -------
    : fielde, fieldw, fieldn, fields :
         np arrays with same dimensions as field.
    '''

    ny = field.shape[1]
    fielde = np.roll(field, -1, 2)
    fieldw = np.roll(field, 1, 2)
    fieldn = np.roll(field, -1, 1)
    fieldn[:, ny-1, :] = field[:, ny-1, :]
    fields = np.roll(field, 1, 1)
    fields[:, 0, :] = field[:, 0, :]

    return fielde, fieldw, fieldn, fields


def ddlambda(field, a, dlong, coslat):
    '''
    Differentiate input field with respect to east-west distance. Uses central
    diferencing. Assumes periodic data in east-west direction.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    a :
         scalar, earth's radius
    dlong :
         scalar, longitude increment for central differencing
    coslat :
         3 dimensional np array of shape (1,ny,1) containing cosine of latitude

    Returns
    -------
    : dfield :
         np array with same dimensions as field.
    '''

    fielde = np.roll(field, -1, 2)
    fieldw = np.roll(field, 1, 2)
    # dfield = np.empty((field.shape[0], field.shape[1], field.shape[2]))
    dfield = np.empty((field.shape))
    dfield = (fielde - fieldw) / (a * dlong * coslat)

    return dfield

def ddlambda_2d(field, a, dlong, coslat):
    '''  
    Description goes here...
    '''  
    fielde = np.roll(field, -1, 1)
    fieldw = np.roll(field, 1, 1)
    dfield = np.empty((field.shape))
    dfield = (fielde - fieldw) / (a * dlong * coslat)

    return dfield

def ddphi(field, a, dlat):
    '''
    Differentiate input field with respect to north-south distance. Uses
    central differencing. Copies boundary values one point north and south.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    a :
         scalar, earth's radius
    dlat :
         scalar, latitude increment for central differencing

    Returns
    -------
    : dfield :
         np array with same dimensions as field.

    '''

    ny = field.shape[1]
    fieldn = np.roll(field, -1, 1)
    fieldn[:, ny-1, :] = field[:, ny-1, :]
    fields = np.roll(field, 1, 1)
    fields[:, 0, :] = field[:, 0, :]
    dfield = (fieldn - fields) / (a * dlat)

    return dfield

def ddphi_2d(field, a, dlat):
    '''  
    Description goes here...
    '''  
    ny = field.shape[0]
    fieldn = np.roll(field, -1, 0)
    fieldn[ny-1, :] = field[ny-1, :]
    fields = np.roll(field, 1, 0)
    fields[0, :] = field[0, :]
    dfield = (fieldn - fields) / (a * dlat)

    return dfield

def ddr(field, thick2level):
    '''
    Differentiate input field with respect to vertical distance. Uses central
    differencing, but one-sided at upper and lower boundary..

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    thick2level :
        3 dimensional np array, dimensions as field, contains reciprocal of
        height difference between points used to calculate difference

    Returns
    -------
    : dfield :
         np array with same dimensions as field.

    '''

    nlevels = field.shape[0]
    fieldu = np.roll(field, -1, 0)
    fieldu[nlevels-1, ] = field[nlevels-1, ]
    fieldd = np.roll(field, 1, 0)
    fieldd[0, ] = field[0, ]
    dfield = (fieldu - fieldd) * thick2level

    return dfield


def ddeta(field):
    '''
    Difference input field with respect to model levels.
    Uses central differencing, but one-sided at upper and lower boundary..

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.
    '''

    nlevels = field.shape[0]
    fieldu = np.roll(field, -1, 0)
    fieldu[nlevels-1, ] = field[nlevels-1, ]
    fieldd = np.roll(field, 1, 0)
    fieldd[0, ] = field[0, ]
    dfield = fieldu - fieldd

    return dfield


def vertdiff(field, updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl):
    '''
    Apply vertical diffusion operator as used in (46) 
    The operator is defined by an input tridiagonal matrix, 
    which incorporates the necessary boundary conditions.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward. First dimension at least nbl.
    updiffcoeff :
        3 dimensional np array of matrix coefficients, first dimension nbl,
        other dimensions as field.
    diagdiffcoeff :
        3 dimensional np array of matrix coefficients, , first dimension nbl,
        other dimensions as field.
    downdiffcoeff :
        3 dimensional np array of matrix coefficients, , first dimension nbl,
        other dimensions as field.
    nbl :
         Scalar. Number of levels over which matric applied.

    Returns
    -------
    : dfield :
         np array, first dimension nbl, other dimensions as field.

    '''

    dfield = np.empty((nbl, field.shape[1], field.shape[2]))
    dfield[1: nbl-1, ] = (updiffcoeff[1: nbl-1, ] * field[2: nbl, ] + diagdiffcoeff[1: nbl-1, ]
                          * field[1: nbl-1, ] + downdiffcoeff[1: nbl-1, ] * field[0: nbl-2, ])
    dfield[0, ] = (updiffcoeff[0, ] * field[1, ]
                   + diagdiffcoeff[0, ] * field[0, ])
    dfield[nbl-1, ] = (downdiffcoeff[nbl-1, ] * field[nbl-2, ]
                       + diagdiffcoeff[nbl-1, ] * field[nbl-1, ])

    return dfield


def vert2diff(field, upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff, nbl):
    '''
    Apply vertical second diffusion operator and add coriolis squared*field
    to field.

    Apply vertical second diffusion operator as used in the left hand side
    of (47). The operator is defined by an input pentadiagonal matrix, which
    incorporates the necessary boundary conditions. Coefficients in downcoeff
    stored in locations [0:nbl-2] though applied at [1:nbl-1]. Coefficients in
    downdowncoeff stored in locations [0:nbl-3] though applied at [2:nbl-1].

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward. First dimension at least nbl.
    upupcoeff :
        3 dimensional np array of matrix coefficients, , first dimension nbl,
        other dimensions as field.
    upcoeff :
        3 dimensional np array of matrix coefficients,, first dimension nbl,
        other dimensions as field.
    diagcoeff :
        3 dimensional np array of matrix coefficients, , first dimension nbl,
        other dimensions as field.
    downcoeff :
        3 dimensional np array of matrix coefficients, , first dimension nbl,
        other dimensions as field.
    nbl :
         Scalar. Number of levels over which matric applied.

    Returns
    -------
    : dfield :
         np array, first dimension nbl, other dimensions as field.

    '''

    dfield = np.empty((nbl, field.shape[1], field.shape[2]))
    dfield[2: nbl-2, ] = (upupcoeff[2: nbl-2, ] * field[4: nbl, ]
                          + upcoeff[2: nbl-2, ] * field[3: nbl-1, ]
                          + diagcoeff[2: nbl-2, ] * field[2: lnbl-2, ]
                          + downcoeff[1: nbl-3, ] * field[1: nbl-3, ]
                          + downdowncoeff[0: nbl-4, ] * field[0: nbl-4, ])
    dfield[0, ] = (upupcoeff[0, ] * field[2, ] + upcoeff[0, ] * field[1, ]
                   + diagcoeff[0, ] * field[0, ])
    dfield[1, ] = (upupcoeff[1, ] * field[3, ] + upcoeff[1, ] * field[2, ]
                   + diagcoeff[1, ] * field[1, ] + downcoeff[0, ] * field[0, ])
    dfield[nbl-2, ] = (upcoeff[nbl-2, ] * field[nbl-1, ] + diagcoeff[nbl-2, ]
                       * field[nbl-2, ] + downcoeff[nbl-3, ] * field[nbl-3, ]
                       + downdowncoeff[nbl-4, ] * field[nbl-4, ])
    dfield[nbl-1, ] = (diagcoeff[nbl-1, ] * field[nbl-1, ] + downcoeff[nbl-2, ]
                       * field[nbl-2, ] + downdowncoeff[nbl-3, ] * field[nbl-3, ])

    return dfield


def merge(field1, field2, lat, lmin, lmax):
    '''
    Merge two fields within a given latitudinal range (both north and south of
    the equator). Assume field1 is extratropical field, and field2 is the
    tropical field.

    Parameters
    ----------
    field1 :
        Extratropical field. 3 dimensional np array, first dimension assumed
        to be vertical, then northward and then eastward.
    field2 :
        Tropical field. Same dimensions as field1
    lat :
        1 dimensional array of latitude points.
    lmin,lmax :
        Defines latitudinal range in which field1 and field2 are merged.
        0 < lmin < lmax

    Returns
    --------

    field :
        Np array, same shape as field1 and field2.
    '''

    lmin = lmin * np.pi / 180
    lmax = lmax * np.pi / 180

    lss = np.abs(lat+lmax).argmin()
    lsn = np.abs(lat+lmin).argmin()
    lns = np.abs(lat-lmin).argmin()
    lnn = np.abs(lat-lmax).argmin()

    Lss = lat[lss]
    Lsn = lat[lsn]
    Lns = lat[lns]
    Lnn = lat[lnn]

    field = field1.copy()

    # Southern layer
    for j in range(lss, lsn):
        field[:, j, :] = ((field1[:, j, :] * (Lsn - lat[j]) + field2[:, j, :] * (lat[j] - Lss))
                          / (Lsn - Lss))

    # Equatorial region
    field[:, lsn:lns+1, :] = field2[:, lsn:lns+1, :]

    # Northern layer
    for j in range(lns+1, lnn):
        field[:, j, :] = ((field1[:, j, :] * (lat[j] - Lns) + field2[:, j, :] * (Lnn - lat[j]))
                          / (Lnn - Lns))

    return field


def blh(km):
    '''
    Finds the boundary layer height according to some criteria.
    Simple idea: first level at which km drops below a threshold value.

    Parameters
    ------------
    km :
        Diffusion coefficient from UM data. 3 dimensional np array, first
        dimension assumed to be vertical, then northward and then eastward.

    Returns
    ------------
    BLH :
        2 dimensional array, first dimension northward, second dimension
        eastward.
    '''

    km0 = 0.5  # Threshold value

    nl, ny, nx = np.shape(km)
    BLH = np.zeros((ny, nx))
    m = 2  # Lower bound on BLH
    for j in range(0, ny):
        for i in range(0, nx):
            BLH[j, i] = np.where(km[m:, j, i] < km0)[0][0] + m

    BLH = np.where(BLH > 8, 8, BLH) # have replaced 8 with 14 as a test 
    return BLH


def contop(blh, thetEstar):

    '''
    Finds the top of the convection layer.
    as first level at which the equivalent potential temperature assuming
    saturation is greater than the equivalent potential temperature at

    the boundary layer top.

    Parameters
    ------------
    thetEstar :
        thetaEstar equivalent potential temperature assuming saturation.
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.

    Returns
    ------------
    contop :
        2 dimensional array, first dimension northward, second dimension
        eastward.
    '''

    nl, ny, nx = np.shape(thetEstar)
    contop = blh.copy()
    thetBL = blh.copy()
    for j in range(0, ny):
        for i in range(0, nx):
            thetBL[j, i] = thetEstar[int(blh[j, i]), j, i]

    for j in range(0, ny):
        for i in range(0, nx):
            if thetEstar[int(blh[j, i]) + 1, j, i] < thetBL[j, i]:
                contop[j, i] = np.where(thetEstar[int(blh[j, i]):, j, i] >
                                        thetBL[j, i])[0][0] + int(blh[j, i])

    contop = np.where(contop > 55, 55, contop)

    return contop


def trisolve_blh(km, h, rhs0, lats, ug, BLH):
    '''
    Tridiagonal solve of equation (10), with variable boundary layer height
    (BLH). BLH is set by function SG.blh.

    Upper BC is ue = ug, applied at level stored in BLH. With BLH varying with
    latitude and longitude, the linear system is defined and solved in two
    nested for loops, looping through all zonal grid points within a
    latitudinal range defined by lats.

    rhs0 is the forcing, and h is the level thickness, required for accurate
    differencing.

    Outputs ue.
    '''

    rhs = rhs0.copy()
    # Regularise Km. This will edit Km throughout the entire core.
    km = np.where(km > 1.e2, 1.e2, km)
    km = np.where(km < 1.e-3, 1.e-3, km)

    h_1 = 10.0  # Needed for lower BC.
    a = 1.00  # Strengthen the diagonal?

    nbl, ny, nx = np.shape(rhs)
    ue = ug.copy()

    for j in lats:
        for i in range(0, nx):
            # Define BL height at each horizontal grid point. Initialise matrix
            # and output.
            N = int(BLH[j, i])
            A = np.zeros((N, N))

            # Lower BC
            A[0, 0] = ((-2 * km[0, j, i] * (1. / h[0, j, i] + 1. / h_1)
                       / (h[0, j, i] + h_1)) * a)
            A[0, 1] = 2 * km[0, j, i] / (h[0, j, i] * (h[0, j, i] + h_1))

            # Upper BC
            A[N-1, N-1] = ((- (km[N-1, j, i] + km[N-2, j, i])
                           * (1./h[N-1, j, i] + 1. / h[N-2, j, i])
                           / (h[N-1, j, i] + h[N-2, j, i])) * a)
            A[N-1, N-2] = ((km[N-1, j, i] + km[N-2, j, i])
                           / (h[N-1, j, i] * (h[N-1, j, i] + h[N-2, j, i]))
                           - (km[N-1, j, i] - km[N-2, j, i])
                           / (h[N-1, j, i] + h[N-2, j, i]) ** 2)

            for k in range(1, N-1):
                A[k, k] = ((-(km[k, j, i] + km[k-1, j, i])
                           * (1. / h[k, j, i] + 1. / h[k-1, j, i])
                           / (h[k, j, i] + h[k-1, j, i])) * a)

                A[k, k+1] = ((km[k, j, i] + km[k-1, j, i])
                             / (h[k, j, i] * (h[k, j, i] + h[k-1, j, i]))
                             + (km[k, j, i] - km[k-1, j, i])
                             / (h[k, j, i] + h[k-1, j, i]) ** 2)

                A[k, k-1] = ((km[k, j, i] + km[k-1, j, i])
                             / (h[k-1, j, i] * (h[k, j, i] + h[k-1, j, i]))
                             - (km[k, j, i] - km[k-1, j, i])
                             / (h[k, j, i] + h[k-1, j, i]) ** 2)

                # End

            # Edit RHS term to account for the upper BC.
            rhs[N-1, j, i] = (rhs[N-1, j, i] - ((km[N-1, j, i] + km[N-2, j, i])
                              / (h[N-1, j, i] * (h[N-1, j, i] + h[N-2, j, i]))
                              + (km[N-1, j, i] - km[N-2, j, i])
                              / (h[N-1, j, i] + h[N-2, j, i]) ** 2)
                              * ug[N, j, i])

            # Solve.
            ue[0:N, j, i] = np.linalg.solve(A[:, :], rhs[0:N, j, i])

            # U = 60.
            # ug = np.where(np.abs(ug)>U,np.sign(ug)*U,ug)

    return ue


def trisolve(field, upcoeff, diagcoeff, downcoeff):
    '''
    Apply inverse of tridiagonal matrix to input data. Input data destroyed.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    upcoeff, diagcoeff, downcoeff :
        3 dimensional np arrays of matrix coefficients, dimensions as field.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.

     '''

    nlevels = field.shape[0]
    dfield = field.copy()
    factor = np.empty((nlevels, field.shape[1], field.shape[2]))
    diagcoeff[0, ] = 1. / diagcoeff[0, ]

    for k in range(1, nlevels):
        factor[k, ] = downcoeff[k, ] * diagcoeff[k-1, ]
        diagcoeff[k, ] = 1. / (diagcoeff[k, ] - factor[k, ] * upcoeff[k-1, ])

    for k in range(1, nlevels):
        dfield[k, ] = dfield[k, ] - factor[k, ] * dfield[k-1, ]
    dfield[nlevels-1, ] = diagcoeff[nlevels-1, ] * dfield[nlevels-1, ]

    for k in range(nlevels-2, -1, -1):
        dfield[k, ] = (diagcoeff[k, ] * (dfield[k, ] - upcoeff[k, ]
                       * dfield[k+1, ]))

    return dfield


def pentasolve(field, upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff):
    '''
    Apply inverse of pentadiagonal matrix to input data. Input data destroyed.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    upupcoeff, upcoeff. diagcoeff, downcoeff, downdowncoeff :
        3 dimensional np arrays of matrix coefficients, dimensions as field.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.
     '''

    nbl = field.shape[0]
    dfield = np.empty((nbl, field.shape[1], field.shape[2]))
    for k in range(1, nbl-1):
        factor = downcoeff[k-1, ] / diagcoeff[k-1, ]
        diagcoeff[k, ] = diagcoeff[k, ] - upcoeff[k-1, ] * factor
        upcoeff[k, ] = upcoeff[k, ] - upupcoeff[k-1, ] * factor
        field[k, ] = field[k, ] - field[k-1, ] * factor
        factor = downdowncoeff[k-1, ] / diagcoeff[k-1, ]
        downcoeff[k, ] = downcoeff[k, ] - factor * upcoeff[k-1, ]
        diagcoeff[k+1, ] = diagcoeff[k+1, ] - factor * upupcoeff[k-1, ]
        field[k+1, ] = field[k+1, ] - factor * field[k-1, ]

    factor = downcoeff[nbl-2, ] / diagcoeff[nbl-2, ]
    diagcoeff[nbl-1, ] = diagcoeff[nbl-1, ] - factor * upcoeff[nbl-2, ]
    dfield[nbl-1, ] = ((field[nbl-1, ] - factor * field[nbl-2, ])
                       / diagcoeff[nbl-1, ])
    dfield[nbl-2, ] = ((field[nbl-2, ] - upcoeff[nbl-2, ] * dfield[nbl-1, ])
                       / diagcoeff[nbl-2, ])

    for k in range(nbl-3, -1, -1):
        dfield[k, ] = ((field[k, ] - upupcoeff[k, ] * dfield[k+2, ]
                       - upcoeff[k, ] * dfield[k+1]) / diagcoeff[k, ])

    return dfield


def pentasolve2(field, upupcoeff, upcoeff, diagcoeff, downcoeff,
                downdowncoeff):
    '''
    Same as above function, but an outer function loops through horizontal
    gridpoints rather than using vectorised approach as in original solver.
    This is slower, but allows for more flexibility, and specifically allows
    for a variable boundary layer height. Outer loops applied by Pinverse2.
    '''

    nbl = field.shape[0]
    dfield = np.empty(nbl)
    for k in range(1, nbl-1):
        factor = downcoeff[k-1] / diagcoeff[k-1]
        diagcoeff[k] = diagcoeff[k] - upcoeff[k-1] * factor
        upcoeff[k] = upcoeff[k] - upupcoeff[k-1] * factor
        field[k] = field[k] - field[k-1] * factor
        factor = downdowncoeff[k-1] / diagcoeff[k-1]
        downcoeff[k] = downcoeff[k] - factor * upcoeff[k-1]
        diagcoeff[k+1] = diagcoeff[k+1] - factor * upupcoeff[k-1]
        field[k+1] = field[k+1] - factor * field[k-1]
    factor = downcoeff[nbl-2] / diagcoeff[nbl-2]
    diagcoeff[nbl-1] = diagcoeff[nbl-1] - factor * upcoeff[nbl-2]
    dfield[nbl-1] = (field[nbl-1] - factor * field[nbl-2]) / diagcoeff[nbl-1]
    dfield[nbl-2] = ((field[nbl-2] - upcoeff[nbl-2] * dfield[nbl-1])
                     / diagcoeff[nbl-2])
    for k in range(nbl-3, -1, -1):
        dfield[k] = ((field[k] - upupcoeff[k] * dfield[k+2] - upcoeff[k]
                     * dfield[k+1]) / diagcoeff[k])

    return dfield


def saveCubes(path, dpidt_cube, dugdt_cube, dvgdt_cube, dthetavdt_cube,
              u_cube, v_cube, w_cube):
    from iris import save as save

    save(dpidt_cube, path + 'dpidt_cube.nc')
    save(dugdt_cube, path + 'dugdt_cube.nc')
    save(dvgdt_cube, path + 'dvgdt_cube.nc')
    save(dthetavdt_cube, path + 'dthetavdt_cube.nc')
    save(u_cube, path + 'u_cube.nc')
    save(v_cube, path + 'v_cube.nc')
    save(w_cube, path + 'w_cube.nc')

    return


def loadCubes(path):
    from iris import load_cube as load

    u = (load(path + 'u_cube.nc')).data
    v = (load(path + 'u_cube.nc')).data
    ug = (load(path + 'dugdt_cube.nc')).data
    vg = (load(path + 'dvgdt_cube.nc')).data

    return u, v, ug, vg


def lat_tropics(lat, lon, outer, inner):
    '''
    Define indices corresponding to tropical band.

    Parameters
    ----------
    lat:
        1-D array of latitudinal values (in radians).
    outer:
        Local parameter, limit of the larger latitudinal band.
    inner:
        Local paramter, limit of narrower latitudinal band.

    Returns
    -------
    tropics:
        Indices corresponding to tropical region.
    '''

    lat_tropics_outer = [j for j in range(0, len(lat))
                         if - outer < lat[j] * 180. / np.pi < outer]
    lat_tropics = [j for j in range(0, len(lat))
                   if - inner < lat[j] * 180. / np.pi < inner]

    return lat_tropics_outer, lat_tropics


def factor1(lat, lat1_deg=4.0, lat2_deg=8.0):
    '''
    Returns Mike's original tropical filter factor.
    '''

    lat1 = lat1_deg  * np.pi / 180
    lat2 = lat2_deg * np.pi / 180
    sinsq_lat1 = np.sin(lat1)**2
    sinsq_lat2 = np.sin(lat2)**2

    ny = lat.size
    factor1 = np.zeros(ny)
    sinlat = np.sin(lat)
    for j in range(0, ny):
        factor1[j] = 1.
        sinsq = sinlat[j] ** 2 # < 0.025 (?). Factors depend on when you transition from one function to the next. 
        if sinsq < sinsq_lat2:
            factor1[j] = sinsq / sinsq_lat2
            if sinsq < sinsq_lat1:
                factor1[j] = sinsq**2 / (sinsq_lat1 * sinsq_lat2)

    print('tropical filter factor: min = {0}, max = {1}'.format(np.min(factor1),np.max(factor1)) )
    return factor1

def gkern(kernel_length: int, nsig: float): 
    '''
    returns a 2D Gaussian kernel with side-length (for use in convolution)
    stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    '''
    x = np.linspace(-nsig, nsig, kernel_length+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

def tropicalfilter(field, latitude):
    '''
    Calculate new field with same zonal mean as input, but rescaled departures
    from zonal mean as specified in the code.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    latitude :
        1 dimensional np array of latitude values.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.

    '''

    factor = factor1(latitude)
    print('working on tf. factor min = {0}, max = {1}.'.format(np.min(factor),np.max(factor)) )

    dfield0 = np.empty((field.shape[0], field.shape[1], field.shape[2]))
    dfield  = dfield0
    ny = field.shape[1]
    nlevels = field.shape[0]
    for k in range(0, nlevels):
        for j in range(0, ny):
            zonalmean = field[k, j, ].sum() / field[k, j, ].size
            dfield0[k, j, :] = zonalmean + (field[k, j, :] - zonalmean) * factor[j]

    # apply convolution function (updated 28/09/20)
    k2d = gkern(7, 1.0)
    '''
    n0 = 5
    k2d = np.array([ [0.5, 1.0, 0.5], [1.0, 2.5, 1.0], [0.5, 1.0, 0.5] ])
    s2d = np.sum(k2d)
    '''
    for k in range(0, nlevels):
        n0 = 7
        #dfield[k,:,:] = signal.convolve2d(dfield0[k,:,:], np.ones((n0,n0))/float(n0**2), boundary='symm', mode='same')
        dfield[k,:,:] = signal.convolve2d(dfield0[k,:,:], k2d, boundary='symm', mode='same')

    return dfield

def tropicalfilter_grad(field, latitude):
    '''
    Calculate new field with same zonal mean as input, but rescaled departures
    from zonal mean as specified in the code. Edit of above function. The
    filter is gradually switched off between levels lmax and lmin, defined
    within function.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.
    latitude :
        1 dimensional np array of latitude values.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.

    '''

    # what type of variable (+ dims) is 'factor'?
    factor = factor1(latitude)
    #print('working on tf_grad. factor min = {0}, max = {1}.'.format(np.min(factor),np.max(factor)) )
    # factor = factor_lin(latitude,5.,20.)
    # factor = factor2(latitude,5.,20.)

    # Levels over which filter is smoothly removed (no filter applied on or below level lmin).
    lmin = 6
    lmax = 12

    # Most areas aren't filtered, and need no operating.
    dfield0 = field.copy(); dfield = dfield0
    ny = field.shape[1]
    nlevels = field.shape[0]
    for k in range(lmax+1, nlevels):
        for j in range(0, ny):
            zonalmean = field[k, j, ].sum() / field[k, j, ].size
            dfield0[k, j, :] = zonalmean + (field[k, j, :] - zonalmean) * factor[j]

    '''
    # apply convolution function (updated 28/09/20)
    k2d = gkern(7, 1.0)
    for k in range(lmax+1, nlevels):
        n0 = 7
        #dfield[k, :, :] = signal.convolve2d(dfield0[k, :, :], np.ones((n0,n0))/float(n0**2), boundary='symm', mode='same')
        dfield[k, :, :] = signal.convolve2d(dfield0[k, :, :], k2d, boundary='symm', mode='same')
    '''

    for k in range(lmin, lmax+1):
        for j in range(0, ny):
            zonalmean = field[k, j, ].sum() / field[k, j, ].size
            factor0 = (((factor[j] - 1) * k + (lmax - factor[j] * lmin))
                       / (lmax - lmin))
            # No need to worry about factor being out of range [f0,1] due
            # to limits of outer loop.
            if factor0 < 0:
                factor0 = 0
            if factor0 > 1:
                factor0 = 1
            dfield0[k, j, :] = zonalmean + (field[k, j, :] - zonalmean) * factor0

    # apply convolution function (updated 28/09/20)
    '''
    n0 = 6
    k2d = np.array([ [0.5, 1.0, 0.5], [1.0, 2.5, 1.0], [0.5, 1.0, 0.5] ])
≈    s2d = np.sum(k2d)
    '''
    k2d = gkern(7, 1.0)
    for k in range(0, nlevels):
        #n0 = 7
        #dfield[k, :, :] = signal.convolve2d(dfield0[k, :, :], np.ones((n0,n0))/float(n0**2), boundary='symm', mode='same')
        dfield[k, :, :] = signal.convolve2d(dfield0[k, :, :], k2d, boundary='symm', mode='same')

    return dfield


def smooth(field):
    '''
    Smooth field horizontally by applying filter as defined in code.

    Parameters
    ----------
    field :
        3 dimensional np array, first dimension assumed to be vertical, then
        northward and then eastward.

    Returns
    -------
    : dfield :
         np array with same dimensions as field.
    '''

    dfield = np.empty((field.shape[0], field.shape[1], field.shape[2]))
    ny = field.shape[1]
    fieldn = np.roll(field, -1, 1)
    fieldn[:, ny-1, :] = field[:, ny-1, :]
    fields = np.roll(field, 1, 1)
    fields[:, 0, :] = field[:, 0, :]
    dfield = (0.6 * field + 0.1 * (np.roll(field, -1, 2)
              + np.roll(field, 1, 2) + fieldn + fields))

    return dfield


def smoother(field, ni):
    '''
    Make a field even smoother than the function above.
    '''

    dfield = np.zeros((field.shape[0], field.shape[1]))
    for i in range(0, ni):
        for j in range(0, ni):
            dfield += np.roll(np.roll(field, j-ni+1, 0), i-ni+1, 1) / ni**2

    return dfield


def smoothv(field):
    '''
    Smooth field vertically
    '''
    dfield = field.copy()
    print(np.shape(dfield))
    for k in range(1, np.shape(field)[0]-1):
        dfield[k, ] = (field[k-1, ] + field[k, ] + field[k+1, ]) / 3.

    return dfield


def div(fieldx, fieldy, fieldz, a, coslat, dlong, dlat, thick2level):
    '''
    calculate three-dimensional divergence of input vector, using model level
     as vertical coordinate.

    Parameters
    ----------
    fieldx, fieldy, fieldz :
        3 dimensional np arrays containing zonal, meridional and vertical
        components of input vector, first dimension
        assumed to be vertical, then northward and then eastward.
    a :
        scalar, earth's radius
    coslat :
        3 dimensional np array shape (1,ny,1) containing cosine of latitude
    dlong :
        scalar, longitude increment for central differencing
    dlong :
        scalar, latitude increment for central differencing
    thick2level :
        3 dimensional np array, dimensions as field, contains reciprocal of
        height difference between points used to calculate difference

    Calls
    -------
    : diagnosticSGfunctions(ddphi), diagnosticSGfunctions(ddlambda),
      diagnosticSGfunctions(ddeta)

    Returns
    -------
    : divergence :
         np array with same dimensions as field.
    '''

    
    fieldy = fieldy * coslat / thick2level
    divergence = ddphi(fieldy, a, dlat)
    divergence = divergence / coslat
    fieldx = fieldx / thick2level
    fieldz = fieldz / thick2level
    divergence = divergence + ddlambda(fieldx, a, dlong, coslat)
    divergence = divergence + ddeta(fieldz)

    return(divergence)


def Pinverse(A1, A2, A3, QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33,
             nbl, ny, coriolis20, upupcoeff, upcoeff, diagcoeff, downcoeff,
             downdowncoeff):
    '''
    multiply vector (A1,A2,A3) by P^{-1} eq. (62)-(64).


    Parameters
    ----------
    A1, A2, A3 :
        3 dimensional np arrays containing zonal, meridional and vertical
        components of input vector, first dimension
        assumed to be vertical, then northward and then eastward.
    QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33 :
        3 dimensional np arrays containing components of the inverse of the
        P_2 matrix, eq. (63), first dimension
        assumed to be vertical, then northward and then eastward.
    nbl :
        Scalar. Number of boundary layer levels.
    ny :
        Scalar. Number of latitudes.
    coriolis20 :
        np array shape (1, ny, 1) of values of Coriolis squared, regularised to avoid
        zero values.
    upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff :
        3 dimensional np arrays of matrix coefficients, dimensions as field.

    Calls
    -------
    : diagnosticSGfunctions(pentasolve)

    Returns
    -------
    : dfield :
         np array with same dimensions as field.
    '''

    # Multiply by P_2^{-1} eq. (63)
    B1 = QI11 * A1 + QI12 * A2 + QI13 * A3
    B2 = QI21 * A1 + QI22 * A2 + QI23 * A3
    B3 = QI31 * A1 + QI32 * A2 + QI33 * A3

    # Multiply by P_3^{-1} eq. (64)
    B1[0:nbl, ] = B1[0:nbl, ] * coriolis20
    B2[0:nbl, ] = B2[0:nbl, ] * coriolis20

    # multiply by P_1^{-1} eq. (62) using pentadiagonal solve to invert P_1
    rhs = B1[0:nbl, ].copy()
    upwork = upcoeff.copy()
    diagwork = diagcoeff.copy()
    downwork = downcoeff.copy()
    lhs = pentasolve(rhs, upupcoeff, upwork, diagwork, downwork, downdowncoeff)
    B1[0:nbl, ] = lhs
    rhs = B2[0:nbl, ].copy()
    upwork = upcoeff.copy()
    diagwork = diagcoeff.copy()
    downwork = downcoeff.copy()
    lhs = pentasolve(rhs, upupcoeff, upwork, diagwork, downwork, downdowncoeff)
    B2[0:nbl, ] = lhs

    return(B1, B2, B3)


def Pinverse2(A1, A2, A3, QI31, QI32, QI33, lats,
              nbl, ny, coriolis20, u, v, w,
              upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff, blh):
    '''
    Same as above Pinverse function, but simplified to solve for uag,vag,w in
    the tropics.

    Input A1,A2,A3 should be different (i.e. new G terms as calculated by
    calcG2 and edited by G_BC).

    Don't need to do all inverse operations of above system: P3 is entirely
    redundant.

    The solve is executed by looping through zonal and meridional grid points,
    as opposed to the vectorised approach in Pinverse. This is slower, but
    allows for variable boundary layer height.

    Calls
    -------
    : diagnosticSGfunctions(pentasolve2)
    '''

    nbl, ny, nx = np.shape(A1)

    B1 = A1.copy()
    B2 = A2.copy()
    B3 = A3.copy()

    for j in lats:
        for i in range(0, nx):
            N = int(blh[j, i])
            # Multiply by P_2^{-1} eq. (63)
            # B3[0:N, j, i] = (QI33[0:N, j, i] * A3[0:N, j, i]
            #                  + QI31[0:N, j, i] * A1[0:N, j, i]
            #                  + QI32[0:N, j, i] * A2[0:N, j, i])
            B3[0:N, j, i] = QI33[0:N, j, i] * A3[0:N, j, i]
            B3[N:nbl, j, i] = w[N:nbl, j, i]

            # multiply by P_1^{-1} eq. (62) using pentadiagonal solve
            # to invert P_1
            rhs = B1[0:N, j, i].copy()
            upwork = upcoeff[0:N, j, i].copy()
            diagwork = diagcoeff[0:N, j, i].copy()
            downwork = downcoeff[0:N, j, i].copy()
            upupwork = upupcoeff[0:N, j, i].copy()
            downdownwork = downdowncoeff[0:N, j, i].copy()
            lhs = pentasolve2(rhs, upupwork, upwork, diagwork, downwork, downdownwork)
            B1[0:N, j, i] = lhs
            B1[N:nbl, j, i] = u[N:nbl, j, i]

            rhs = B2[0:N, j, i].copy()
            upwork = upcoeff[0:N, j, i].copy()
            diagwork = diagcoeff[0:N, j, i].copy()
            downwork = downcoeff[0:N, j, i].copy()
            lhs = pentasolve2(rhs, upupwork, upwork, diagwork, downwork, downdownwork)
            B2[0:N, j, i] = lhs
            B2[N:nbl, j, i] = v[N:nbl, j, i]

    return(B1, B2, B3)


def CalcG(Q11R, Q12, Q21, Q22R, Q31, Q32,
          ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
          updiffcoeff, diagdiffcoeff, downdiffcoeff):
    '''
    Calculate G vector including forcing eq (60)

    Parameters
    ----------
    Q11R, Q12, Q21, Q22R, Q31, Q32 :
        3 dimensional np arrays containing components of the G matrix,
        eq. (60), with forcing term S omitted, first dimension assumed to be
        vertical, then northward and then eastward.
    ug, vg :
        3 dimensional np arrays containing zonal and meridional components of
        ue, ve. Dimensions as Q11R.
    g :
         Scalar. Acceleration due to gravity.
    thetav :
         3 dimensional np arrays containing thetav. Dimensions as Q11R.
    uinc, vinc, heat :
        3 dimensional np arrays containing increments to zonal wind,
        medinional wind and potential temperature. Dimensions as Q11R.
    nbl :
         Scalar. Number of boundary layer levels.
    ny :
         Scalar. Number of latitudes.
    coriolis :
          np array shape (1, ny, 1) of values of Coriolis parameter.
    updiffcoeff, diagdiffcoeff, downdiffcoeff :
        3 dimensional np arrays of matrix coefficients for vertical diffusion,
        dimensions as field.

    Calls
    -------
    : diagnosticSGfunctions(pentasolve)

    Returns
    -------
    : G1, G2, G3 :
        np arrays containing zonal, meriodional and vertical components of
        G vector with same dimensions as Q11R.
    '''

    (thetae, thetaw, thetan, thetas) = fieldshift(thetav)
    ugbythetaheat = 2. * vg / (thetae + thetaw) * heat
    vgbythetaheat = 2. * ug / (thetan + thetas) * heat

    G1 = -Q11R * ug - Q12 * vg
    G2 = -Q21 * ug - Q22R * vg
    G3 = -Q31 * ug - Q32 * vg + g * heat / thetav

    G1 = G1 - coriolis * (vgbythetaheat - vinc)
    G2 = G2 + coriolis * (ugbythetaheat - uinc)

    G1[0:nbl, ] = G1[0:nbl, ] - vertdiff(ugbythetaheat, updiffcoeff,
                                         diagdiffcoeff, downdiffcoeff, nbl)
    G2[0:nbl, ] = G2[0:nbl, ] - vertdiff(vgbythetaheat, updiffcoeff,
                                         diagdiffcoeff, downdiffcoeff, nbl)

    return(G1, G2, G3)


def CalcG2(Q11R, Q31, Q32, ug, vg, g, thetav, uinc, vinc, heat, nbl, ny,
           coriolis, updiffcoeff, diagdiffcoeff, downdiffcoeff):
    '''
    Exactly the same as the above function, but here we set f=0 in the tropics,
    and remove advection terms. To be used to prepare input for Pinverse2 and
    pentasolve2.
    '''

    G1_ = np.zeros(np.shape(Q11R))
    G2_ = np.zeros(np.shape(Q11R))

    (thetae, thetaw, thetan, thetas) = fieldshift(thetav)
    ugbythetaheat = 2. * vg / (thetae + thetaw) * heat
    vgbythetaheat = 2. * ug / (thetan + thetas) * heat

    G3_ = g * heat / thetav  # - Q31 * ug - Q32 * vg

    G1_[0:nbl, ] = - vertdiff(ugbythetaheat, updiffcoeff,
                              diagdiffcoeff, downdiffcoeff, nbl)
    G2_[0:nbl, ] = - vertdiff(vgbythetaheat, updiffcoeff,
                              diagdiffcoeff, downdiffcoeff, nbl)

    return(G1_, G2_, G3_)


def G_BC(G1_, G2_, u, v, upcoeff, upupcoeff, lats, blh):
    '''
    Edit G1_ and G2_ so that they include an upper BC for updated uag, vag in
    new ageostrophic solver.

    Upper BC is applied at blh, which may vary horizontally, so code is written
    to allow for this.

    N = blh is the number of degrees of freedom in the vertical solve, with
    upper BCs applied at two

    levels (pentadiagonal solve). We therefore edit Gi_[N-1] and Gi_[N-2].
    '''

    nbl, ny, nx = np.shape(G1_)

    for j in lats:
        for i in range(0, nx):
            N = int(blh[j, i])
            G1_[N-1, j, i] = (G1_[N-1, j, i] - upcoeff[N-1, j, i]
                              * u[N, j, i] - upupcoeff[N-1, j, i]
                              * u[N+1, j, i])
            G2_[N-1, j, i] = (G2_[N-1, j, i] - upcoeff[N-1, j, i]
                              * v[N, j, i] - upupcoeff[N-1, j, i]
                              * v[N+1, j, i])
            G1_[N-2, j, i] = G1_[N-2, j, i] - upupcoeff[N-2, j, i] * u[N, j, i]
            G2_[N-2, j, i] = G2_[N-2, j, i] - upupcoeff[N-2, j, i] * v[N, j, i]

    return G1_, G2_


def CalcGinc(ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
             updiffcoeff, diagdiffcoeff, downdiffcoeff):
    '''
    Calculate G vector from external forcing increments only

    Parameters
    ----------
    ug, vg :
        3 dimensional np arrays containing zonal and meridional components of
        ue, ve. Dimensions as Q11R.
    g :
         Scalar. Acceleration due to gravity.
    thetav :
         3 dimensional np arrays containing thetav. Dimensions as Q11R.
    uinc, vinc, heat :
        3 dimensional np arrays containing increments to zonal wind, medinional
        wind and potential temperature. Dimensions as Q11R.
    ny :
         Scalar. Number of latitudes.
    coriolis :
          np array shape (1, ny, 1) of values of Coriolis parameter.
    updiffcoeff, diagdiffcoeff, downdiffcoeff :
          3 dimensional np arrays of matrix coefficients for vertical
          diffusion, dimensions as field.

    Calls
    -------
    : diagnosticSGfunctions(vertdiff)

    Returns
    -------
    : G1, G2, G3 :
        np arrays containing zonal, meriodional and vertical components of
        G vector with same dimensions as heat.
    '''

    (thetae, thetaw, thetan, thetas) = fieldshift(thetav)
    ugbythetaheat = 2. * vg / (thetae + thetaw) * heat
    vgbythetaheat = 2. * ug / (thetan + thetas) * heat
    G3 = g * heat / thetav
    G1 = G3.copy()
    G2 = G3.copy()
    G1 = -coriolis * (vgbythetaheat - vinc)
    G2 = coriolis * (ugbythetaheat - uinc)
    G1[0:nbl, ] = G1[0:nbl, ] - vertdiff(ugbythetaheat, updiffcoeff,
                                         diagdiffcoeff, downdiffcoeff, nbl)
    G2[0:nbl, ] = G2[0:nbl, ] - vertdiff(vgbythetaheat, updiffcoeff,
                                         diagdiffcoeff, downdiffcoeff, nbl)

    return(G1, G2, G3)


def normvector(A1, A2, A3, title):
    '''
    Calculate and print norm of input vector

    Parameters
    ----------
    A1, A2, A3 :
        3 dimensional np arrays containing zonal, meridional and vertical
        components of input vector, first dimension
        assumed to be vertical, then northward and then eastward.
    title :
        string
    '''

    usq = A1 * A1
    normu = np.sqrt(usq.sum() / usq.size)
    vsq = A2 * A2
    normv = np.sqrt(vsq.sum() / vsq.size)
    wsq = A3 * A3
    normw = np.sqrt(wsq.sum() / wsq.size)

    return


def newcube(field,cube,units=None):    
    '''
    Create new cube from supplied data and template
    ''' 
    import iris
    model_level_number = cube.coord('model_level_number').points
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    # creating new coordinates
    
    level_coord_name = cube.coord('model_level_number').standard_name
    level_coord_units = cube.coord('model_level_number').units
    level_coord_attributes = cube.coord('model_level_number').attributes
    level_coord = iris.coords.DimCoord(model_level_number, standard_name=level_coord_name,
                                     units=level_coord_units, var_name=level_coord_name)
    level_coord.guess_bounds()


    lat_coord_name = cube.coord('latitude').standard_name
    lat_coord_units = cube.coord('latitude').units
    lat_coord_system = cube.coord_system()
    lat_coord = iris.coords.DimCoord(latitude, standard_name=lat_coord_name,
                                 units=lat_coord_units, var_name=lat_coord_name,
                                 coord_system=lat_coord_system)
    lat_coord.guess_bounds()

    lon_coord_name = cube.coord('longitude').standard_name
    lon_coord_units = cube.coord('longitude').units
    lon_coord_system = cube.coord_system()
    lon_coord = iris.coords.DimCoord(longitude, standard_name=lon_coord_name,
                                 units=lon_coord_units, var_name=lon_coord_name,
				 coord_system=lon_coord_system, circular=True)
    lon_coord.guess_bounds()

    dimension_list = [(level_coord, 0), (lat_coord, 1), (lon_coord, 2)]
    new_cube = iris.cube.Cube(field, standard_name=cube.standard_name, units=units,
                          dim_coords_and_dims=dimension_list)
    return new_cube

def newcube2d(field,cube,units=None):
    '''
    create new 2D cube (slice) from supplied data and template 
    '''
    import iris
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points

    # creating new coordinates
    lat_coord_name = cube.coord('latitude').standard_name
    lat_coord_units = cube.coord('latitude').units
    lat_coord_system = cube.coord_system()
    lat_coord = iris.coords.DimCoord(latitude, standard_name=lat_coord_name,
                                 units=lat_coord_units, var_name=lat_coord_name,
                                 coord_system=lat_coord_system)
    lat_coord.guess_bounds()

    lon_coord_name = cube.coord('longitude').standard_name
    lon_coord_units = cube.coord('longitude').units
    lon_coord_system = cube.coord_system()
    lon_coord = iris.coords.DimCoord(longitude, standard_name=lon_coord_name,
                                 units=lon_coord_units, var_name=lon_coord_name,
                                 coord_system=lon_coord_system, circular=True)
    lon_coord.guess_bounds()

    dimension_list = [(lat_coord, 0), (lon_coord, 1)]
    new_cube = iris.cube.Cube(field, standard_name=cube.standard_name, units=units,
                          dim_coords_and_dims=dimension_list)
    return new_cube

def calc_f(cube):
    '''
    Calculate Coriolis parameter from Iris cube
    '''
    import iris 
    lats = cube.coord('latitude').points
    lons = cube.coord('longitude').points
    lon2d, lat2d = np.meshgrid(lons,lats)
    omega = 2*np.pi / 86400 # 1/s 
    f = 2. * omega * np.sin(np.deg2rad(lat2d))
    f[f == 0] = 0.00002
    return f 

def exner_to_pres(exner):
    '''
    calculate the pressure from the Exner function 
    https://github.com/LSaffin/iris-extensions/blob/0fad44d97b9c363306425e87545fa5cfb75d1af9/irise/variable.py
    '''
    c_p = 1005.0
    R_d = 287.05
    p_sfc = 100000
    pres = p_sfc * np.power( exner, (c_p / R_d))
    return pres 

def exner(pres): 
    '''
    '''
    c_p = 1005.0
    R_d = 287.05
    p_sfc = 100000
    exner = np.power( (pres / p_sfc), (R_d / c_p) )
    return exner

# def geo_wind(u, v, pres, rho, f):
#     '''
#     Calculate geostrophic wind components from full horizontal wind 
#     https://github.com/WimUU/Climate-Physics/blob/
#     38d30eab9691820fe20e457e6f8a3030d08e1f7e/Simulation%20of%20Ocean%20Atmosphere%20and%20Climate/
#     Code_Exercise_1.py
#     '''
#     import iris 
#     from windspharm.iris import VectorWind
#     #w = VectorWind(u,v)
#     #px, py = w.gradient(pres, truncation=21)
#     # read in pressure data
#     pres = pres.regrid(u,iris.analysis.Linear())
#     # calculate gradients using Windspharm
#     w = VectorWind(u,v)
#     px0, py0 = w.gradient(pres, truncation=21) 
#     # set constants
#     re = 6371000
#     pi = 3.14159265
#     # calculate gradient using built-in function
#     xn = pres.coord('longitude').points * (pi / 180.)
#     yt = pres.coord('latitude').points * (pi / 180.)
#     # calculate spacing in radian between grid points 
#     nx = pres.shape[1]
#     ny = pres.shape[0]
#     dlon = 4. * pi / nx
#     dlat = 2. * pi / ny
#     # calculate cosine of latitude
#     coslat = np.cos(yt)
#     coslat = coslat.reshape(1, ny, 1)
#     # call function to calculate gradient
#     px = ddlambda_2d(pres.data, re, dlon, coslat)
#     py = ddphi_2d(pres.data, re, dlat)
#     # reshape arrays
#     px = px.reshape(ny,nx)
#     py = py.reshape(ny,nx)
#     # create new Iris cubes containing pressure gradient data
#     pgf_v = pres.copy(data=px)
#     pgf_u = pres.copy(data=py)
#     # define each coriolis term
#     cor_u = (-1/(f * rho))
#     cor_v = (1/(f * rho))
#     # calculate geostrophic wind components 
#     ug = pgf_u
#     ug.data = cor_u * pgf_u.data
#     vg = pgf_v
#     vg.data = cor_v * pgf_v.data

#     ug.units = 'm s**-1'
#     vg.units = 'm s**-1'
#     return ug, vg, pgf_u, pgf_v, cor_u, cor_v

# def geo_wind_new():
#     '''
#     copy of part of 'diagnostic_plotter.py' used to calculate geostrophic wind 
#     '''
#     # regrid pressure to same grid as wind components                      
#     pres = pres.regrid(ut[lev],iris.analysis.Linear())
#     # set constants                                                        
#     r_e = 6371000.
#     pi = 3.14159265
#     # get grid in radians                                                  
#     xn = pres.coord('longitude').points * (pi / 180.)
#     yt = pres.coord('latitude').points * (pi / 180.)
#     # calculate spacing in radians between grid points                     
#     nx = pres.shape[1]
#     ny = pres.shape[0]
#     dlon = 4. * pi / nx
#     dlat = 2. * pi / ny
#     # calculate cosine of latitude                                         
#     coslat = np.cos(yt)
#     coslat = coslat.reshape(1, ny, 1)
#     # call function to calculate gradient                                  
#     px0 = SG.ddlambda_2d(pres.data, r_e, dlon, coslat)
#     py0 = SG.ddphi_2d(pres.data, r_e, dlat)
#     # reshape arrays                                                       
#     px0 = px0.reshape(ny,nx)
#     py0 = py0.reshape(ny,nx)
#     # create new Iris cubes containing pressure gradient data              
#     pgf_u = pres.copy(data=px0)
#     pgf_v = pres.copy(data=py0)
#     # calculate Coriolis term in geostrophic wind components               
#     f0    = calc_f(ut[lev])
#     f0[f0 == 0] = 0.00002
#     f     = ut[lev].copy(data=f0)
#     f2    = imath.exponentiate(f,2)
#     cor_v = imath.divide(f,f2*rho[lev])
#     cor_u = imath.multiply(imath.divide(f,f2*rho[lev]),-1)
#     # calculate geostrophic wind components                                
#     ug = pgf_u * cor_u; vg = pgf_v * cor_v
#     # add metadata (units)                                                 
#     ug.units = 'm s**-1'; vg.units = 'm s**-1'

def reverse_lat(u, v, axis=0):
    '''
    Reverse the order of a lat/lon array 
    '''
    slicelist = [slice(0, None)] * u.ndim
    slicelist[axis] = slice(None, None, -1)
    u = u.copy()[slicelist]
    v = v.copy()[slicelist]
    return u, v 

def order_lat(latdim, u, v, axis=0):
    '''
    Call function above to reverse order of lat/lon array
    '''
    latdim = latdim.copy()
    if latdim[0] < latdim[-1]:
        latdim = latdim[::-1]
        u, v = reverse_lat(u, v, axis=axis)
    else:
        u, v = u.copy(), v.copy()
    return latdim, u, v 


