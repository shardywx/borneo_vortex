"""
(C) British Crown Copyright 2017 Met Office.
"""

def diagnostic_core(exner_cube, theta_cube, thetavd_cube, rhod_cube,
                    q_cube, rh_cube, km_cube, dbdz_cube, tinc_cube, uinc_cube, vinc_cube,
                    lcl_cube, cloud, heating):
    '''
    This is an edited version of the original diagnostic_core function. Edits
    include updated geotriptic and ageotriptic solvers used in the tropics. 

    Calculates geostrophic pressure tendencies from model exner, theta, rhod
    and thetavd fields. Input fields are assumed to be global on model levels
    at a horizontal resolution appropriate for a semi-geostrophic diagnostic
    (typically 160x120). Calculates consistent tendencies of geostrophic winds
    and hydrostatic thetav. Also calculates ageostrophic winds and vertical
    velocity required to maintain geostrophic and hydrostatic balance.  
    
    

    All input cubes assumed to have 3 dimensions (model_level, latitude,
    longitude) and assumed to have altitude as an auxiliary coordinate.

    Standard values assumed for physical constants.

    Can use moist stability to compute response to forcing (cloud>0) or
    artificial heating instead of model forcing (heating>0). If cloud=2,
    use moist stability from boundary layer scheme.
    
    Changes July 2020:
    implementation of boundary layer solve corrected. Definition of tropical
    filtering region rationalised, but no change in results.
    
    Changes September 2020:
    Thermal dynamics forcing enabled in Calc_G2. More selective filtering of
    Qnn coefficients, forcing fields and output using tropicalfilter_grad. 
    Regularisation of Qnn coefficients not applied to rhs calculations.

    Parameters
    ----------
    exner_cube : iris.cube.Cube
        Cube containing exner data
    theta_cube : iris.cube.Cube
        Cube containing theta data
    thetavd_cube : iris.cube.Cube
        Cube containing thetavd data
    rhod : iris.cube.Cube
        Cube containing rhod data
    q : iris.cube.Cube
        Cube containing q data
    rh : iris.cube.Cube
        Cube containing relative humidity data
    km : iris.cube.Cube
        Cube containing boundary layer momentum mixing coefficients
    dbdz : iris.cube.Cube
        Cube containing moist vertical stability used by boundary layer scheme
    tinc : iris.cube.Cube
        Cube containing forcing terms for temperature in deg K sec^{-1}
    uinc : iris.cube.Cube
        Cube containing forcing terms for zonal wind in m sec^{-2}
    vinc : iris.cube.Cube
        Cube containing forcing terms for meridional wind in m sec^{-2}
    lcl : iris.cube.Cube
        Cube containing lifting condensation level
    cloud : integer indicating whether dry or moist stability is to be used.
    heating : integer indicating whether only the external increments are to
        be used heating is to be used.

    Returns
    -------
    dpidt : iris.cube.Cube
        Cube containing time tendency of exner in sec^{-1}
    dugdt : iris.cube.Cube
        Cube containing time tendency of ug in sec^{-1}
    dvgdt : iris.cube.Cube
        Cube containing time tendency of vg in sec^{-1}
    dthetavdt : iris.cube.Cube
        Cube containing time tendency of thetav in sec^{-1}
    u : iris.cube.Cube
        Cube containing uag
    v : iris.cube.Cube
        Cube containing vag
    w : iris.cube.Cube
        Cube containing w
    balheat1 : iris.cube.Cube
        Cube containing w*(drystability-moiststability) (cloud=2 only)
    balheat2 : iris.cube.Cube
        Cube containing difference in vertical component of forcing matrix G
        to allow for convection (cloud=2 only)
    '''

    import iris
    import numpy as np
    import scipy.stats as sci
    import diagnosticSGfunctions as SG
    import matplotlib.pyplot as plt

    tropicalfilter = SG.tropicalfilter
    tropicalfilter_grad = SG.tropicalfilter_grad

    # Extract data
    exner = exner_cube.data
    theta = theta_cube.data
    thetavd = thetavd_cube.data
    rhod = rhod_cube.data
    q = q_cube.data
    rh = rh_cube.data
    km = km_cube.data
    dbdz = dbdz_cube.data
    tinc = tinc_cube.data
    uinc = uinc_cube.data
    vinc = vinc_cube.data
    lcl = lcl_cube.data
    # Regularise km. km[0] is the surface exchange coefficient
    nbl=km.shape[0]
    km[0, ]=np.maximum(km[0, ],1.e-3)
    km[1:nbl-1, ]=np.maximum(km[1:nbl-1, ],0.4)
    km = np.where(km > 1.e2, 1.e2, km)
    # Smooth km
    kmsmooth = SG.smooth(km)
    km = SG.smooth(kmsmooth)
    # Smooth dbdz
    dbdzsmooth = SG.smooth(dbdz)
    dbdz = SG.smooth(dbdzsmooth)
    # Set up vertical information for calculations
    theta_height = thetavd_cube.coord('altitude').points
    height = exner_cube.coord('altitude').points
    nx = height.shape[2]
    ny = height.shape[1]
    nlevels = height.shape[0]
    thick = theta_height[1: nlevels, ] - theta_height[0: nlevels-1, ] + 0.001
    # Convert temperature increment to theta increment and smooth
    exneru = np.roll(exner, -1, 0)
    exneru[nlevels-1, ] = exner[nlevels-1, ] - .0001
    exnerd = np.roll(exner, 1, 0)
    exnerd[0, ] = exner[0, ] + .0001
    heatsmooth = SG.smooth(2. * tinc / (exner + exneru))
    heat = SG.smooth(heatsmooth)
    # Set up horizontal information
    pi = 3.14159265
    cpd = 1005.
    a = 6371000.
    g = 9.81
    kappa = 287.05 / cpd
    twoomega = 1.4584E-4
    beta = twoomega / a

    latitude = thetavd_cube.coord('latitude').points * pi / 180.
    longitude = thetavd_cube.coord('longitude').points * pi / 180.

    # Define tropical latitudes, and define boundary layer height (dependent on Km).
    lat0 = 8 # outer latitude of tropical belt  
    lat1 = 4 # inner latitude of tropical belt  
    lat_tropics_outer, lat_tropics = SG.lat_tropics(latitude,longitude,lat0,lat1)
    # calculate boundary layer height --> level at which kmh drops below a threshold value (diagnosticSGfunctions)
    BLH = SG.blh(km)

    coslat = np.cos(latitude)
    coslat=coslat.reshape(1, ny, 1)
    coriolis = twoomega * np.sin(latitude)
    coriolis = coriolis.reshape(1, ny, 1)
    tanlat = np.tan(latitude)
    tanlat = tanlat.reshape(1, ny, 1)
    beta = twoomega * np.cos(latitude) / a
    beta = beta.reshape(1, ny, 1)
    coriolis2 = coriolis * coriolis
    # Minimum value of f appropriate to 4 deg latitude.
    coriolis0 = np.where(coriolis2 < 1.e-10, 1.E-5, coriolis) 
    coriolis0 = np.copysign(coriolis0, coriolis)
    coriolis20 = coriolis0 * coriolis0
    # Calculate latitude/longitude increments for central differencing
    dlong = 4. * pi / nx
    dlat = 2. * pi / ny
    drdlambda = SG.ddlambda(height, a, dlong, coslat)
    drdphi = SG.ddphi(height, a, dlat)
    # Calculate vertical grid information
    heightu = np.roll(height, -1, 0)
    heightu[nlevels-1, ] = 2.*height[nlevels-1, ] -height[nlevels-2, ]
    heightd = np.roll(height, 1, 0)
    heightd[0, ] = height[0, ] - .001
    # Assume eta value=level number for all model layers. UM eta values not
    # needed.
    # Calculate deta/dr
    thick2level = 1. / (heightu - heightd)
    # Calculate pressure gradients
    dpidlambda = SG.ddlambda(exner, a, dlong, coslat)
    dpidphi = SG.ddphi(exner, a, dlat)
    # Calculate thetav from dpi/dr
    thetav = -g / (cpd * SG.ddr(exner, thick2level))

    # Calculate geostrophic winds using (40)
    (thetae, thetaw, thetan, thetas) = SG.fieldshift(thetav)
    thetadpdlambda = dpidlambda * cpd * 0.5 * (thetae + thetaw) + g * drdlambda
    thetadpdphi = dpidphi * cpd * 0.5 * (thetan + thetas) + g * drdphi

    # For outputting horizontal gradients of pressure and temperature.
    # Add code to create output cubes if required
    t1 = thetadpdlambda / (0.5 * (thetae + thetaw)); t2 = thetadpdphi / (0.5 * (thetan + thetas))
    dthetadlambda = g * SG.ddr(t1, thick2level) / ((cpd * SG.ddr(exner, thick2level)) ** 2)
    dthetadphi = g * SG.ddr(t2, thick2level) / ((cpd * SG.ddr(exner, thick2level)) ** 2)

    vg = thetadpdlambda / coriolis0  # Use this ug and vg as a BC
    ug = -thetadpdphi / coriolis0
    SG.normvector(ug, vg, vg, 'norms of ug, vg')
    # save the geostrophic wind components for output later (18/08/20)
    ug_um = ug.copy(); vg_um = vg.copy()
    # Define forcing terms for the geotriptic tropical solver. Perform this here before filtering.
    rhs3u = thetadpdlambda[0:nbl,].copy() 
    rhs3v = thetadpdphi[0:nbl,].copy()

    # Smooth ug vg
    # Smooth exner derivatives
    Qsm = tropicalfilter_grad(ug, latitude) # Use gradient filter.
    ug = Qsm
    Qsm = tropicalfilter_grad(vg, latitude)
    vg = Qsm
    Qsm = tropicalfilter_grad(thetadpdlambda, latitude)
    thetadpdlambda = Qsm
    SG.normvector(ug, vg, vg, 'norms of ug, vg')
    Qsm = tropicalfilter_grad(thetav, latitude)
    thetav = Qsm
    Qsm = tropicalfilter_grad(theta, latitude)
    theta = Qsm
    Qsm = tropicalfilter_grad(thetavd, latitude)
    thetavd= Qsm
    (thetae, thetaw, thetan, thetas) = SG.fieldshift(thetav)

    # calculate flux coefficients km/dr
    kmdz = np.empty((nbl + 1, ny, nx))
    kmdz[0: nbl, :, :] = km[0: nbl, :, :]
    kmdz[nbl, :, :] = kmdz[nbl-1, :, :]
    # km[0]*delta u already defines a momentum flux
    kmdz[1: nbl, ] = kmdz[1: nbl, ] / (heightu[0: nbl-1, ] - height[0: nbl-1, ])
    kmdz[nbl, :, :] = kmdz[nbl-1, :, :]
    # Calculate Ekman balanced winds using (50) and (51)
    # Set up discrete version of friction operator
    updiffcoeff = np.zeros((nbl, ny, nx))
    downdiffcoeff = np.zeros((nbl, ny, nx))
    diagdiffcoeff = np.zeros((nbl, ny, nx))
    updiffcoeff[0: nbl-1, ] = kmdz[1: nbl, ] / thick[0: nbl-1, ]
    downdiffcoeff[0: nbl, ] = kmdz[0: nbl, ] / thick[0: nbl, ]
    diagdiffcoeff = -(updiffcoeff + downdiffcoeff)
    # Calculate coefficients for pentadiagonal solve of (51)
    upupcoeff = np.zeros((nbl, ny, nx))
    upcoeff = np.zeros((nbl, ny, nx))
    diagcoeff = np.zeros((nbl, ny, nx))
    downcoeff = np.zeros((nbl, ny, nx))
    downdowncoeff = np.zeros((nbl, ny, nx))
    upupcoeff[0: nbl-2, ] = kmdz[1: nbl-1, ] * updiffcoeff[1: nbl-1, ] / thick[0: nbl-2, ]
    upcoeff[0: nbl-2, ] = ((kmdz[1: nbl-1, ]
                           * (diagdiffcoeff[1: nbl-1, ] - updiffcoeff[0: nbl-2, ])
                           - kmdz[0: nbl-2, ] * updiffcoeff[0: nbl-2, ]) / thick[0: nbl-2, ])
    upcoeff[nbl - 2, ] = ((kmdz[nbl-1, ] * diagdiffcoeff[nbl-1, ]
                          - kmdz[nbl-2, ] * updiffcoeff[nbl-2, ]) / thick[nbl-2, ])
    # Store downcoeff in locations [0:nbl-2] though applied at [1:nbl-1]
    downcoeff[0: nbl-1, ] = -((kmdz[2: nbl+1, ] * downdiffcoeff[1: nbl, ]
                               + kmdz[1: nbl, ] * (downdiffcoeff[1: nbl, ]
                               - diagdiffcoeff[0: nbl-1, ])) / thick[1: nbl, ])
    # Store downdowncoeff in locations [0:nbl-3] though applied at [2:nbl-1]
    downdowncoeff[0: nbl-2, ] = kmdz[2: nbl, ] * downdiffcoeff[1: nbl-1, ] / thick[2: nbl, ]
    diagcoeff[2: nbl, ] = -(upupcoeff[2: nbl, ] + upcoeff[2: nbl, ] + downcoeff[1: nbl-1, ]
                            + downdowncoeff[0: nbl-2, ])
    diagcoeff[1, ] = ((kmdz[2, ] * (downdiffcoeff[2, ] - diagdiffcoeff[1, ])
                      - kmdz[1, ] * (diagdiffcoeff[1, ] - updiffcoeff[0, ])) / thick[1, ])
    diagcoeff[0, ] = ((kmdz[1, ] * (downdiffcoeff[1, ] - diagdiffcoeff[0, ])
                      - kmdz[0, ] * diagdiffcoeff[0, ]) / thick[0, ])
    diagcoeff_ageo = diagcoeff.copy()
    # Complete calculation of diagcoeff for (51)
    diagcoeff[0: nbl, ] = diagcoeff[0: nbl,] + coriolis20
    # calculate rhs of (51). Calculation is in boundary layer only
    rhs = np.zeros((nbl, ny, nx))
    upwork = upcoeff.copy()
    diagwork = diagcoeff.copy()
    downwork = downcoeff.copy()
    rhs[0: nbl, ] = ug[0: nbl, ] * coriolis20
    rhs = rhs + SG.vertdiff(thetadpdlambda, updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl)
    
    ''' 
    Solve (51) for ue and back substitute in (50) to calculate ve (use filtered geostrophic winds to calculate geotriptic winds)  
    STEP 3       
    pentasolve: apply inverse of pentadiagonal matrix to input data (input data destroyed).    
    arg1: field to apply function to (3D np-array)  
    args2-->6: matrix coefficients (3D np-arrays) 
    ''' 
    lhs = SG.pentasolve(rhs, upupcoeff, upwork, diagwork, downwork, downdowncoeff)
    ug[0: nbl, ] = lhs[0: nbl, ]
    vg = thetadpdlambda.copy()
    ''' 
    vertdiff --> apply vertical diffusion operator (defined by an input tridiagonal matrix)
    arg1: field to apply function to (3D np-array)
    args2-4: matrix coefficients (3D np-arrays)
    arg5: number of levels over which metric applied (scalar) 
    ''' 
    vg[0: nbl, ] = vg[0: nbl, ] - SG.vertdiff(ug, updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl)
    vg = vg / coriolis0
    # calculate and print norm of input vector --> unsure why the 'vg' component is there twice 
    SG.normvector(ug, vg, vg, 'norms of ue, ve')
    ug0 = ug.copy(); vg0 = vg.copy()
    '''
    trisolve_blh --> tridiagonal solve of eq(10), with variable boundary layer height
    input geostrophic wind --> output geotriptic wind
    km: BL mixing coefficients (kmh, kmv)
    thick: level thickness for finite differencing (?)
    rhs3u/rhs3v: RHS forcing terms (u/v components)
    lat_tropics_outer: outer latitude corresponding to the tropical filter (calculated elsewhere in diagnosticSGfunctions)
    ug: geostrophic wind
    BLH:
    '''
    ug_tmp = SG.trisolve_blh(km,thick,rhs3u,lat_tropics_outer,ug,BLH)
    '''                                                              
    merge --> merge two fields within a given latitudinal range, both N and S of the equator
    ug0: extratropical field
    ug_tmp: tropical field
    latitude: 1D array of latitude points
    10,18: latitudinal range in which extratropical + tropical fields are merged
    IF STEPS ARE CORRECT, EXTRATROPICAL + TROPICAL FIELDS SHOULD NOW BE IDENTICAL (no filter applied)
    '''
    #ug = SG.merge(ug0,ug_tmp,latitude,lat_tropics,lat_tropics_outer)
    ug = SG.merge(ug0,ug_tmp,latitude,lat1,lat0)
    '''
    repeat above steps for 'v' component of the geostrophic wind
    '''
    vg_tmp = SG.trisolve_blh(km,thick,rhs3v,lat_tropics_outer,vg,BLH)
    #vg = SG.merge(vg0,vg_tmp,latitude,lat_tropics,lat_tropics_outer)
    vg = SG.merge(vg0,vg_tmp,latitude,lat1,lat0)

    # set up PV matrix coefficients, eq. (65)
    drystability = (cpd * thetav * ((exneru - exner) / (heightu - height)
                    - (exner - exnerd) / (height - heightd)) * 2. * thick2level)
    # Smooth stability
    drystabsmooth = SG.smooth(drystability)
    drystability = SG.smooth(drystabsmooth)
    drystability = np.where(drystability < 1.e-5, 1.e-5, drystability)
    drystability[0, ] = drystability[1, ]
    Q33 = drystability.copy()
    thetEstar=thetavd.copy()
    if cloud > 0:
        # Replace dry stability by moist stability
        thetEstar = SG.thetaEstar(exner, theta, q, rh)
        # Calculate information for convective spreading of forcing
        contop = SG.contop(lcl,thetEstar)
        for j in range(0,ny):
            for i in range(0,nx):
                k=int(lcl[j,i])
                Q33[k-1:nbl,j,i] = dbdz[k-1:nbl,j,i]
        Q33 = np.where(drystability < Q33, drystability, Q33)
    Q33 = np.where(Q33 < 1.e-5, 1.e-5, Q33)
    # calculate first row of PV matrix
    vgbytheta = 2. * vg / (thetae + thetaw)
    ugbytheta = 2. * ug / (thetan + thetas)
    Q11 = SG.ddlambda(vgbytheta, a, dlong, coslat) * thetav
    Q12 = SG.ddphi(vgbytheta, a, dlat) * thetav
    Q13 = thetav * SG.ddr(vgbytheta, thick2level)
    Q11 = coriolis * (Q11 + ug * tanlat / a) + coriolis20
    Q12 = coriolis * Q12
    Q13 = coriolis * Q13
    # filter tropical information
    Qsm = Q12.copy()
    Q12 = tropicalfilter_grad(Qsm, latitude)
    Qsm = Q13.copy()
    Q13 = tropicalfilter_grad(Qsm, latitude)
    # calculate second row of PV matrix
    Q21 = -SG.ddlambda(ugbytheta, a, dlong, coslat) * thetav
    Q22 = -SG.ddphi(ugbytheta, a, dlat) * thetav
    Q23 = -thetav * SG.ddr(ugbytheta, thick2level)
    Q21 = coriolis * (Q21 + vg * tanlat / a)
    Q22 = (coriolis * Q22 - beta * cpd * dpidphi / coriolis0 + coriolis20)
    Q23 = coriolis * Q23
    # filter tropical information
    Qsm = Q21.copy()
    Q21 = tropicalfilter_grad(Qsm, latitude)
    Qsm = Q23.copy()
    Q23 = tropicalfilter_grad(Qsm, latitude)
    # calculate remainder of third row
    Q31 = g * SG.ddlambda(thetav, a, dlong, coslat) / thetav
    Q32 = g * SG.ddphi(thetav, a, dlat) / thetav

    # filter tropical information
    Qsm = Q31.copy()
    Q31 = tropicalfilter_grad(Qsm, latitude)
    Qsm = Q32.copy()
    Q32 = tropicalfilter_grad(Qsm, latitude)
    # calculate reduced Q matrix used in calculating G eq. (62), 
    # avoiding regularisation below
    Q11R = Q11.copy()
    Q22R = Q22.copy()
    Q11R = Q11R - coriolis20
    Q22R = Q22R - coriolis20
    Q12R = Q12.copy()
    Q21R = Q21.copy()
    Q31R = Q31.copy()
    Q32R = Q32.copy()
    # Modify PV matrix to ensure positive definiteness
    Q11 = np.where(Q11 < 0.1 * coriolis20, 0.1 * coriolis20, Q11)
    Q22 = np.where(Q22 < 0.1 * coriolis20, 0.1 * coriolis20, Q22)
    Qsm = Q11.copy()
    Q11 = tropicalfilter(Qsm, latitude)
    Qsm = Q22.copy()
    Q22 = tropicalfilter(Qsm, latitude)
    limit = np.empty((nlevels, ny, nx))
    det = 0.3 * np.sqrt(Q22 * Q33)
    limit = np.copysign(det, Q23)
    Q23 = np.where(np.abs(Q23) > det, limit, Q23)
    limit = np.copysign(det, Q32)
    Q32 = np.where(np.abs(Q32) > det, limit, Q32)
    det = 0.3 * np.sqrt(Q33 * Q11)
    limit = np.copysign(det, Q13)
    Q13 = np.where(np.abs(Q13) > det, limit, Q13)
    limit = np.copysign(det, Q31)
    Q31 = np.where(np.abs(Q31) > det, limit, Q31)
    det = 0.3 * np.sqrt(Q11 * Q22)
    limit = np.copysign(det, Q12)
    Q12 = np.where(np.abs(Q12) > det, limit, Q12)
    limit = np.copysign(det, Q21)
    Q21 = np.where(np.abs(Q21) > det, limit, Q21)
    # Calculate G, eq. (62)
    (G1, G2, G3) = SG.CalcG(Q11R, Q12R, Q21R, Q22R, Q31R, Q32R,
                            ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                            updiffcoeff, diagdiffcoeff, downdiffcoeff)
    G3old=G3.copy()
    if cloud > 0:
        #  Use dry stability if G3<0
        Q33 = np.where(G3 < 0., drystability, Q33) 
        # Spread G3 in vertical
        G3BL = np.zeros((ny, nx))
        for j in range(0,ny):
            for i in range(0,nx):
                k=int(lcl[j,i])
                G3BL[j,i]=G3[k-2,j,i]
                if rh[k,j,i] > 70.:
                    if G3BL[j,i] > 0.:
                        G3[k-1:int(contop[j,i]),j,i]=G3BL[j,i]+G3[k-1:int(contop[j,i]),j,i]
                        Q33[k-1:int(contop[j,i]),j,i]=drystability[k-1:int(contop[j,i]),j,i]
        Q33 = np.where(Q33 < 1.e-5, 1.e-5, Q33)
        Q33[0, ] = Q33[1, ]
    G3old=G3-G3old
    # Use if only the response to forcing increments required
    if heating > 0:
        (G1, G2, G3) = SG.CalcGinc(ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                            updiffcoeff, diagdiffcoeff, downdiffcoeff)
    # Calculate P_2^{-1 T}
    det = (Q11 * (Q22 * Q33 - Q23 * Q32) - Q12 * (Q21 * Q33 - Q23 * Q31)
           + Q13 * (Q21 * Q32 - Q22 * Q31))
    detmin = det.min()
    detmin = 1. / detmin
    QI11 = (Q22 * Q33 - Q23 * Q32) / det
    QI21 = -(Q21 * Q33 - Q23 * Q31) / det
    QI31 = (Q21 * Q32 - Q22 * Q31) / det
    QI12 = -(Q12 * Q33 - Q13 * Q32) / det
    QI22 = (Q11 * Q33 - Q13 * Q31) / det
    QI32 = -(Q11 * Q32 - Q12 * Q31) / det
    QI13 = (Q12 * Q23 - Q13 * Q22) / det
    QI23 = -(Q11 * Q23 - Q13 * Q21) / det
    QI33 = (Q11 * Q22 - Q12 * Q21) / det

# calculate P^{-1}G
    (B1, B2, B3) = SG.Pinverse(G1, G2, G3,
                               QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33,
                               nbl, ny, coriolis20,
                               upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff)
    # Include ue,ve in divergence calculation on rhs of (67)
    # Remove if only the response to artificial heating required
    if heating <= 0:
        B1 = B1 + ug
        B2 = B2 + vg
    # Multiply by R
    B3 = B3 - drdlambda * B1
    B3 = (B3 - drdphi * B2) * thick2level
    # Remove B3 at bottom level from divergence calculation, 
    # cancelled by Neumann boundary condition
    B3[0, ] = 0.
    # Apply scaling factors
    B1 = B1 * a ** 2 * rhod
    B2 = B2 * a ** 2 * rhod
    B3 = B3 * a ** 2 * rhod
    # calculate divergence to give rhs of (67)
    divergence = 1.E-20 * SG.div(B1, B2, B3, a, coslat, dlong, dlat, thick2level)
    zerocoeff = 1.e-20 * (kappa - 1.) * rhod * a ** 2 / (kappa * exner * thick2level)
    firstcoeff = 1.e-20 * rhod * a ** 2 * cpd * thetav ** 2 / (g * thetavd)
    secondcoeff = 1.e-20 * rhod * a ** 2 * cpd * thetav
    # Vertical preconditioning, section 5.1. Set up eq. (77)
    QI3 = QI33 * QI33
    for k in range(0, nlevels):
        QI3[k] = np.sqrt(QI3[k, ].sum() / QI3[k, ].size)
    diagterm = secondcoeff * QI3
    upterm = np.roll(diagterm, -1, 0)
    upterm[nlevels-1, ] = diagterm[nlevels-1, ]
    downterm = np.roll(diagterm, 1, 0)
    downterm[0, ] = diagterm[0, ]
    uptricoeff = upterm * thick2level
    downtricoeff = downterm * thick2level
    # Add horizontal term in (73)
    diagtricoeff = -4. * QI22 / (a * dlat) ** 2
    diagtricoeff = (diagtricoeff - 4. * QI11 / (a * dlong * coslat) ** 2)
    diagtricoeff = zerocoeff * 1.e2 + diagtricoeff * secondcoeff / thick2level
    # The value 1.e2 is tuneable
    diagtricoeff[0: nlevels-1, ] = diagtricoeff[0: nlevels-1, ] - uptricoeff[0: nlevels-1, ]
    diagtricoeff[1: nlevels, ] = diagtricoeff[1: nlevels, ] - downtricoeff[1: nlevels, ]
    diagtricoeff = 1.5 * diagtricoeff
    # The value 1.5 is tuneable
    # Start outer iteration, given (B1,B2,B3)
    # Code assumes zero initial guess for dpidt
    dpidt = np.zeros((nlevels, ny, nx))
    Adpidt = np.zeros((nlevels, ny, nx))
    outer = 10
    for outeriter in range(0, outer):
        # soln holds dpidt increment at each iteration
        soln = np.zeros((nlevels, ny, nx))
        residual = divergence - Adpidt
        rhs = SG.smooth(residual)
        diagwork = diagtricoeff.copy()
        lhs = SG.trisolve(rhs, uptricoeff, diagwork, downtricoeff)
        p = SG.smooth(lhs)
        # GCR iteration,solution stored in dpidt, p holds updates,
        # r holds preconditioned residual
        r = p.copy()
        Ap = p.copy()
        SG.normvector(divergence, residual, Adpidt, 'norms of divergence, residual, Adpidt')
        niter = 25
        beta = 1.
        rAr0 = 1.
        for iter in range(0, niter):
            # Apply elliptic operator defined in (74) to r
            Ar = zerocoeff * r
            A3 = SG.ddr(r, thick2level)
            A1 = SG.ddlambda(r, a, dlong, coslat) - A3 * drdlambda
            A2 = SG.ddphi(r, a, dlat) - A3 * drdphi
            Ar = Ar + firstcoeff * A3
            (B1, B2, B3) = SG.Pinverse(A1, A2, A3,
                                       QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33,
                                       nbl, ny, coriolis20,
                                       upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff)
            # Multiply by R
            B3 = B3 - drdlambda * B1
            B3 = (B3 - drdphi * B2) * thick2level
            # Remove B3 at bottom level from divergence calculation, 
            # cancelled by Neumann boundary condition
            B3[0, ] = 0. 
	    # Apply scaling factors
            B1 = B1 * secondcoeff
            B2 = B2 * secondcoeff
            B3 = B3 * secondcoeff
            Ar = Ar + 1.2 * SG.div(B1, B2, B3, a, coslat, dlong, dlat, thick2level)
            # The constant 1.2 is tuneable
            # update beta
            if iter > 0:
                rAr0 = rAr.copy()
            rAr = np.sum(r * Ar)
            beta = rAr / rAr0
            if iter == 0:
                beta = 0.
            # Update p and Ap, no update on first iteration as beta=0
            p = r + beta * p
            Ap = Ar + beta * Ap
            # precondition Ap
            MAp = Ap.copy()
            rhs = SG.smooth(MAp)
            diagwork = diagtricoeff.copy()
            lhs = SG.trisolve(rhs, uptricoeff, diagwork, downtricoeff)
            MAp = SG.smooth(lhs)
            # Calculate alpha
            alpha = np.sum(Ap * MAp)
            alpha = rAr / alpha
            # Update soln and r
            soln = soln + alpha * p
            r = r - alpha * MAp
        # End GCR iteration
        # update dpidt
        dpidt = dpidt + soln
        SG.normvector(soln, dpidt, dpidt, ' soln,dpidt')
        # Calculate ageostrophic velocity fields (u-ug,v-vg,etadot) using (63)
        # Calculate G using (60)
        (G1, G2, G3) = SG.CalcG(Q11R, Q12R, Q21R, Q22R, Q31R, Q32R,
                                ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                                updiffcoeff, diagdiffcoeff, downdiffcoeff)
        (G1_, G2_, G3_) = SG.CalcG2(Q11R, Q31R, Q32R,
                                ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                                updiffcoeff, diagdiffcoeff, downdiffcoeff)
        # Spread G3 in vertical
        if cloud > 0:
            for j in range(0,ny):
                for i in range(0,nx):
                    k=int(lcl[j,i])
                    if rh[k,j,i] > 70.:
                        if G3BL[j,i] > 0.:
                            G3[k-1:int(contop[j,i]),j,i]=G3BL[j,i]+G3[k-1:int(contop[j,i]),j,i]
                        if G3_[k-2,j,i] > 0.:
                            G3_[k-1:int(contop[j,i]),j,i]=G3_[k-2,j,i]+G3_[k-1:int(contop[j,i]),j,i]
      
        # Use if only the response to forcing increments required
        if heating > 0:
            (G1, G2, G3) = SG.CalcGinc(ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                            updiffcoeff, diagdiffcoeff, downdiffcoeff)
            (G1_, G2_, G3_) = SG.CalcGinc(ug, vg, g, thetav, uinc, vinc, heat, nbl, ny, coriolis,
                            updiffcoeff, diagdiffcoeff, downdiffcoeff)
        SG.normvector(G1, G2, G3, 'norms of G')
        # Subtract dpidt term
        A3 = SG.ddr(dpidt, thick2level)
        A1 = SG.ddlambda(dpidt, a, dlong, coslat) - A3 * drdlambda
        A2 = SG.ddphi(dpidt, a, dlat) - A3 * drdphi
        G3 = G3 - cpd * thetav * A3; G3_ = G3_ - cpd * thetav * A3
        G1 = G1 - cpd * thetav * A1; G1_ = G1_ - cpd * thetav * A1
        G2 = G2 - cpd * thetav * A2; G2_ = G2_ - cpd * thetav * A2
        # Calculate ageostrophic winds from eq. (63)
        (u, v, w) = SG.Pinverse(G1, G2, G3,
                                QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33,
                                nbl, ny, coriolis20,
                                upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff)

        #BLH=6*np.ones((ny,nx))
        # Edit G1_ and G2_ to include BC in updated ageotriptic winds.
        G1_, G2_ = SG.G_BC(G1_, G2_, u, v, upcoeff, upupcoeff, lat_tropics_outer, BLH)
        # Calculate ageostrophic winds in tropics. Update original wind.
        (u_tmp, v_tmp, w_tmp) = SG.Pinverse2(G1_, G2_, G3_, QI31, QI32, QI33, lat_tropics_outer,
                                nbl, ny, coriolis20, u, v, w,
                                upupcoeff, upcoeff, diagcoeff_ageo, downcoeff, downdowncoeff, BLH)
            
        # merge old and new u,v,w in some latitudinal range (was '10,18')
        u = SG.merge(u,u_tmp,latitude,lat1,lat0)
        v = SG.merge(v,v_tmp,latitude,lat1,lat0)
        w = SG.merge(w,w_tmp,latitude,lat1,lat0)

        # Remove w at bottom level, 
        # cancelled by Neumann boundary condition
        w[0, ] = 0.
        SG.normvector(u, v, w, 'norms of uag,vag,w')
        SG.normvector(G1, G2, G3, 'norms of G-dpidt')
        # Calculate residual in (67) for next outer iteration
        # Right hand side term is held in divergence
        # dpidt terms
        Adpidt = zerocoeff * dpidt
        Adpidt = Adpidt + firstcoeff * A3
        (B1, B2, B3) = SG.Pinverse(A1, A2, A3,
                                   QI11, QI12, QI13, QI21, QI22, QI23, QI31, QI32, QI33,
                                   nbl, ny, coriolis20,
                                   upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff)

        # Multiply B vector by R
        B3 = B3 - drdlambda * B1
        B3 = (B3 - drdphi * B2) * thick2level
        # Remove B3 at bottom level from divergence calculation, 
        # cancelled by Neumann boundary condition
        B3[0, ] = 0.
	# Apply scaling factors
        B1 = B1 * secondcoeff
        B2 = B2 * secondcoeff
        B3 = B3 * secondcoeff
        Adpidt = Adpidt + SG.div(B1, B2, B3, a, coslat, dlong, dlat, thick2level)

    # End outer iteration
    residual = divergence - Adpidt
    SG.normvector(divergence, residual, Adpidt, 'norms of divergence, residual, Adpidt')
    dugdt = -A2 * cpd * thetav / coriolis0
    dvgdt = A1 * cpd * thetav / coriolis0

    # call tropicalfilter again, before writing to Iris cubes 
    Qsm = tropicalfilter_grad(dpidt, latitude)
    dpidt = Qsm
    Qsm = tropicalfilter_grad(dugdt, latitude)
    dugdt = Qsm
    Qsm = tropicalfilter_grad(dvgdt, latitude)
    dvgdt = Qsm

    dthetavdt = g * SG.ddr(dpidt, thick2level) / (cpd * SG.ddr(exner, thick2level) ** 2)
    # Replace bottom level calculation by Neumann boundary condition
    dthetavdt[0, ] = heat[0, ]
    # Create cubes for output
    ug_cube = SG.newcube(ug, exner_cube)
    vg_cube = SG.newcube(vg, exner_cube)
    dpidt_cube = SG.newcube(dpidt, exner_cube)
    dugdt_cube = SG.newcube(dugdt, exner_cube)
    dvgdt_cube = SG.newcube(dvgdt, exner_cube)
    dthetavdt_cube = SG.newcube(dthetavdt, exner_cube)
    '''
    calculate updated, balanced geotriptic wind components
    is 'u' here the geotriptic wind, and 'ug' the geostrophic wind?
    or, is this the equivalent of updating the value of 'u' after the time-step?
    '''
    u = u + ug
    v = v + vg
    u_cube = SG.newcube(u, exner_cube)
    v_cube = SG.newcube(v, exner_cube)
    w_cube = SG.newcube(w, exner_cube)
    # output geostrophic wind components direct from the MetUM (18/08/20)
    ug_um_cube = SG.newcube(ug_um, exner_cube)
    vg_um_cube = SG.newcube(vg_um, exner_cube)
    # also output boundary layer height (10/11/20)
    exner_slice = exner_cube.extract(iris.Constraint(model_level_number=1))
    blh_cube    = SG.newcube2d(BLH, exner_slice)
    # additional outputs if cloud=2, these are zero if cloud=0.
    balheat1=w*(drystability-Q33)*thetav/g
    balheat2=G3old*thetav/g
    balheat1_cube = SG.newcube(balheat1, exner_cube)
    balheat2_cube = SG.newcube(balheat2, exner_cube)
    return ug_cube, vg_cube, dpidt_cube, dugdt_cube, dvgdt_cube, dthetavdt_cube,\
           u_cube, v_cube, w_cube, balheat1_cube, balheat2_cube, ug_um_cube, vg_um_cube, blh_cube
