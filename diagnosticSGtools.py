def fieldshift(field):
    import numpy as np
    ny=field.shape[1]
    fielde=np.roll(field,-1,2)
    fieldw=np.roll(field,1,2)
    fieldn=np.roll(field,-1,1)
    fieldn[:,ny-1,:] = field[:,ny-1,:] 
    fields=np.roll(field,1,1)
    fields[:,0,:] = field[:,0,:]
    return fielde, fieldw, fieldn, fields
def ddlambda(field, a, dlong, coslat):
    '''
    Differentiate input field with respect to lambda
    '''
    import numpy as np
    fielde=np.roll(field,-1,2)
    fieldw=np.roll(field,1,2)
    dfield=np.empty( (field.shape[0],field.shape[1],field.shape[2]) )
    for j in range(field.shape[1]):
      dfield[:,j,:] = (fielde[:,j,:]-fieldw[:,j,:])/(a*dlong*coslat[j])
    return dfield
def ddphi(field, a, dlat):
    '''
    Differentiate input field with respect to phi
    '''
    import numpy as np
    ny=field.shape[1]
    fieldn=np.roll(field,-1,1)
    fieldn[:,ny-1,:] = field[:,ny-1,:] 
    fields=np.roll(field,1,1)
    fields[:,0,:] = field[:,0,:]
    dfield = (fieldn-fields)/(a*dlat)
    return dfield
def ddr(field, thick2level):
    '''
    Differentiate input field with respect to r
    '''
    import numpy as np
    nlevels=field.shape[0]
    fieldu=np.roll(field,-1,0)
    fieldu[nlevels-1, ] = field[nlevels-1, ]
    fieldd=np.roll(field,1,0)
    fieldd[0, ] = field[0, ]
    dfield = (fieldu-fieldd)*thick2level
    return dfield
def ddeta(field):
    '''
    Differentiate input field with respect to eta
    '''
    import numpy as np
    nlevels=field.shape[0]
    fieldu=np.roll(field,-1,0)
    fieldu[nlevels-1, ] = field[nlevels-1, ]
    fieldd=np.roll(field,1,0)
    fieldd[0, ] = field[0, ]
    dfield = fieldu-fieldd
    return dfield
def vertdiff(field, updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl):
    '''
    Apply vertical diffusion operator to field
    '''
    import numpy as np
    dfield=np.empty( (nbl,field.shape[1],field.shape[2]) )
    dfield[1:nbl-1,]=(updiffcoeff[1:nbl-1, ]*field[2:nbl, ]+
    diagdiffcoeff[1:nbl-1, ]*field[1:nbl-1, ]+downdiffcoeff[1:nbl-1, ]*
    field[0:nbl-2, ])
    dfield[0, ]=updiffcoeff[0, ]*field[1, ]+diagdiffcoeff[0, ]*field[0, ]
    dfield[nbl-1, ]=(downdiffcoeff[nbl-1, ]*field[nbl-2, ]+
    diagdiffcoeff[nbl-1, ]*field[nbl-1, ])
    return dfield
def vert2diff(field, upupcoeff, upcoeff, diagcoeff, downcoeff, downdowncoeff, nbl):
    '''
    Apply vertical second diffusion operator and add coriolis squared*field
    to field
    Recall downcoeff stored in locations [0:nbl-2] though applied at [1:nbl-1]
    Similar for downdowncoeff
    '''
    import numpy as np
    dfield=np.empty( (nbl,field.shape[1],field.shape[2]) )
    dfield[2:nbl-2,]=(upupcoeff[2:nbl-2, ]*field[4:nbl, ]+upcoeff[2:nbl-2, ]*
    field[3:nbl-1, ]+diagcoeff[2:nbl-2, ]*field[2:nbl-2, ]+downcoeff[1:nbl-3, ]*
    field[1:nbl-3, ]+downdowncoeff[0:nbl-4, ]*field[0:nbl-4, ])
    dfield[0, ]=(upupcoeff[0, ]*field[2, ]+upcoeff[0, ]*
    field[1, ]+diagcoeff[0, ]*field[0, ])
    dfield[1, ]=(upupcoeff[1, ]*field[3, ]+upcoeff[1, ]*
    field[2, ]+diagcoeff[1, ]*field[1, ]+downcoeff[0, ]*field[0, ])
    dfield[nbl-2, ]=(upcoeff[nbl-2, ]*
    field[nbl-1, ]+diagcoeff[nbl-2, ]*field[nbl-2, ]+downcoeff[nbl-3, ]*
    field[nbl-3, ]+downdowncoeff[nbl-4, ]*field[nbl-4, ])
    dfield[nbl-1, ]=(diagcoeff[nbl-1, ]*field[nbl-1, ]+downcoeff[nbl-2, ]*
    field[nbl-2, ]+downdowncoeff[nbl-3, ]*field[nbl-3, ])
    return dfield
def trisolve(field,upcoeff,diagcoeff,downcoeff):
    '''
    Tridiagonal matrix solve. Input data destroyed.
    '''
    import numpy as np
    nlevels=field.shape[0]
    dfield=field.copy()
    factor = np.empty( (nlevels,field.shape[1],field.shape[2]) )
    diagcoeff[0, ]=1./diagcoeff[0, ]
    for k in range(1,nlevels): 
      factor[k, ] = downcoeff[k, ]*diagcoeff[k-1, ]
      diagcoeff[k, ] = 1./(diagcoeff[k, ]-factor[k, ]*upcoeff[k-1, ])
    for k in range(1,nlevels): 
      dfield[k, ]=dfield[k, ]-factor[k, ]*dfield[k-1, ]
    dfield[nlevels-1, ]=diagcoeff[nlevels-1, ]*dfield[nlevels-1, ]
    for k in range(nlevels-2,-1,-1):
      dfield[k, ]=diagcoeff[k, ]*(dfield[k, ]-upcoeff[k, ]*dfield[k+1, ])
    return(dfield)
def pentasolve(field,upupcoeff,upcoeff,diagcoeff,downcoeff,downdowncoeff):
    '''
    Pentadiagonal matrix solve. Input data destroyed.
    '''
    import numpy as np
    nbl=field.shape[0]
    dfield=np.empty( (nbl,field.shape[1],field.shape[2]) )
    for k in range(1,nbl-1):
      factor=downcoeff[k-1, ]/diagcoeff[k-1, ]
      diagcoeff[k, ]=diagcoeff[k, ]-upcoeff[k-1, ]*factor
      upcoeff[k, ]=upcoeff[k, ]-upupcoeff[k-1, ]*factor
      field[k, ]=field[k, ]-field[k-1, ]*factor
      factor=downdowncoeff[k-1, ]/diagcoeff[k-1, ]
      downcoeff[k, ]=downcoeff[k, ]-factor*upcoeff[k-1, ]
      diagcoeff[k+1, ]=diagcoeff[k+1, ]-factor*upupcoeff[k-1, ]
      field[k+1, ]=field[k+1, ]-factor*field[k-1, ]
    factor=downcoeff[nbl-2, ]/diagcoeff[nbl-2, ]
    diagcoeff[nbl-1, ]=diagcoeff[nbl-1, ]-factor*upcoeff[nbl-2, ]
    dfield[nbl-1, ]=(field[nbl-1, ]-factor*field[nbl-2, ])/diagcoeff[nbl-1, ]
    dfield[nbl-2, ]=(field[nbl-2, ]-upcoeff[nbl-2, ]*dfield[nbl-1, ])/diagcoeff[nbl-2, ]
    for k in range(nbl-3, -1,-1):
      dfield[k, ]=((field[k, ]-upupcoeff[k, ]*dfield[k+2, ]-upcoeff[k, ]*dfield[k+1])/
      diagcoeff[k, ])
    return dfield 
def tropicalfilter(field,latitude):
    '''
    rescale field in zonal direction in the tropics
    '''
    import numpy as np
    dfield=np.empty( (field.shape[0],field.shape[1],field.shape[2]) )
    ny=field.shape[1]
    nlevels=field.shape[0]
    sinlat=np.sin(latitude)
    for k in range(0,nlevels):
      for j in range(0,ny):
        zonalmean=field[k,j,].sum()/field[k,j,].size
        factor=1.
        sinsq=sinlat[j]**2
        if sinsq<0.1:
          factor=10.*sinsq
        if sinsq<0.025:
          factor=400.*sinsq**2
        dfield[k,j,:]=zonalmean+(field[k,j,:]-zonalmean)*factor
    return dfield

def smooth(field):
    '''
    smooth field horizontally
    '''
    import numpy as np
    dfield=np.empty( (field.shape[0],field.shape[1],field.shape[2]) )
    ny=field.shape[1]
    fieldn=np.roll(field,-1,1)
    fieldn[:,ny-1,:] = field[:,ny-1,:] 
    fields=np.roll(field,1,1)
    fields[:,0,:] = field[:,0,:]
    dfield=0.6*field+0.1*(np.roll(field,-1,2)+np.roll(field,1,2) +fieldn+fields)   
    return dfield 
    
def div(fieldx,fieldy,fieldz,a,coslat,dlong,dlat,thick2level):
    '''
    calculate divergence in eta coordinates
    '''
    import numpy as np
    import diagnosticSGtools as SG
    for j in range(fieldy.shape[1]):
      fieldy[:,j, ]=fieldy[:,j, ]*coslat[j]/thick2level[:,j,:]
    divergence=SG.ddphi(fieldy, a, dlat)
    for j in range(fieldy.shape[1]):
      divergence[:,j,:]=divergence[:,j,:]/coslat[j]
    fieldx=fieldx/thick2level
    fieldz=fieldz/thick2level
    divergence=divergence+SG.ddlambda(fieldx, a, dlong, coslat)
    divergence=divergence+SG.ddeta(fieldz)
    return(divergence)  
def Pinverse(A1,A2,A3,QI11,QI12,QI13,QI21,QI22,QI23,QI31,QI32,QI33,
              nbl,ny,coriolis20,
              upupcoeff,upcoeff,diagcoeff,downcoeff,downdowncoeff):
    '''
    multiply vector (A1,A2,A3) by P^{-1} eq. 
    '''
    import numpy as np
    import diagnosticSGtools as SG
    # Multiply by P_2^{-1} eq. (63)
    B1=QI11*A1+QI12*A2+QI13*A3
    B2=QI21*A1+QI22*A2+QI23*A3
    B3=QI31*A1+QI32*A2+QI33*A3
    # Multiply by P_3^{-1} eq. (64)
    for j in range(0,ny):
      B1[0:nbl,j, ]=B1[0:nbl,j, ]*coriolis20[j]
      B2[0:nbl,j, ]=B2[0:nbl,j, ]*coriolis20[j]
    # multiply by P_1^{-1} eq. (62) using pentadiagonal solve to invert P_1
    rhs=B1[0:nbl, ].copy()
    upwork=upcoeff.copy()
    diagwork=diagcoeff.copy()
    downwork=downcoeff.copy()
    lhs=SG.pentasolve(rhs,upupcoeff,upwork,diagwork,downwork,downdowncoeff)
    B1[0:nbl, ]=lhs
    rhs=B2[0:nbl, ].copy()
    upwork=upcoeff.copy()
    diagwork=diagcoeff.copy()
    downwork=downcoeff.copy()
    lhs=SG.pentasolve(rhs,upupcoeff,upwork,diagwork,downwork,downdowncoeff)
    B2[0:nbl, ]=lhs
    return(B1,B2,B3)
def CalcG(Q11R,Q12,Q21,Q22R,Q31,Q32,
              ug,vg,g,thetav,heat,nbl,ny,coriolis,
              updiffcoeff,diagdiffcoeff,downdiffcoeff):
    '''
    Calculate G vector including forcing eq (60)
    '''
    import numpy as np
    import diagnosticSGtools as SG
    (thetae, thetaw, thetan, thetas)=SG.fieldshift(thetav)
    ugbythetaheat=2.*vg/(thetae+thetaw)*heat
    vgbythetaheat=2.*ug/(thetan+thetas)*heat
    G1=-Q11R*ug-Q12*vg
    G2=-Q21*ug-Q22R*vg
    G3=-Q31*ug-Q32*vg+g*heat/thetav
    for j in range(0,ny):
      G1[:,j, ]=G1[:,j, ]-coriolis[j]*vgbythetaheat[:,j, ]
      G2[:,j, ]=G2[:,j, ]+coriolis[j]*ugbythetaheat[:,j, ]
    G1[0:nbl, ]=G1[0:nbl, ]-SG.vertdiff(ugbythetaheat,updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl)
    G2[0:nbl, ]=G2[0:nbl, ]-SG.vertdiff(vgbythetaheat,updiffcoeff, diagdiffcoeff, downdiffcoeff, nbl)
    return(G1,G2,G3)
def normvector(A1,A2,A3,title):
    '''
    Calculate and print norm of input vector
    '''
    import numpy as np
    usq=A1*A1
    normu=np.sqrt(usq.sum()/usq.size)
    vsq=A2*A2
    normv=np.sqrt(vsq.sum()/vsq.size)
    wsq=A3*A3
    normw=np.sqrt(wsq.sum()/wsq.size)
    print(title,normu,normv,normw) 
    return     
def plotarray(field,cube,amin,amax,aint,title,figure):
    '''
    Plot array using grid defined by supplied cube
    '''
    import numpy as np
    import socket
    if 'spice' in socket.gethostname():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    model_level=cube.coord('model_level_number').points
    x, y = np.meshgrid(longitude, latitude)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0),label=figure)
    plt.contourf(x, y, field, levels=np.arange(amin, amax, aint), cmap='rainbow',
      extend='both',transform=ccrs.PlateCarree()) 
    plt.colorbar(shrink=0.5) 
    ax.coastlines(color='dimgrey')
    long_int=(round(longitude.max())-round(longitude.min()))/8.
    long_int=round(long_int)
    lat_int=(round(latitude.max())-round(latitude.min()))/6.
    lat_int=round(lat_int)
    xlocs1=np.arange(round(longitude.min()),round(longitude.max())+long_int, long_int)
    ylocs1=np.arange(round(latitude.min()),round(latitude.max())+lat_int, lat_int)
    ax.gridlines(xlocs=xlocs1, ylocs=ylocs1)
    # Set the location of the ticks.
    ax.set_xticks(xlocs1, crs=ccrs.PlateCarree())
    # For the y ticks, don't include the 90 degrees North location.
    ax.set_yticks(ylocs1[:-1], crs=ccrs.PlateCarree())
    # Use Cartopy's tick formatter to include units
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.title(title, color='grey')
    plt.savefig(figure)
    
def plotEWxsect(field,cube,lat,amin,amax,aint,title,figure):
    '''
    Plot array using grid defined by supplied cube
    '''
    import numpy as np
    import socket
    if 'spice' in socket.gethostname():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    model_level=cube.coord('model_level_number').points
    index = (np.abs(latitude - lat)).argmin()
    plotfield=np.squeeze(field[:,index,:])
    xx, yy = np.meshgrid(longitude, model_level) 
    plt.figure()
    plt.contourf(xx, yy, plotfield, levels=np.arange(amin, amax, aint), cmap='rainbow') 
    plt.colorbar(shrink=0.5) 
    plt.title(title, color='grey')
    plt.savefig(figure)
def plotNSxsect(field,cube,long,amin,amax,aint,title,figure):
    '''
    Plot array using grid defined by supplied cube
    '''
    import numpy as np
    import socket
    if 'spice' in socket.gethostname():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    model_level=cube.coord('model_level_number').points
    index = (np.abs(longitude - long)).argmin()
    plotfield=np.squeeze(field[:,:,index])
    xx, yy = np.meshgrid(latitude, model_level) 
    plt.figure()
    plt.contourf(xx, yy, plotfield, levels=np.arange(amin, amax, aint), cmap='rainbow') 
    plt.colorbar(shrink=0.5) 
    plt.title(title, color='grey')
    plt.savefig(figure)
def plotarea(field,cube,lat1,lat2,long1,long2):
    '''
    Set cube and data for limited area plot
    '''
    import numpy as np
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    cube.coord('latitude').bounds = None
    cube.coord('longitude').bounds = None
    cube_cut = cube.intersection(latitude=(lat1, lat2),longitude=(long1, long2))
    field1=field[np.where((latitude>lat1)&(latitude<lat2))[0], ]
    field_cut=field1[:,np.where((longitude>long1)&(longitude<long2))[0]]
    print(field_cut.shape)
    latitude = cube_cut.coord('latitude').points
    longitude = cube_cut.coord('longitude').points
    print(latitude.shape,longitude.shape)
    return field_cut, cube_cut
def smallarea(field,cube,lat1,lat2,long1,long2):
    '''
    Set cube and data for limited area processing
    '''
    import numpy as np
    latitude = cube.coord('latitude').points
    longitude = cube.coord('longitude').points
    cube.coord('latitude').bounds = None
    cube.coord('longitude').bounds = None
    cube_cut = cube.intersection(latitude=(lat1, lat2),longitude=(long1, long2))
    field1=field[:,np.where((latitude>lat1)&(latitude<lat2))[0], ]
    field_cut=field1[:,:,np.where((longitude>long1)&(longitude<long2))[0]]
    print(field_cut.shape)
    latitude = cube_cut.coord('latitude').points
    longitude = cube_cut.coord('longitude').points
    print(latitude.shape,longitude.shape)
    return field_cut, cube_cut
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
