import argparse
import pdb
import sys

import numpy as np
import cmocean
import datetime

import xarray as xr
import pandas as pd
import iris
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import cross_section
from windspharm.xarray import VectorWind
from matplotlib.patches import Rectangle
from custom_cmap import *

import matplotlib
matplotlib.use('Agg')

# set the colormap and centre the colorbar                                             
class MidpointNormalize(colors.Normalize):
    """                                                                                
    Normalise the colorbar so that diverging bars work there way either side           
    from a prescribed midpoint value)                                                  
                                                                                       
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        #colors.Normalize.__init__(self, vmin, vmax, clip)
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases
        # to make a simple example...                                                            
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) #, np.isnan(value))

def open_file(file, hr, ftype=''):
    """                                                                                  
    open netCDF file using Xarray                                                        
                                                                                         
                                                                                         
    Args:                                                                                
      file (str): path to input file                                                     
      hr (int): desired output time after start of forecast ('T+')                       
      stream (str): string corresponding to input file ('pb', 'pc', etc)

    Kwargs:
      ftype (str): 
    """

    # read in multiple files (two different data streams)
    # 03/09/21 --> currently only set up for use with 4p4 MetUM data  
    file_str = ['{0}{1}.nc'.format(file,'_pverc'), '{0}{1}.nc'.format(file,'_pverd')]

    # read in data using xarray
    # add capability to search for all files in this path, and open simultaneously
    if ftype == '4p4':
        data_pc = xr.open_dataset(file_str[0]).metpy.parse_cf()
        data_pd = xr.open_dataset(file_str[1]).metpy.parse_cf()

    # extract string beginning 'YYYYMMDD...' from file path string                       
    date = file.split("/")[-1]
    yr = date[0:4]; mn = date[4:6]; dy = date[6:8]; hh = date[9:11]

    # create datetime object corresponding to start of forecast using input information 
    sstr = datetime.datetime(int(yr), int(mn), int(dy), int(hh))

    # update datetime object to match the validity time                                  
    tp = datetime.timedelta(hours=hr)
    dstr = sstr + tp

    # formatted strings for output file                                                   
    tstr = dstr.strftime("%Y%m%dT%H00")
    sstr = sstr.strftime("%Y%m%dT%H00Z")

    if ftype == '4p4':
        return sstr, dstr, tstr, data_pc, data_pd
    else:
        return sstr, dstr, tstr

def subset(data, bounds, var='', vtime=[]):
    """                                                                                  
    create subset of gridded data                                                        
                                                                                         
    Args:                                                                                
      data (xarray data array): multi-dimensional data array                             
      bounds (list): latitude / longitude bounds for subset                              
                                                                                         
    Kwargs:                                                                              
      var (str): variable to plot                                                        
      vtime (datetime object): validity time (e.g. 12Z on 23rd Oct 2018)                 
    """

    # get size of time dimension                                                            
    coords = data.coords.dims
    for c in coords:
        if c == 'p':
            dset = '4p4'
        #elif c == 'level':
        elif c == 'isobaricInhPa':
            dset = 'era5'
        elif c == 'hybrid_ht_1':
            dset = 'n768'

    # use select function to subset data                                                 
    if var == 'circ':
        if dset == '4p4':
            data = data.sel( longitude = slice(bounds[0], bounds[1]),
                             latitude = slice(bounds[2], bounds[3]),
                             longitude_1 = slice(bounds[0], bounds[1]),
                             latitude_1 = slice(bounds[2], bounds[3]))
        else:
            data = data.reindex(latitude=data.latitude[::-1]) # rearrange to S --> N
            data = data.sel( time = slice('2018-10-21T12', '2018-10-26T12'), 
                             longitude = slice(bounds[0], bounds[1]),
                             latitude = slice(bounds[2], bounds[3]) )
    else:
        if dset == '4p4':
            data = data.sel( t_1=slice(vtime,vtime), t=slice(vtime,vtime),
                             longitude=slice(bounds[0], bounds[1]),
                             latitude=slice(bounds[2], bounds[3]),
                             longitude_1=slice(bounds[0], bounds[1]),
                             latitude_1=slice(bounds[2], bounds[3]) )
        else:
            data = data.reindex(latitude=data.latitude[::-1]) # rearrange to S --> N 
            data = data.sel( time=slice(vtime,vtime), 
                             longitude=slice(bounds[0], bounds[1]),
                             latitude=slice(bounds[2], bounds[3]) )

    return data

def calc_area_ave_acc_prcp(prcp, dataset, bv_lat, bv_lon, r0=3.0, plt_prcp=True):
    """
    calculate area-averaged, accumulated precipitation following a storm, within a specified radius 

    Args:
      prcp (Xarray DataArray): multi-dimensional data array 
      dataset (string): dataset to analyse (e.g. 4p4, N768, GPM) 
      bv_lat (Pandas DataFrame): vortex centre latitude from ERA-I tracking 
      bv_lon (Pandas DataFrame): vortex centre longitude from ERA-I tracking 

    Kwargs:
      r0 (float): radius for calculation (degrees) 
      plt_prcp (bool): plot precipitation as well as calculating acc precip following the storm 
    """

    # read in data and split into 12-h chunks 
    if dataset == '4p4':
        prcp = prcp * 3600. 
        prcp = prcp.resample(time="12H").sum().sel(time = slice('2018-10-21T12', 
                                                                '2018-10-26T12') )
    elif dataset == 'gpm':
        prcp = prcp[::2,:,:].resample(time="12H").sum().sel(time = slice('2018-10-21T12',
                                                                         '2018-10-26T00') )
        prcp = prcp.transpose('time','latitude','longitude')

    # initialise array to hold values of accumulated precipitation
    prcp_arr = xr.DataArray(np.ones(prcp.time.shape[0]),
                            dims=["time"],
                            coords={
                                "time": prcp.time
                                },
                            )

    # extract elements from (6-h) BV centre position arrays
    t_ind = np.arange(0, prcp.time.shape[0])
    if dataset == 'n768':
        xi = [4,6,8,10,12,14,16,18,20]
    else: 
        xi = [2,4,6,8,10,12,14,16,18,20]

    # define plot limits 
    n0 = np.rint(prcp.longitude[0].data); n1 = np.rint(prcp.longitude[-1].data)
    t0 = np.rint(prcp.latitude[0].data); t1 = np.rint(prcp.latitude[-1].data)
            
    # loop over times
    for i, it in enumerate(t_ind):

        # counter variable 
        xii = xi[i]

        if plt_prcp:

            # plot precipitation rate 
            fig = plt.figure(figsize=[9,6])
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

            # subset data 
            if dataset == 'n768':
                Levels=[0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
            else:
                Levels=[1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
            prcp = prcp.sel(longitude=slice(n0+0.5,n1-0.5), latitude=slice(t0+0.5,t1-0.5) )
            prcp[it,:,:].plot.contourf(ax=ax, levels=Levels, extend='max', 
                                       transform=ccrs.PlateCarree(),
                                       cbar_kwargs={'label': 'mm'},
                                       cmap=cmocean.cm.haline_r)

            # coastlines, tickmarks, etc
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
        
            # date string 
            dstr = prcp.time.dt.strftime("%Y%m%dT%H").values

            # set up plot 
            fili = './prcp_{1}_{0}.png'.format(dstr[it],dataset)

            # define plot limits and set up tickmarks 
            xint = 5; yint = 3
            ns = np.rint(prcp.longitude[0].data); nf = np.rint(prcp.longitude[-1].data)
            ts = np.rint(prcp.latitude[0].data); tf = np.rint(prcp.latitude[-1].data)
            ax.set_xticks(np.arange(ns, nf+1, xint))
            ax.set_xticklabels(np.arange(ns, nf+1, xint))
            ax.set_yticks(np.arange(ts, tf+1, yint))
            ax.set_yticklabels(np.arange(ts, tf+1, yint))

            # add gridlines 
            plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)
            plt.title('')

            # overlay BV box position 
            ax.add_patch( Rectangle( (bv_lon[xii]-r0, bv_lat[xii]-r0),
                                     2*r0, 2*r0, linewidth=2,
                                     facecolor='none', edgecolor='k') )

            # save figure and continue 
            fig.savefig(fili,dpi=200)

        # calculate area-averaged, accumulated precipitation 
        pr = prcp[it,:,:]
        pr = pr.loc[bv_lat[xii]-r0:bv_lat[xii]+r0,
                    bv_lon[xii]-r0:bv_lon[xii]+r0].mean()

        prcp_arr[i] = pr.data

    return prcp_arr



def calc_circ(u, v, bv_lat, bv_lon, plev=800, mlev=1.810000e+03, r0=3.0):
    """                                                                                  
    calculate circulation following the vortex for multiple validity times               
    storm the values in an array for plotting                                            
                                                                                         
    Args:
      u (Xarray DataArray): multi-dimensional data array (zonal wind)
      v (Xarray DataArray): multi-dimensional data array (meridional wind)
      bv_lat (Pandas DataFrame): vortex centre latitude
      bv_lon (Pandas DataFrame): vortex centre longitude
                                                                                         
    Kwargs:                                                                              
      plev (int): pressure level for calculation (4p4)                                   
      mlev (int): model level for calculation (N768)                                     
      r0 (float): radius for calculation (degrees)
    """

    # get size of time dimension
    coords = u.coords.dims
    for c in coords:
        if c == 'time':
            ntimes = u.time.shape[0]
        elif c == 't':
            ntimes = u.t.shape[0]

    times = np.arange(0, ntimes)

    # determine dataset from coordinate names (beta)
    for c in coords:
        if c == 'p':
            dset = '4p4'
        elif c == 'isobaricInhPa':
            dset = 'era5'
        elif c == 'hybrid_ht_1':
            dset = 'n768'

    # initialise array (better way to do this?)                                          
    if dset == 'n768':
        t_ind = times[:]
        circ_arr = xr.DataArray(t_ind)
        xi = [4,6,8,10,12,14,16,18,20]
    else:
        t_ind = times[::2]
        circ_arr = xr.DataArray(t_ind)

    # loop over times                                                                     
    for i, it in enumerate(t_ind):
        # read in wind components on single level and calculate relative vorticity        
        if dset == 'n768':
            u0 = u[it,:,:,:].sel(hybrid_ht_1=int(mlev), method="nearest" )
            v0 = v[it,:,:,:].sel(hybrid_ht_1=int(mlev), method="nearest" )
            v0 = v0.interp(latitude_1=u0.latitude)
        elif dset == 'era5':
            u0 = u[it,:,:,:].sel(isobaricInhPa=int(plev) )
            v0 = v[it,:,:,:].sel(isobaricInhPa=int(plev) )
        else:
            u0 = u[it,:,:,:].sel(p=int(plev) )
            v0 = v[it,:,:,:].sel(p=int(plev) )

        vort = mpcalc.vorticity(u0, v0, dx = None, dy = None) * 1000000
        vort.attrs['units'] = '10-5 s-1'

        # calculate circulation following the vortex                                      
        if dset == 'n768':
            xii = xi[i]
            circ = (vort.loc[bv_lat[xii]-r0:bv_lat[xii]+r0,
                             bv_lon[xii]-r0:bv_lon[xii]+r0]).mean()
        else:
            circ = (vort.loc[bv_lat[i]-r0:bv_lat[i]+r0, bv_lon[i]-r0:bv_lon[i]+r0]).mean()
        circ_arr[i] = circ

    return circ_arr



def plot_pv(pv, u, v, dataset, plev=800, mlev=1.810000e+03, bv_centre=True):
    """
    produce an x-y plot of potential vorticity 

    Args:
      pv (xarray DataArray): multi-dimensional data array 
      u (xarray DataArray): multi-dimensional data array
      v (xarray DataArray): multi-dimensional data array 
      dataset (str): dataset to analyse (e.g. 4p4, N768, ERA5) 

    Kwargs:
      plev (int): pressure level for calculation (4p4) 
      mlev (int): model level for calculation (N768)  
      bv_centre (bool): overlay BV centre from Kevin Hodges' algorithm  
    """

    # set up plots and axes
    fig = plt.figure(figsize=[9,6])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

    # plot data
    if dataset == '4p4' or dataset == 'era5':
        #Levels = (-2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1, 
        #          0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
        Levels = (-2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1,
                  0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
        pv.attrs['units'] = 'PVU'
        pv.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                         cbar_kwargs={'label': 'PVU'},
                         cmap='PuOr')

    # add coastlines, gridlines and tickmarks
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
    xint = 5; yint = 3
    n0 = np.rint(u.longitude[0].data); n1 = np.rint(u.longitude[-1].data)
    t0 = np.rint(u.latitude[0].data); t1 = np.rint(u.latitude[-1].data)
    ax.set_xticks(np.arange(n0, n1+xint, xint))
    ax.set_xticklabels(np.arange(n0, n1+xint, xint))
    ax.set_yticks(np.arange(t0, t1+yint, yint))
    ax.set_yticklabels(np.arange(t0, t1+yint, yint))
    plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

    # add horizontal wind barbs
    if dataset == '4p4':
        skip = 24
    elif dataset == 'era5':
        skip = 4
    else:
        skip = 1
    ax.quiver(u.longitude.values[::skip], u.latitude.values[::skip],
              u.values[::skip, ::skip], v.values[::skip, ::skip], angles='xy', scale=400)

    return fig 


def plot_n2(pv_grad, u, v, bv_centre, dataset, plev=800, mlev=1.810000e+03):
    """                                                                                         
    produce an x-y plot of potential vorticity                                                  
                                                                                                
    Args:                                                                                       
      pv_grad (xarray DataArray): multi-dimensional data array                                       
      dataset (str): dataset to analyse (e.g. 4p4, N768, ERA5)                                  
                                                                                                
    Kwargs:                                                                                     
      plev (int): pressure level for calculation (4p4)                                          
      mlev (int): model level for calculation (N768)                                            
      bv_centre (bool): overlay BV centre from Kevin Hodges' algorithm                          
    """

    # vortex track information between 12Z on 21st and 26th October 2018 
    bv_lat = bv_centre.loc[0:20, "lat_vort"]; bv_lon = bv_centre.loc[0:20, "lon_vort"]
    bv_time = bv_centre.loc[0:20, "Time"]

    # find index of vortex centre information that matches the chosen time
    if dataset == '4p4':
        filter = bv_time==pv_grad.coords['t'].data
    time_match = bv_time.where(filter).notna()
    ind = int(time_match.loc[time_match==True].index.values)

    # set up plots and axes                                                                     
    fig = plt.figure(figsize=[9,6])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

    # plot data                                                                                 
    if dataset == '4p4' or dataset == 'era5':
        #Levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        dl = 5.0; qmin = 0.0; qmax = 100.0; Levels = np.arange(qmin,qmax+dl,dl)
        pv_grad.attrs['units'] = 'q/N^2'
        pv_grad.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': r'$q/N^2\,(s^{-2})$'},
                         cmap='plasma_r')

    # add coastlines, gridlines and tickmarks                                                    
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
    xint = 5; yint = 3
    n0 = np.rint(pv_grad.longitude[0].data); n1 = np.rint(pv_grad.longitude[-1].data)
    t0 = np.rint(pv_grad.latitude[0].data); t1 = np.rint(pv_grad.latitude[-1].data)
    ax.set_xticks(np.arange(n0, n1+xint, xint))
    ax.set_xticklabels(np.arange(n0, n1+xint, xint))
    ax.set_yticks(np.arange(t0, t1+yint, yint))
    ax.set_yticklabels(np.arange(t0, t1+yint, yint))
    plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

    # add horizontal wind barbs                                                                  
    if dataset == '4p4':
        skip = 24
    elif dataset == 'era5':
        skip = 4
    else:
        skip = 1
    ax.quiver(u.longitude.values[::skip], u.latitude.values[::skip],
              u.values[::skip, ::skip], v.values[::skip, ::skip], angles='xy', scale=400)

    # overlay vortex centre position (Kevin Hodges' tracking algorithm)
    ax.plot(bv_lon[ind], bv_lat[ind], marker='*', color='c', markersize=12)

    return fig

def plot_xy(data, dataset, var, bv_centre, plev=800, mlev=1500, wind='', overlay=''):
    """
    produce a simple x-y plot given a dataset (e.g. ERA5) and a variable (e.g. vorticity)

    Args:
      data (xarray DataArray): multi-dimensional data array
      dataset (str): dataset to analyse (e.g. 4p4, N768, ERA5)
      bv_centre (Pandas DataFrame): Borneo vortex tracking information
      var (str): variable to plot (e.g. vorticity, geopotential height)
    Kwargs:
      plev (int): pressure level for calculation (4p4) --> NOT CURRENTLY ACTIVE
      mlev (int): model level for calculation (N768) --> NOT CURRENTLY ACTIVE 
      bv_centre (bool): overlay BV centre from Kevin Hodges' algorithm 
      wind (str): plot full, geostrophic or ageostrophic wind (default=full)
      overlay (str): overlay line contours of additional diagnostic (e.g. w,u,v)
    """

    # extract vortex track information between 12Z on 21st and 26th October
    bv_lat = bv_centre.loc[0:20, "lat_vort"]; bv_lon = bv_centre.loc[0:20, "lon_vort"]
    bv_time = bv_centre.loc[0:20, "Time"]

    # find index of vortex centre information that matches the chosen time
    if dataset == '4p4':
        filter = bv_time==data.coords['t'].data
    elif dataset == 'n768' or dataset == 'sgt':
        filter = bv_time==data[0].coords['t'].data
    elif dataset == 'era5':
        filter = bv_time==data.coords['time'].data[0]
    time_match = bv_time.where(filter).notna()
    ind = int(time_match.loc[time_match==True].index.values)

    # separate into specific variables 
    if dataset == 'era5':
        w = data["w"].squeeze('time'); q = data["q"].squeeze('time') * 1000.
        q.attrs['units']='g kg-1'
        z = data["z"].squeeze('time'); u = data["u"].squeeze('time')
        v = data["v"].squeeze('time'); pv = data["pv"].squeeze('time')
        t = data["t"].squeeze('time'); z = data["z"].squeeze('time') / 10
    elif dataset == '4p4':
        omg = data["omega"]; u = data["u"]; v = data["v"]
        q = data["q"] * 1000.; q.attrs['units']='g kg-1'
        u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'
        temp = data["temp"]

        # interpolate all variables onto same grid before plotting 
        temp = temp.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")

        #z = z.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")
        #z.attrs['units'] = 'dam'

        # calculate additional diagnostics 
        th = mpcalc.potential_temperature(temp['p'], temp); th.attrs['units'] = 'K'
        mix = mpcalc.mixing_ratio_from_specific_humidity(q)

    else: # read in both N768 and SGT tool datasets

        # N768 diagnostics 
        u_gl = data[0]; v_gl = data[1]; w_gl = data[2] * 100.; pv_gl = data[3]
        q_gl = data[4]; th_gl = data[5]; ug_gl = data[6]; vg_gl = data[7]
        # interpolate geostrophic wind components (output from SGT tool) onto N768 grid 
        ug_gl = ug_gl.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
        vg_gl = vg_gl.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")
        # calculate ageostrophic wind components
        ua_gl = u_gl - ug_gl; va_gl = v_gl - vg_gl

        # SGT tool diagnostics 
        u_sg = data[8]; v_sg = data[9]; w_sg = data[10] * 100.; ug_sg = data[11]; vg_sg = data[12]
        # calculate ageostrophic wind components 
        ua_sg = u_sg - ug_sg; va_sg = v_sg - vg_sg

        # interpolate SGT tool data onto N768 grid 
        u_sg = u_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
        v_sg = v_sg.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")
        w_sg = w_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
        ug_sg = ug_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
        vg_sg = vg_sg.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")
        ua_sg = ua_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
        va_sg = va_sg.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")

    # calculate vector wind speed and relative vorticity 
    if dataset == 'n768':
        spd = mpcalc.wind_speed(u_gl,v_gl); spd.attrs['units'] = 'm/s'
    elif dataset == 'sgt':
        spd = mpcalc.wind_speed(u_sg,v_sg); spd.attrs['units'] = 'm/s'
    else:
        spd = mpcalc.wind_speed(u,v); spd.attrs['units'] = 'm/s'

    if dataset == '4p4' or dataset == 'era5':
        vort = mpcalc.vorticity(u, v, dx=None, dy=None) * 100000; vort.attrs['units'] = '10-5 s-1'
    elif dataset == 'n768':
        vort_gl = mpcalc.vorticity(u_gl, v_gl, dx=None, dy=None) * 100000
        vort_gl.attrs['units'] = '10-5 s-1'

    """
    # smooth fields before plotting 
    pv_gl = mpcalc.smooth_gaussian(pv_gl, 4); pv_gl.attrs['units'] = 'PVU'
    v_gl  = mpcalc.smooth_gaussian(v_gl, 64); v_gl.attrs['units'] = 'm/s'
    u_gl  = mpcalc.smooth_gaussian(u_gl, 64); u_gl.attrs['units'] = 'm/s'
    """

    # effective PV gradient (q/N^2)
    if dataset == '4p4':

        # first, calculate mixing ratio                                                     
        mix = mpcalc.mixing_ratio_from_specific_humidity(q)
        # then, calculate density (using mixing ratio)                                      
        rho = mpcalc.density(th.p, temp, mix)

        # next, calculate d(theta)/dp                                                       
        th_dp = mpcalc.first_derivative(th, axis=0) / 100.; th_dp.attrs['units'] = 'K/Pa'
        # finally, calculate N^2 in pressure coordinates                                    
        n2 = th_dp * -( (rho * np.square(9.81) ) / th)

        # now calculate effective PV gradient (q / N^2)                                     
        pv_grad = (q / n2) / 10000.

    # set up plots and axes
    fig = plt.figure(figsize=[9,6])
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

    """
    subsections in the IF statement below are currently identical
    will change as I explore the sensitivity of the data 
    """

    # ERA5 
    if dataset == 'era5':
        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'v':
            dl = 2.0; lmin = -20; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            v.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': v.units},
                            cmap=Cmap)            
        elif var == 'u':
            dl = 2.0; lmin = -20; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            u.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': u.units},
                            cmap=Cmap)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            vort.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': vort.units},
                               cmap='seismic')
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            w.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': w.units},
                            cmap='seismic')
        elif var == 'geo':
            levs = np.arange(197.0, 208.0, 1.0)
            z.plot.contourf(ax=ax, extend='max', levels=levs, transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': z.units},
                            cmap='plasma_r')
        elif var == 'th':
            th.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': th.units},
                             cmap='plasma_r')
        elif var == 'q':
            dl = 1.0; qmin = 1.0; qmax = 16.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
            q.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': q.units},
                            cmap=Cmap)
        elif var == 'pv':
            Levels = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0)
            pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
            pv.attrs['units'] = 'PVU'
            pv.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': 'PVU'},
                             cmap='twilight')

    # 4.4 km MetUM forecast 
    elif dataset == '4p4':
        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'u':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            u.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': r'Zonal wind $\mathrm{(m\,s^{-1})}$'},
                            cmap=Cmap)
        elif var == 'v':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            v.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': r'Meridional wind $\mathrm{(m\,s^{-1})}$'},
                            cmap=Cmap)
        elif var == 'ua':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ua.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': u.units},
                             cmap=Cmap)
        elif var == 'va':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            va.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': v.units},
                             cmap=Cmap)
        elif var == 'ug':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ug.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': u.units},
                             cmap=Cmap)
        elif var == 'vg':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            vg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': v.units},
                             cmap=Cmap)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl, 'bwr')
            vort.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': vort.units},
                               cmap='seismic')
        elif var == 'omg':
            dl = 0.5; lmin = -10; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin, lmax, 0, dl, 'bwr')
            omg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': omg.units},
                              cmap=Cmap)
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin, lmax, 0, dl, 'bwr')
            w.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': 'cm/s'},
                            cmap=Cmap)
        elif var == 'geo':
            levs = np.arange(197.0, 208.0, 1.0)
            z.plot.contourf(ax=ax, extend='max', levels=levs, transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': z.units},
                            cmap='plasma_r')
        elif var == 'th':
            th.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': th.units},
                             cmap='plasma_r')
        elif var == 'rh':
            dl = 5.0; rmin = 40.0; rmax = 100.0; Levels=np.arange(rmin,rmax+dl,dl)
            rh = mpcalc.relative_humidity_from_specific_humidity(q.p, temp, q) * 100.
            rh.attrs['units'] = '%'
            rh.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': '%'},
                             cmap='BuPu')            
        elif var == 'q':
            dl = 1.0; qmin = 1.0; qmax = 16.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
            q.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': q.units},
                            cmap=Cmap)
        elif var == 'pv':
            Levels = (-0.2, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
            pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
            pv.attrs['units'] = 'PVU'
            Cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("magma_r").colors[:250])
            pv.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': 'PVU'},
                             cmap=Cmap)            
        elif var == 'n2':
            Levels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            Cmap = 'plasma_r'
            pv_grad.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                                  cbar_kwargs={'label': 'q/N^2'},
                                  cmap=Cmap)
            
    elif dataset == 'n768':

        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'u':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            u_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': u_gl.units},
                               cmap=Cmap)
        elif var == 'v':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            v_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': v_gl.units},
                               cmap=Cmap)
        elif var == 'ua':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ua_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': u_gl.units},
                                cmap=Cmap)
        elif var == 'va':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            va_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': v_gl.units},
                                cmap=Cmap)
        elif var == 'ug':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ug_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': u_gl.units},
                                cmap=Cmap)
        elif var == 'vg':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            vg_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': v_gl.units},
                                cmap=Cmap)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            vort_gl.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                                  cbar_kwargs={'label': vort_gl.units},
                                  cmap='seismic')
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin, lmax, 0, dl, 'bwr')
            w_gl.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': 'cm/s'},
                               cmap=Cmap)
        elif var == 'th':
            th_gl.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': th_gl.units},
                                cmap='plasma_r')
        elif var == 'q':
            dl = 1.0; qmin = 1.0; qmax = 16.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
            q_gl.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': q_gl.units},
                               cmap=Cmap)
        elif var == 'pv':
            pv = pv_gl * 1000000.

            tstr = str(pv.coords['t'].data)
            # T+0 and T+12
            if tstr == '2018-10-22T00:00:00.000000000' or tstr == '2018-10-21T12:00:00.000000000':
                pv = pv.interp(longitude_1=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            # all other times (T+24 onwards)
            else:
                pv = pv.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")

            """
            Levels = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
                      1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
            Cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("magma_r").colors[:250])
            Cmap.set_under('white')
            """
            dl = 0.2; vmin = -2.4; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'PuOr_r')
            pv.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': 'PVU'},
                             cmap=Cmap)
            """
            Levels = (-2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1,
                      0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
            #Levels = (-2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1,
            #          0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
            pv.attrs['units'] = 'PVU'
            pv.plot.contourf(ax=ax, extend='max', levels=Levels, transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': 'PVU'},
                             cmap='PuOr')
            """

    else: # SGT tool 
        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            vort_sg.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                                  cbar_kwargs={'label': vort_sg.units},
                                  cmap='seismic')
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin, lmax, 0, dl, 'bwr')
            w_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': 'cm/s'},
                               cmap=Cmap)
        elif var == 'u':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            u_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': u_sg.units},
                               cmap=Cmap)
        elif var == 'v':
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            v_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': v_sg.units},
                               cmap=Cmap)
        elif var == 'ua':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ua_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': u_sg.units},
                                cmap=Cmap)
        elif var == 'va':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            va_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': v_sg.units},
                                cmap=Cmap)
        elif var == 'ug':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            ug_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': u_sg.units},
                                cmap=Cmap)
        elif var == 'vg':
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels = normalise_cmap(vmin,vmax,0,dl,'bwr')
            vg_sg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                                cbar_kwargs={'label': v_sg.units},
                                cmap=Cmap)

    # skip out some vectors when plotting 
    if dataset == 'era5':
        skip = 4
    elif dataset == 'sgt':
        skip = 8
    elif dataset == 'n768':
        if overlay == 'v':
            v_sg.plot.contour(ax=ax, levels=np.arange(4, 20, 2), 
                              transform=ccrs.PlateCarree(), colors=['slategray'])
        elif overlay == 'w':
            w_sg.plot.contour(ax=ax, levels=np.arange(-10, 10, 4),
                              transform=ccrs.PlateCarree(), colors=['deepskyblue'])
        elif overlay == 'vw':
            v_sg.plot.contour(ax=ax, levels=np.arange(4, 20, 2),
                              transform=ccrs.PlateCarree(), colors=['slategray'])
            w_sg.plot.contour(ax=ax, levels=np.arange(1, 10, 1),
                              transform=ccrs.PlateCarree(), colors=['deepskyblue'])
        skip = 8
    else: # 4p4
        skip = 24

    # add coastlines, gridlines and tickmarks 
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
    xint = 5; yint = 3
    if dataset == 'n768':
        n0 = np.rint(u_gl.longitude[0].data)+1; n1 = np.rint(u_gl.longitude[-1].data)-1
        t0 = np.rint(u_gl.latitude[0].data)+1; t1 = np.rint(u_gl.latitude[-1].data)-1
    elif dataset == 'sgt':
        n0 = np.rint(u_sg.longitude[0].data)+1; n1 = np.rint(u_sg.longitude[-1].data)-1
        t0 = np.rint(u_sg.latitude[0].data)+1; t1 = np.rint(u_sg.latitude[-1].data)-1
    else: # 4p4 or ERA5
        n0 = np.rint(u.longitude[0].data); n1 = np.rint(u.longitude[-1].data)
        t0 = np.rint(u.latitude[0].data); t1 = np.rint(u.latitude[-1].data)        
    ax.set_xticks(np.arange(n0, n1+xint, xint))
    ax.set_xticklabels(np.arange(n0, n1+xint, xint))
    ax.set_yticks(np.arange(t0, t1+yint, yint))
    ax.set_yticklabels(np.arange(t0, t1+yint, yint))
    plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

    # add horizontal wind barbs
    if dataset == 'n768':
        if wind == 'full':
            ax.quiver(u_gl.longitude.values[::skip], u_gl.latitude.values[::skip],
                      u_gl.values[::skip, ::skip], v_gl.values[::skip, ::skip], angles='xy', scale=300)
        elif wind == 'geo':
            ax.quiver(u_gl.longitude.values[::skip], u_gl.latitude.values[::skip],
                      ug_gl.values[::skip, ::skip], vg_gl.values[::skip, ::skip], angles='xy', scale=200)
        else: # ageo 
            ax.quiver(u_gl.longitude.values[::skip], u_gl.latitude.values[::skip],
                      ua_gl.values[::skip, ::skip], va_gl.values[::skip, ::skip], angles='xy', scale=100)
    elif dataset == 'sgt':
        if wind == 'full':
            ax.quiver(u_sg.longitude.values[::skip], u_sg.latitude.values[::skip],
                      u_sg.values[::skip, ::skip], v_sg.values[::skip, ::skip], angles='xy', scale=300)
        elif wind == 'geo':
            ax.quiver(u_sg.longitude.values[::skip], u_sg.latitude.values[::skip],
                      ug_sg.values[::skip, ::skip], vg_sg.values[::skip, ::skip], angles='xy', scale=200)
        else: # ageo
            ax.quiver(u_sg.longitude.values[::skip], u_sg.latitude.values[::skip],
                      ua_sg.values[::skip, ::skip], va_sg.values[::skip, ::skip], angles='xy', scale=100)
    else:
        ax.quiver(data.longitude.values[::skip], data.latitude.values[::skip],
                  u.values[::skip, ::skip], v.values[::skip, ::skip], angles='xy', scale=400)

    # overlay vortex centre position 
    ax.plot(bv_lon[ind], bv_lat[ind], marker='*', color='k', markersize=12)

    return fig
    

def plot_xz(data, dataset, var):
    """
    produce a vertical cross-section plot
    currently only N-S or E-W, but working on adding functionality 

    Args:
      data (xarray DataArray): multi-dimensional data array
      dataset (str): dataset to analyse (e.g. 4p4, N768, ERA5)
      var (str): variable to plot (e.g. vorticity, geopotential height)  

    Kwargs:
      (): 
      ():
    """



    return fig 
