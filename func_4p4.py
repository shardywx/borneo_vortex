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
import metpy.calc as mpcalc
from metpy.units import units
from metpy.interpolate import cross_section
from windspharm.xarray import VectorWind
from custom_cmap import *

import matplotlib
matplotlib.use('TkAgg')

def open_file(file, hr):
    """                                                                                  
    open netCDF file using Xarray                                                        
                                                                                         
                                                                                         
    Args:                                                                                
      file (str): path to input file                                                     
      hr (int): desired output time after start of forecast ('T+')                       
                                                                                         
    """

    # read in data using xarray                                                          
    data = xr.open_dataset(file).metpy.parse_cf()

    # extract string beginning 'YYYYMMDD...' from file path string                       
    date = file.split("/")[-1]
    yr = date[0:4]; mn = date[4:6]; dy = date[6:8]; hh = date[9:11]

    # create datetime object corresponding to start of forecast using input information 
    dstr = datetime.datetime(int(yr), int(mn), int(dy), int(hh))

    # update datetime object to match the validity time                                  
    tp = datetime.timedelta(hours=hr); dstr = dstr + tp

    # formatted string for output file                                                   
    tstr = dstr.strftime("%Y%m%dT%H00")

    return data, dstr, tstr


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

def acc_prcp(prcp, dataset, bv_lat, bv_lon, r0=3.0):
    """
    calculate accumulated precipitation following a storm, within a specified radius 

    Args:
      prcp (Xarray DataArray): multi-dimensional data array 
      dataset (string): dataset to analyse (e.g. 4p4, N768, GPM) 
      bv_lat (Pandas DataFrame): vortex centre latitude from ERA-I tracking 
      bv_lon (Pandas DataFrame): vortex centre longitude from ERA-I tracking 

    Kwargs:
      r0 (float): radius for calculation (degrees) 
    """

    # read in data and split into 12-h chunks 
    if dataset == '4p4':
        prcp = prcp * 3600. 
        prcp = prcp.resample(time="12H").sum().sel(time = slice('2018-10-21T12', 
                                                                '2018-10-26T12') )
    elif dataset == 'gpm':
        prcp = prcp[::2,:,:].resample(time="12H").sum().sel(time = slice('2018-10-21T12',
                                                                         '2018-10-26T00') )

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

    # loop over times
    for i, it in enumerate(t_ind):

        # calculate accumulated precipitation 
        xii = xi[i]

        if dataset == 'gpm':
            pr = prcp[it,:,:]
            pr = pr.loc[bv_lon[xii]-r0:bv_lon[xii]+r0,
                        bv_lat[xii]-r0:bv_lat[xii]+r0].mean()
        else: # 4p4 or N768
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
            u0 = u[it,:,:,:].sel(hybrid_ht_1=int(mlev) )
            v0 = v[it,:,:,:].sel(hybrid_ht_1=int(mlev) )
        elif dset == 'era5':
            u0 = u[it,:,:,:].sel(isobaricInhPa=int(plev) )
            v0 = v[it,:,:,:].sel(isobaricInhPa=int(plev) )
        else:
            u0 = u[it,:,:,:].sel(p=int(plev) )
            v0 = v[it,:,:,:].sel(p=int(plev) )

        vort = mpcalc.vorticity(u0, v0, dx = None, dy = None) * 10000
        vort.attrs['units'] = '10-5 s-1'

        # calculate circulation following the vortex                                      
        if dset == 'n768':
            xii = xi[i]
            circ = (vort.loc[bv_lat[xii]-r0:bv_lat[xii]+r0,
                             bv_lon[xii]-r0:bv_lon[xii]+r0]).sum()
        else:
            circ = (vort.loc[bv_lat[i]-r0:bv_lat[i]+r0, bv_lon[i]-r0:bv_lon[i]+r0]).sum()
        circ_arr[i] = circ

    return circ_arr



def plot_xy(data, dataset, var, plev=800, mlev=1.810000e+03, bv_centre=True):
    """
    produce a simple x-y plot given a dataset (e.g. ERA5) and a variable (e.g. vorticity)

    Args:
      data (xarray DataArray): multi-dimensional data array
      dataset (str): dataset to analyse (e.g. 4p4, N768, ERA5)
      var (str): variable to plot (e.g. vorticity, geopotential height)

    Kwargs:
      plev (int): pressure level for calculation (4p4)
      mlev (int): model level for calculation (N768)
      bv_centre (bool): overlay BV centre from Kevin Hodges' algorithm 
    """

    # separate into specific variables 
    if dataset == 'era5':
        w = data["w"].squeeze('time'); q = data["q"].squeeze('time')
        z = data["z"].squeeze('time'); u = data["u"].squeeze('time')
        v = data["v"].squeeze('time'); pv = data["pv"].squeeze('time')
        t = data["t"].squeeze('time'); z = data["z"].squeeze('time')
    elif dataset == '4p4':
        omg = data["omega"].squeeze('t_1'); q = data["q"].squeeze('t_1')
        u = data["u"].squeeze('t'); v = data["v"].squeeze('t')
        u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'
        temp = data["temp"].squeeze('t_1'); z = data["ht"].squeeze('t_1') / 10
        # interpolate all variables onto same grid before plotting 
        temp = temp.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")
        z = z.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")
        z.attrs['units'] = 'dam'
        # calculate additional diagnostics 
        th = mpcalc.potential_temperature(temp['p'], temp); th.attrs['units'] = 'K'
        mix = mpcalc.mixing_ratio_from_specific_humidity(q)
    else: # N768
        u = data["u"].squeeze("t"); v = data["v"].squeeze("t")
        pv = data["field83"].squeeze("t"); w = data["dz_dt"].squeeze("t")
        # interpolate all variables onto same grid before plotting 
        u = u.interp(longitude_1=pv["longitude"], method="linear")
        v = v.interp(latitude_1=pv["latitude"], method="linear")

    # calculate vector wind speed and relative vorticity 
    spd = mpcalc.wind_speed(u,v); spd.attrs['units'] = 'm/s'
    vort = mpcalc.vorticity(u, v, dx=None, dy=None) * 100000; vort.attrs['units'] = '10-5 s-1'

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
            dl = 2.0; lmin = -20; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            v.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': v.units},
                            cmap='seismic')            
        elif var == 'u':
            dl = 2.0; lmin = -20; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            u.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': u.units},
                            cmap='seismic')
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl)
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
    # 4.4 km MetUM forecast 
    elif dataset == '4p4':
        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl)
            vort.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': vort.units},
                               cmap='seismic')
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            omg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': omg.units},
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
    # N768 MetUM forecast 
    else: # N768
        if var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl)
            vort.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': vort.units},
                               cmap='seismic')
        elif var == 'w':
            dl = 0.5; lmin = -10; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            omg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': omg.units},
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
    skip = 20
    ax.quiver(data.longitude.values[::skip], data.latitude.values[::skip],
              u.values[::skip, ::skip], v.values[::skip, ::skip], angles='xy', scale=400)
    
    return fig
    
