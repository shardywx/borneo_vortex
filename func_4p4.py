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

    # use select function to subset data                                                 
    if var == 'circ':
        data = data.sel( longitude = slice(bounds[0], bounds[1]),
                         latitude = slice(bounds[2], bounds[3]),
                         longitude_1 = slice(bounds[0], bounds[1]),
                         latitude_1 = slice(bounds[2], bounds[3]))
    else:
        data = data.sel( t_1=slice(vtime,vtime), t=slice(vtime,vtime),
                         longitude=slice(bounds[0], bounds[1]),
                         latitude=slice(bounds[2], bounds[3]),
                         longitude_1=slice(bounds[0], bounds[1]),
                         latitude_1=slice(bounds[2], bounds[3]) )

    return data


def calc_circ(u, v, bv_lat, bv_lon, glb=False, plev=800, mlev=1.810000e+03, r0=3.0):
    """                                                                                  
    calculate circulation following the vortex for multiple validity times               
    storm the values in an array for plotting                                            
                                                                                         
    Args:
      u (Xarray DataArray): multi-dimensional data array (zonal wind)
      v (Xarray DataArray): multi-dimensional data array (meridional wind)
      bv_lat (Pandas DataFrame): vortex centre latitude
      bv_lon (Pandas DataFrame): vortex centre longitude
                                                                                         
    Kwargs:                                                                              
      glb (boolean): plot regional (4p4) or global (N768) data                           
      plev (int): pressure level for calculation (4p4)                                   
      mlev (int): model level for calculation (N768)                                     
      r0 (float): radius for calculation (degrees)
    """

    # get size of time dimension                                                         
    ntimes = u.t.shape[0]
    times = np.arange(0, ntimes)

    # initialise array (better way to do this?)                                          
    if glb:
        t_ind = times[:]
        circ_arr = xr.DataArray(t_ind)
        xi = [4,6,8,10,12,14,16,18,20]
    else:
        t_ind = times[::2]
        circ_arr = xr.DataArray(t_ind)

    # loop over times                                                                     
    for i, it in enumerate(t_ind):
        # read in wind components on single level and calculate relative vorticity        
        if glb:
            u0 = u[it,:,:,:].sel(hybrid_ht_1=int(mlev) )
            v0 = v[it,:,:,:].sel(hybrid_ht_1=int(mlev) )
            vort = mpcalc.vorticity(u0, v0, dx = None, dy = None) * 10000
        else:
            u0 = u[it,:,:,:].sel(p=int(plev) )
            v0 = v[it,:,:,:].sel(p=int(plev) )
            vort = mpcalc.vorticity(u0, v0, dx = None, dy = None) * 10000

        #vort = mpcalc.vorticity(u, v, dx = None, dy = None) * 100000
        vort.attrs['units'] = '10-5 s-1'

        # calculate circulation following the vortex                                      
        if glb:
            xii = xi[i]
            circ = (vort.loc[bv_lat[xii]-r0:bv_lat[xii]+r0,
                             bv_lon[xii]-r0:bv_lon[xii]+r0]).sum()
        else:
            circ = (vort.loc[bv_lat[i]-r0:bv_lat[i]+r0, bv_lon[i]-r0:bv_lon[i]+r0]).sum()
        circ_arr[i] = circ

    return circ_arr
