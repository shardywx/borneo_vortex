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

import func_4p4 as fp

def main(inargs):
    """
    produce simple plots using 4.4 km MetUM data 
    """

    # read in 4.4 km MetUM data (xarray)
    data, dstr, tstr = fp.open_file(inargs.input_file, inargs.hr)

    # read in ERA5 reanalysis data (xarray)
    era_file = '/nobackup/earshar/borneo/bv_oct2018.grib'
    era5 = xr.open_dataset(era_file, engine="cfgrib").metpy.parse_cf()

    # subset the data     
    nn = [95.0, 120.0, -1.0, 17.0]

    # read in Kevin Hodges' Borneo vortex track data from text file
    df = pd.read_csv('/nobackup/earshar/borneo/bv_2018102112_track.csv',
                     na_filter=True,na_values="1.000000e+25")

    # convert time integers to datetime objects
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')

    # extract track information between 12Z on 21st and 26th October
    bv_lat = df.loc[0:20, "lat_vort"]; bv_lon = df.loc[0:20, "lon_vort"]
    bv_time = df.loc[0:20, "Time"]

    # only read in precipitation data if we're producing time series 
    if inargs.var == 'prcp':
        # 4.4 km MetUM data (Iris) --> 1 h output interval
        pfili  = '/nobackup/earshar/borneo/20181021T1200Z_SEA4_km4p4_ra1tld_pverb.pp'
        pcubes = iris.load(pfili)
        prcp_4p4 = xr.DataArray.from_iris(pcubes.extract('stratiform_rainfall_flux')[1])
        prcp_4p4 = prcp_4p4.sel(longitude=slice(nn[0],nn[1]),latitude=slice(nn[2],nn[3]) )

        # GPM data (xarray) --> 30 min output interval 
        gpm_file = '/nobackup/earshar/borneo/GPMHH_201810.nc'
        gpm = xr.open_dataset(gpm_file).sel(lon=slice(nn[0],nn[1]),lat=slice(nn[2],nn[3]) )
        prcp_gpm = gpm.precipitationCal

        # N768 MetUM data (xarray) --> 12 h output interval
        Tp = [24, 36, 48, 60, 72, 84, 96, 108, 120]
        # set up array to hold data at all output intervals 
        xr_dims = [len(Tp), prcp_4p4.latitude.shape[0], prcp_4p4.longitude.shape[0]]
        # retrieve time coordinate information from GPM data 
        time_co = prcp_gpm.resample(time="12H").sum().sel(time = slice('2018-10-22T12',
                                                                       '2018-10-26T12') )

        # initialise array 
        prcp_gl = xr.DataArray(np.ones(xr_dims),
                               dims=["time", "latitude", "longitude"],
                               coords={
                                   "time": time_co.time, 
                                   "latitude": prcp_4p4.latitude, 
                                   "longitude": prcp_4p4.longitude, 
                                   },
                           )

        # read in data for all output intervals 
        for i, t in enumerate(Tp):
            nfili='/nobackup/earshar/borneo/case_20181021T1200Z_N768/umglaa_pa{0:03d}.nc'.format(t-12)
            data_gl=xr.open_dataset(nfili)
            prcp=data_gl["tot_precip"].squeeze('t_1').squeeze('surface').sel(longitude=slice(nn[0],
                                        nn[1]),
                                        latitude=slice(nn[2],
                                                       nn[3]))
            # regrid before calculation 
            prcp_gl[i,:,:]=prcp.interp(longitude=prcp_4p4["longitude"],
                                       latitude=prcp_4p4["latitude"],
                                       method="linear")

        # also regrid GPM data
        prcp_gpm = prcp_gpm.interp(lon=prcp_4p4["longitude"],lat=prcp_4p4["latitude"],
                                   method="linear")

        # calculate accumulated rainfall at each time and plot 
        prcp_gpm = fp.acc_prcp(prcp_gpm, 'gpm', bv_lat, bv_lon, r0=inargs.r0)
        prcp_4p4 = fp.acc_prcp(prcp_4p4, '4p4', bv_lat, bv_lon, r0=inargs.r0)
        prcp_gl = fp.acc_prcp(prcp_gl, 'n768', bv_lat, bv_lon, r0=inargs.r0)
        # set up plot
        fig, ax = plt.subplots(figsize=(10,6))
        fili = './acc_prcp_oct2018_{0}deg.png'.format(inargs.r0)
        # produce time series of accumulated rainfall 
        ax.plot(bv_time[2:21:2], prcp_4p4, color='k', label='4p4 rainfall')
        ax.plot(bv_time[4:21:2], prcp_gl, color='b', label='N768 rainfall')
        ax.plot(bv_time[2:21:2], prcp_gpm, color='r', label='GPM rainfall')
        # add details (grid, legend, labels)
        ax.set(xlabel='Time', ylabel='Accumulated rainfall (mm)',
               title='Accumulated rainfall following the Borneo vortex')
        ax.grid(True); ax.legend(loc='upper left')
        fig.savefig(fili,dpi=200)
        exit()

    """
    if calculating circulation, keep all times for analysis  
    otherwise, only read in single time 
    """
    if inargs.var == 'circ':
        data = fp.subset(data, nn, var=inargs.var)
        era5 = fp.subset(era5, nn, var=inargs.var)
    else:
        data = fp.subset(data, nn, var=inargs.var, vtime=dstr)
        era5 = fp.subset(era5, nn, var=inargs.var, vtime=dstr)

    # read in N768 data using xarray
    if inargs.var == 'circ':
        gfile='/nobackup/earshar/borneo/case_20181021T1200Z_N768/umglaa_pe*.nc'
        gdata=xr.open_mfdataset(gfile, combine='by_coords', chunks={"t": 5}).metpy.parse_cf()
        gdata=gdata.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                         longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
        u_gl=gdata.u; v_gl=gdata.v; w_gl=gdata.dz_dt; pv_gl=gdata.field83

    else:
        Tp = int(inargs.hr)
        gfile='/nobackup/earshar/borneo/case_20181021T1200Z_N768/umglaa_pe{0:03d}.nc'.format(Tp-12)
        gdata=xr.open_dataset(gfile).metpy.parse_cf()
        gdata=gdata.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                         longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
            
    # interpolate all data onto 4p4 grid 
    if inargs.var == 'circ':
        # N768 onto 4p4 
        u_gl = gdata.u.interp(longitude_1=data.u["longitude"], 
                              latitude=data.u["latitude"], 
                              method="linear")
        v_gl = gdata.v.interp(longitude=data.v["longitude"],
                              latitude_1=data.v["latitude"], 
                              method="linear")
        pv_gl = gdata.field83.interp(longitude=data.v["longitude"],
                                     latitude=data.v["latitude"], 
                                     method="linear")
        w_gl  = gdata.dz_dt.interp(longitude=data.v["longitude"],
                                   latitude=data.v["latitude"], 
                                   method="linear")
        # ERA5 onto 4p4 
        u_era = era5.u.interp(longitude=data.u["longitude"],
                              latitude=data.u["latitude"],
                              method="linear")
        v_era = era5.v.interp(longitude=data.v["longitude"],
                              latitude=data.v["latitude"],
                              method="linear")

        # calculate circulation at each time and store in array for plotting 
        circ_era  = fp.calc_circ(u_era, v_era, bv_lat, bv_lon, plev=inargs.plev, r0=inargs.r0)
        circ_4p4  = fp.calc_circ(data.u, data.v, bv_lat, bv_lon, plev=inargs.plev, r0=inargs.r0)
        circ_n768 = fp.calc_circ(u_gl, v_gl, bv_lat, bv_lon, mlev=inargs.mlev, r0=inargs.r0)
        # set up plot  
        fig, ax = plt.subplots(figsize=(10,6))
        fili = './circ_oct2018_{0}deg.png'.format(inargs.r0)
        # produce time series of circulation 
        ax.plot(bv_time, circ_4p4, color='k', label='4p4 circulation')
        ax.plot(bv_time[4:21:2], circ_n768, color='b', label='N768 circulation')
        ax.plot(bv_time, circ_era, color='r', label='ERA5 circulation')
        # add details (grid, legend, labels)
        ax.set(xlabel='Time', ylabel='Normalised circulation',
               title='Circulation centred on the vortex')
        ax.grid(True); ax.legend(loc='upper left')
        fig.savefig(fili,dpi=200)
        exit()

    # interpolate N768 variables onto the same vertical levels (20/06/21)
    #
    # 

    # either produce x-y or x-z plot 

    if inargs.plot_type == 'xy': # horizontal cross-section ('xy')

        # focus on single pressure level and remove 1D time dimension
        if inargs.data == '4p4':
            data = data.sel(p=int(inargs.plev) )
        elif inargs.data == 'n768':
            data = gdata.sel(hybrid_ht_1=inargs.mlev)
        else: # ERA5 
            data = era5.sel(isobaricInhPa=int(inargs.plev) )
        
        """
        # get correct BV centre information to calculate circulation
        filter = bv_time==vort.coords['t'].data
        # find element of 'bv_time' that matches the time of our model data 
        time_match = bv_time.where(filter).notna()
        # retrive the index of the element above and use to get the BV centre
        ind = int(time_match.loc[time_match==True].index.values)

        # overlay Borneo vortex track (ERA-Interim)
        ax.plot(bv_lon[ind], bv_lat[ind], 'cD--', markersize=7)
        """

        # call the plotting function, and save the file 
        fig = fp.plot_xy(data, inargs.data, inargs.var)
        fili = './{2}{1}_{3}_{0}.png'.format(tstr, inargs.plev, inargs.var, inargs.data)
        fig.savefig(fili)

        print('output file created. moving onto next one...')

    else: # vertical cross section (inargs.plot_type == 'xz')

        start = (5.0, 105.0); end = (5.0, 115.0)
        u = data["u"].squeeze('t'); v = data["v"].squeeze('t')
        z = data["ht"].squeeze('t_1'); temp = data["temp"].squeeze('t_1')
        q = data["q"].squeeze('t_1'); omg = data["omega"].squeeze('t_1')
        print(data)
        exit()
        cross = cross_section(q, start, end).set_coords(('latitude','longitude'))
        print(cross)
        exit()

if __name__ == '__main__':
    description = 'Plot data from a 4.4 km MetUM forecast'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input_file", type=str, help="MetUM input file path")
    parser.add_argument("hr", type=int, help="Validity time (e.g. T+24)") 
    parser.add_argument("plot_type", type=str, default="xy", help="Type of output plot")
    parser.add_argument("var", type=str, help="Variable for plotting (e.g. vort)")
    parser.add_argument("--data", type=str, help="Dataset to plot (e.g. 4p4, ERA5)")
    parser.add_argument("--plev", type=int, default=800, help="Pressure level")
    parser.add_argument("--mlev", type=float, default=1.810000e+03, help="Model level height")
    parser.add_argument("--r0", type=float, default=3.0, help="Radius for circulation calc")

    args = parser.parse_args()

    main(args)
