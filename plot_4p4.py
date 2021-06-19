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

    # read in 4.4 km data using xarray
    data, dstr, tstr = fp.open_file(inargs.input_file, inargs.hr)

    # subset the data 
    nn = [95.0, 120.0, -1.0, 17.0]
    
    """
    if calculating circulation, keep all times for analysis  
    otherwise, only read in single time 
    """
    if inargs.var == 'circ':
        data = fp.subset(data, nn, var=inargs.var)
    else:
        data = fp.subset(data, nn, var=inargs.var, vtime=dstr)

    # read in N768 data using xarray
    if inargs.var == 'circ':
        gfile = '/nobackup/earshar/borneo/case_20181021T1200Z_N768/umglaa_pe*.nc'
        gdata = xr.open_mfdataset(gfile, combine='by_coords', chunks={"t": 5}).metpy.parse_cf()
        gdata = gdata.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                           longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
        u_gl = gdata.u; v_gl = gdata.v; w_gl = gdata.dz_dt; pv_gl = gdata.field83

    else:
        Tp = int(inargs.hr)
        gfile = '/nobackup/earshar/borneo/case_20181021T1200Z_N768/umglaa_pe{0:03d}.nc'.format(Tp-12)
        gdata = xr.open_dataset(gfile).metpy.parse_cf()
        gdata = gdata.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                               longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
            
    # read in Kevin Hodges' Borneo vortex track data from text file
    df = pd.read_csv('/nobackup/earshar/borneo/bv_2018102112_track.csv',
                     na_filter=True,na_values="1.000000e+25")

    # convert time integers to datetime objects 
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')

    # extract track information between 12Z on 21st and 26th October
    bv_lat = df.loc[0:20, "lat_vort"]; bv_lon = df.loc[0:20, "lon_vort"]
    bv_time = df.loc[0:20, "Time"]

    # if analysing circulation
    if inargs.var == 'circ':
        # calculate circulation at each time and store in array for plotting 
        circ_arr = fp.calc_circ(data, bv_lat, bv_lon, plev=inargs.plev, r0=inargs.r0)
        circ_gl_arr = fp.calc_circ(gdata, bv_lat, bv_lon, glb=True, mlev=inargs.mlev, r0=inargs.r0)
        # set up plot 
        fig, ax = plt.subplots(figsize=(10,6))
        fili = './circ_oct2018_{0}deg.png'.format(inargs.r0)
        # produce time series of circulation
        ax.plot(bv_time, circ_arr, color='k', label='4p4 circulation')
        ax.plot(bv_time[4:21:2], circ_gl_arr, color='b', label='N768 circulation')
        # add details (grid, legend, labels)
        ax.set(xlabel='Time', ylabel='Normalised circulation', 
               title='Circulation centred on the vortex')
        ax.grid(True); ax.legend(loc='upper left')
        fig.savefig(fili,dpi=200)
        exit()

    # either produce x-y or x-z plot 

    if inargs.plot_type == 'xy': # horizontal cross-section ('xy')

        # focus on single pressure level and remove 1D time dimension
        data = data.sel(p=int(inargs.plev) ); gdata = gdata.sel(hybrid_ht_1=inargs.mlev)

        # separate into specific variables 
        omg = data["omega"].squeeze('t_1'); q = data["q"].squeeze('t_1')
        u = data["u"].squeeze('t'); v = data["v"].squeeze('t'); spd = mpcalc.wind_speed(u,v)
        u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'; spd.attrs['units'] = 'm/s'
        temp = data["temp"].squeeze('t_1'); z = data["ht"].squeeze('t_1') / 10

        # interpolate all variables onto same grid before plotting 
        temp = temp.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")
        z = z.interp(longitude_1=u["longitude"], latitude_1=u["latitude"], method="linear")
        z.attrs['units'] = 'dam'

        # calculate additional diagnostics 
        vort = mpcalc.vorticity(u, v, dx=None, dy=None) * 100000 ;vort.attrs['units'] = '10-5 s-1'
        th = mpcalc.potential_temperature(temp['p'], temp); th.attrs['units'] = 'K'
        mix = mpcalc.mixing_ratio_from_specific_humidity(q)
        #w = mpcalc.vertical_velocity(omg,omg['p'],temp,mixing_ratio=mix)

        # repeat for global N768 data 
        u_gl = gdata["u"].squeeze("t"); v_gl = gdata["v"].squeeze("t")
        pv_gl = gdata["field83"].squeeze("t"); w_gl = gdata["dz_dt"].squeeze("t")
        u_gl = u_gl.interp(longitude_1=pv_gl["longitude"], method="linear")
        v_gl = v_gl.interp(latitude_1=pv_gl["latitude"], method="linear")
        vort_gl = mpcalc.vorticity(u_gl, v_gl, dx=None, dy=None) * 100000
        vort_gl.attrs['units'] = '10-5 s-1'

        # set up plot and axes 
        fig = plt.figure(figsize=[9,6])
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))

        # plot the data 
        if inargs.var == 'spd':
            levs = np.arange(2.0, 30.0, 2.0)
            spd.plot.contourf(ax=ax, levels=levs, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': spd.units},
                              cmap=cmocean.cm.haline_r)
        elif inargs.var == 'vort':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl)
            vort.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                               cbar_kwargs={'label': vort.units},
                               cmap='seismic')
        elif inargs.var == 'vort_gl':
            dl = 2; lmin = -80; lmax = -lmin + dl
            Cmap, norm, Levels = normalise_cmap(lmin,lmax,0,dl)
            vort_gl.plot.contourf(ax=ax, levels=Levels, transform=ccrs.PlateCarree(),
                                  cbar_kwargs={'label': vort_gl.units},
                                  cmap='seismic')        
        elif inargs.var == 'omg':
            dl = 0.5; lmin = -10; lmax = -lmin + dl; Levels = np.arange(lmin, lmax, dl)
            omg.plot.contourf(ax=ax, levels=Levels, extend='max', transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': omg.units},
                              cmap='seismic')
        elif inargs.var == 'geo':
            levs = np.arange(197.0, 208.0, 1.0)
            z.plot.contourf(ax=ax, extend='max', levels=levs, transform=ccrs.PlateCarree(),
                            cbar_kwargs={'label': z.units},
                            cmap='plasma_r')
        elif inargs.var == 'th':
            #levs = np.arange(280.0, 380.0, 2.0)
            th.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),
                             cbar_kwargs={'label': th.units},
                             cmap='plasma_r')

        # add coastlines, gridlines and tickmarks
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),linewidth=0.75)
        xint = 5; yint = 3
        ax.set_xticks(np.arange(n0, n1+xint, xint))
        ax.set_xticklabels(np.arange(n0, n1+xint, xint))
        ax.set_yticks(np.arange(t0, t1+yint, yint))
        ax.set_yticklabels(np.arange(t0, t1+yint, yint))
        plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

        # add horizontal wind barbs 
        skip = 20
        ax.quiver(data.longitude.values[::skip], data.latitude.values[::skip], 
                  u.values[::skip, ::skip], v.values[::skip, ::skip], angles='xy', scale=400)

        # get correct BV centre information to calculate circulation
        filter = bv_time==vort.coords['t'].data
        # find element of 'bv_time' that matches the time of our model data 
        time_match = bv_time.where(filter).notna()
        # retrive the index of the element above and use to get the BV centre
        ind = int(time_match.loc[time_match==True].index.values)

        # overlay Borneo vortex track (ERA-Interim)
        ax.plot(bv_lon[ind], bv_lat[ind], 'cD--', markersize=7)

        # save the file 
        fili='./{2}{1}_{0}.png'.format(tstr, inargs.plev, inargs.var)
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
    parser.add_argument("--plev", type=int, default=800, help="Pressure level")
    parser.add_argument("--mlev", type=float, default=1.810000e+03, help="Model level height")
    parser.add_argument("--r0", type=float, default=3.0, help="Radius for circulation calc")
    #parser.add_argument("--all_times", action="store_true", default=False, 
    #                    help="Analyse single time, or all times")

    args = parser.parse_args()

    main(args)
