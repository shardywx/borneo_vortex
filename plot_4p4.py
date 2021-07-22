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
from diagnosticSGsetup import output_names
from custom_cmap import *

import matplotlib
matplotlib.use('TkAgg')

import func_4p4 as fp

def main(inargs):
    """
    produce simple plots using 4.4 km MetUM data 
    """

    # read in 4.4 km MetUM data (xarray)
    data, sstr, dstr, tstr = fp.open_file(inargs.input_file, inargs.hr, pd)
    """
    # interpolate onto new set of pressure levels (14/07/21)
    plevs = [950, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150]
    data = data.assign_coords(p_levels=("p", plevs)).swap_dims({"p": "p_levels"})
    """

    # read in ERA5 reanalysis data (xarray)
    era_file = '/nobackup/earshar/borneo/bv_oct2018.grib'
    era5 = xr.open_dataset(era_file, engine="cfgrib").metpy.parse_cf()

    # subset the data     
    if inargs.data != 'n768':
        nn = [93.0, 123.0, -3.0, 20.0]
    else:
        nn = [93.0, 122.98, -2.98, 19.98]

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
            nfili='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pa{0:03d}.nc'.format(t-12)
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
        if inargs.data == 'era5':
            era5 = fp.subset(era5, nn, var=inargs.var, vtime=dstr)

    # read in N768 data using xarray
    if inargs.var == 'circ':
        gl_pe='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pe*.nc'
        # need to manually specify the coordinate reference system (CRS; ?)
        data_pe=xr.open_mfdataset(gl_pe, combine='by_coords', chunks={"t": 5}).metpy.parse_cf()
        gdata_pe=data_pe.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                              longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
        u_gl=gdata_pe.u; v_gl=gdata_pe.v; w_gl=gdata_pe.dz_dt; pv_gl=gdata_pe.field83

    else:
        if inargs.data == 'n768' or inargs.data == 'sgt':
            Tp = int(inargs.hr)
            gl_pe='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pe{0:03d}.nc'.format(Tp-12)
            gl_pb='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pb{0:03d}.nc'.format(Tp-12)
            # edit from here 
            #data_pe=xr.open_dataset(gl_pe).metpy.parse_cf()
            #data_pb=xr.open_dataset(gl_pb).metpy.parse_cf()
            data_pe=xr.open_dataset(gl_pe).metpy.assign_crs(
                grid_mapping_name='latitude_longitude',
                earth_radius=6371229.0)
            data_pb=xr.open_dataset(gl_pb).metpy.assign_crs(
                grid_mapping_name='latitude_longitude',
                earth_radius=6371229.0)
            gdata_pe=data_pe.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                                  longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
            gdata_pb=data_pb.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]) )


    # read in SGT tool data 
    variabledict={}
    for name in output_names:
        var_name='{}'.format(name)
        diri = '/nobackup/earshar/borneo/SGTool/N768/oct/{0}/filter_4_8/conv_g7x_v5/'.format(inargs.sgt)
        fn = '{0}/OUT_{1}_{2}_T{3:03d}.nc'.format(diri,name,sstr,inargs.hr)
        variabledict[name] = iris.load(fn)[0]
        variabledict[name].rename(name)

    # extract required variables 
    w_sgt = xr.DataArray.from_iris(variabledict['w'].extract('w') )
    u_sgt = xr.DataArray.from_iris(variabledict['u'].extract('u') )
    v_sgt = xr.DataArray.from_iris(variabledict['v'].extract('v') )
    ug_sgt = xr.DataArray.from_iris(variabledict['ug'].extract('ug') )
    vg_sgt = xr.DataArray.from_iris(variabledict['vg'].extract('vg') )
    ug_um  = xr.DataArray.from_iris(variabledict['ug_um'].extract('ug_um') )
    vg_um  = xr.DataArray.from_iris(variabledict['vg_um'].extract('vg_um') )

    # interpolate all data onto 4p4 grid 

    # N768 onto 4p4 
    if inargs.data == 'n768' or inargs.data == 'sgt':

        # create new array of height levels 
        ht_coords = np.arange(0, 17500, 250)

        # interpolate onto new grid (horizontal and vertical) 
        u_gl = gdata_pe.u.interp(#longitude_1=data.u["longitude"], 
                                 #latitude=data.u["latitude"],
                                 longitude_1=gdata_pe.v["longitude"],
                                 hybrid_ht_1=ht_coords,
                                 method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                ht_coords)).swap_dims({"hybrid_ht_1":
                                                       "height_levels"})
        v_gl = gdata_pe.v.interp(#longitude=data.v["longitude"],
                                 #latitude_1=data.v["latitude"], 
                                 latitude_1=gdata_pe.u["latitude"],
                                 hybrid_ht_1=ht_coords,
                                 method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                ht_coords)).swap_dims({"hybrid_ht_1":
                                                       "height_levels"})
        pv_gl = gdata_pe.field83.interp(#longitude=data.v["longitude"],
                                        #latitude=data.v["latitude"],
                                        hybrid_ht=ht_coords,
                                        method="linear").assign_coords(height_levels=("hybrid_ht",
                                        ht_coords)).swap_dims({"hybrid_ht":
                                                               "height_levels"})
        w_gl  = gdata_pe.dz_dt.interp(#longitude=data.v["longitude"],
                                      #latitude=data.v["latitude"],
                                      hybrid_ht=ht_coords,
                                      method="linear").assign_coords(height_levels=("hybrid_ht",
                                    ht_coords)).swap_dims({"hybrid_ht":
                                                           "height_levels"})
        if inargs.var != 'circ':
            q_gl  = gdata_pb.q.interp(#longitude=data.v["longitude"],
                                      #latitude=data.v["latitude"],
                                      hybrid_ht=ht_coords,
                                      method="linear").assign_coords(height_levels=("hybrid_ht",
                                    ht_coords)).swap_dims({"hybrid_ht":
                                                           "height_levels"})
            th_gl = gdata_pb.theta.interp(#longitude=data.v["longitude"],
                                          #latitude=data.v["latitude"],
                                          hybrid_ht=ht_coords,
                                          method="linear").assign_coords(height_levels=("hybrid_ht",
                                    ht_coords)).swap_dims({"hybrid_ht":
                                                           "height_levels"})

            q_gl = q_gl * 1000.; q_gl.attrs['units'] = 'g kg-1'

        # SGT onto 4p4 
        ht_sgt = gdata_pe.u['hybrid_ht_1'].data.astype('int32')

        u_sgt = u_sgt.interp(longitude=data.u["longitude"],
                             latitude=data.u["latitude"],
                             method="linear").assign_coords(height_levels=("model_level_number",
                                        ht_sgt)).swap_dims({"model_level_number":
                                                            "height_levels"})
        v_sgt = v_sgt.interp(longitude=data.u["longitude"],
                             latitude=data.u["latitude"],
                             method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})
        w_sgt = w_sgt.interp(longitude=data.u["longitude"],
                             latitude=data.u["latitude"],
                             method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})
        ug_sgt = ug_sgt.interp(longitude=data.u["longitude"],
                               latitude=data.u["latitude"],
                               method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})
        vg_sgt = vg_sgt.interp(longitude=data.u["longitude"],
                               latitude=data.u["latitude"],
                               method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})
        ug_um = ug_um.interp(longitude=data.u["longitude"],
                             latitude=data.u["latitude"],
                             method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})
        vg_um = vg_um.interp(longitude=data.u["longitude"],
                             latitude=data.u["latitude"],
                             method="linear").assign_coords(height_levels=("model_level_number",
                            ht_coords)).swap_dims({"model_level_number":
                                                   "height_levels"})

        u_sgt = u_sgt.interp(height_levels=ht_coords,method="linear")
        v_sgt = v_sgt.interp(height_levels=ht_coords,method="linear")
        w_sgt = w_sgt.interp(height_levels=ht_coords,method="linear")
        ug_sgt = ug_sgt.interp(height_levels=ht_coords,method="linear")
        vg_sgt = vg_sgt.interp(height_levels=ht_coords,method="linear")
        ug_um = ug_um.interp(height_levels=ht_coords,method="linear")
        vg_um = vg_um.interp(height_levels=ht_coords,method="linear")

        u_sgt.attrs['units'] = 'm/s'; v_sgt.attrs['units'] = 'm/s'; w_sgt.attrs['units'] = 'm/s'
        ug_sgt.attrs['units'] = 'm/s'; vg_sgt.attrs['units'] = 'm/s'
        ug_um.attrs['units'] = 'm/s'; vg_um.attrs['units'] = 'm/s'

    # ERA5 onto 4p4 
    u_era = era5.u.interp(longitude=data.u["longitude"],
                          latitude=data.u["latitude"],
                          method="linear")
    v_era = era5.v.interp(longitude=data.v["longitude"],
                          latitude=data.v["latitude"],
                          method="linear")
    w_era = era5.w.interp(longitude=data.v["longitude"],
                          latitude=data.v["latitude"],
                          method="linear")
    q_era = era5.q.interp(longitude=data.v["longitude"],
                          latitude=data.v["latitude"],
                          method="linear") * 1000.
    z_era = era5.z.interp(longitude=data.v["longitude"],
                          latitude=data.v["latitude"],
                          method="linear")
    pv_era = era5.pv.interp(longitude=data.v["longitude"],
                            latitude=data.v["latitude"],
                            method="linear")
    temp_era = era5.t.interp(longitude=data.v["longitude"],
                             latitude=data.v["latitude"],
                             method="linear")
    q_era.attrs['units'] = 'g kg-1'

    if inargs.var == 'circ':
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

    # either produce x-y or x-z plot 

    if inargs.plot_type == 'xy': # horizontal cross-section ('xy')

        # focus on single pressure level and remove 1D time dimension
        if inargs.data == '4p4':
            if inargs.var == 'pv':
                # read in data 
                u=data["u"].squeeze('t'); v=data["v"].squeeze('t'); temp=data["temp"].squeeze('t_1')
                u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'
                # interpolate onto same grid 
                temp=temp.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],
                                 method="linear")
                # calculate potential temperature 
                th = mpcalc.potential_temperature(temp['p'], temp); th.attrs['units'] = 'K'
                # calculate potential vorticity 
                pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
                pv.attrs['units'] = 'PVU'
                # select pressure level to analyse 
                pv=pv.sel(p=int(inargs.plev)); u=u.sel(p=int(inargs.plev))
                v=v.sel(p=int(inargs.plev)) 
            else:
                data = data.sel(p=int(inargs.plev) )
        elif inargs.data == 'era5':
            if inargs.var == 'pv':
                # read in data, as above for 4p4 MetUM
                u = era5["u"].squeeze('time'); v = era5["v"].squeeze('time')
                t = era5["t"].squeeze('time'); u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'
                # calculate potential temperature
                th = mpcalc.potential_temperature(t['isobaricInhPa'], t); th.attrs['units'] = 'K'
                # calculate potential vorticity 
                pv = mpcalc.potential_vorticity_baroclinic(th, th.isobaricInhPa, u, v) * 1000000.
                pv.attrs['units'] = 'PVU'
                # select pressure level to analyse 
                pv=pv.sel(isobaricInhPa=int(inargs.plev))
                u=u.sel(isobaricInhPa=int(inargs.plev)); v=v.sel(isobaricInhPa=int(inargs.plev))
            else:
                data = era5.sel(isobaricInhPa=int(inargs.plev) )
        else: # N768 or SGT tool 

            # N768 MetUM 
            u_gl=u_gl.sel(height_levels=inargs.mlev).squeeze("t")
            v_gl=v_gl.sel(height_levels=inargs.mlev).squeeze("t")
            w_gl=w_gl.sel(height_levels=inargs.mlev).squeeze("t")
            pv_gl=pv_gl.sel(height_levels=inargs.mlev).squeeze("t")
            q_gl=q_gl.sel(height_levels=inargs.mlev).squeeze("t")
            th_gl=th_gl.sel(height_levels=inargs.mlev).squeeze("t")
            ug_gl=ug_um.sel(height_levels=inargs.mlev)
            vg_gl=vg_um.sel(height_levels=inargs.mlev)

            # SGT tool 
            u_sgt=u_sgt.sel(height_levels=inargs.mlev); v_sgt=v_sgt.sel(height_levels=inargs.mlev)
            w_sgt=w_sgt.sel(height_levels=inargs.mlev); ug_sgt=ug_sgt.sel(height_levels=inargs.mlev)
            vg_sgt=vg_sgt.sel(height_levels=inargs.mlev)
            u_sgt.attrs['units'] = 'm/s'; v_sgt.attrs['units'] = 'm/s'
            ug_sgt.attrs['units'] = 'm/s'; vg_sgt.attrs['units'] = 'm/s'
            w_sgt.attrs['units'] = 'm/s' # will convert to cm/s within function (func_4p4) 

            """
            # interpolate N768 onto SGT grid --> has no effect (offset between grids still there)
            u_gl = u_gl.interp(longitude=u_sgt["longitude"],
                               latitude=u_sgt["latitude"],
                               method="linear")            
            v_gl = v_gl.interp(longitude=v_sgt["longitude"],
                               latitude=v_sgt["latitude"],
                               method="linear")
            w_gl = w_gl.interp(longitude=v_sgt["longitude"],
                               latitude=v_sgt["latitude"],
                               method="linear")            
            """

            # arrange in larger array (not a dictionary; what is it?)
            data_gl=[u_gl, v_gl, w_gl, pv_gl, q_gl, th_gl, ug_gl, vg_gl]
            data_sgt=[u_sgt, v_sgt, w_sgt, ug_sgt, vg_sgt] 
           
        
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
        if inargs.var == 'pv':
            fig = fp.plot_pv(pv, u, v, inargs.data)
        else:
            if inargs.data == 'n768':
                fig = fp.plot_xy(data_gl, inargs.data, inargs.var, wind=inargs.wind)
            elif inargs.data == 'sgt':
                fig = fp.plot_xy(data_sgt, inargs.data, inargs.var, wind=inargs.wind)
            else:
                fig = fp.plot_xy(data, inargs.data, inargs.var)

        if inargs.data == '4p4' or inargs.data == 'era5':
            fili = './{2}{1}_{3}_{0}.png'.format(tstr, inargs.plev, inargs.var, inargs.data)
        elif inargs.data == 'n768':
            fili = './{2}{1}_{3}_{4}_{0}.png'.format(tstr, inargs.mlev, inargs.var, 
                                                     inargs.data, inargs.wind)
        else: # SGT 
            fili = './{2}{1}_{3}_{4}_{5}_{0}.png'.format(tstr, inargs.mlev, 
                                                         inargs.var, inargs.data, 
                                                         inargs.sgt, inargs.wind)
        fig.savefig(fili)

        print('output file created. moving onto next one...')

    else: # vertical cross section (inargs.plot_type == 'xz')

        #fig = fp.plot_xz(data, inargs.data, inargs.var, inargs.plane)
        if int(inargs.hr) == 24:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = (1.0, 112.0); end = (12.0, 112.0)
                else:
                    start = (1.0, 112.02); end = (12.0, 112.06)
            else:
                start = (3.0, 105.00); end = (3.04, 118.00)
        elif int(inargs.hr) == 36:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = (1.0, 110.0); end = (12.0, 110.0)
                else:
                    start = (1.0, 110.02); end = (12.0, 110.06)
            else:
                start = (4.0, 103.00); end = (4.04, 116.00)
        elif int(inargs.hr) == 48:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = (2.0, 108.0); end = (12.0, 108.0)
                else:
                    start = (2.0, 108.02); end = (12.0, 108.06)
            else:
                start = (6.0, 102.00); end = (6.04, 114.00)
        elif int(inargs.hr) == 60:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = (2.0, 106.0); end = (12.0, 106.0)
                else:
                    start = (2.0, 106.02); end = (12.0, 106.06)
            else:
                start = (7.0, 101.00); end = (7.04, 114.00)
        elif int(inargs.hr) == 72:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = (2.0, 102.0); end = (12.0, 102.0)
                else:
                    start = (2.0, 102.02); end = (12.0, 102.06)
            else:
                start = (8.0, 98.00); end = (8.04, 111.00)

        if inargs.plane == 'ns':
            var_dim = 'longitude'
            ts = str(start[0]); tf = str(end[0])
            pstr = ts+'_'+tf
        else:
            var_dim = 'latitude'
            ns = str(start[1]); nf = str(end[1])
            pstr = ns+'_'+nf

        # remove unused dimensions 
        if inargs.data == '4p4':
            u = data["u"].squeeze('t'); v = data["v"].squeeze('t')
            z = data["ht"].squeeze('t_1'); temp = data["temp"].squeeze('t_1')
            q = data["q"].squeeze('t_1') * 1000.; q.attrs['units'] = 'g kg-1'
            omg = data["omega"].squeeze('t_1')

            # interpolate variables onto same grid 
            temp = temp.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")
            z = z.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")

            # calculate potential temperature
            th = mpcalc.potential_temperature(temp.p, temp)
            th_cs = th.sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           p=slice(950.0, 150.0) ).squeeze(var_dim)

        elif inargs.data == 'era5':
            w = era5["w"].squeeze('time'); q = era5["q"].squeeze('time') * 1000.
            z = era5["z"].squeeze('time'); u = era5["u"].squeeze('time')
            v = era5["v"].squeeze('time'); pv = era5["pv"].squeeze('time')
            temp = era5["t"].squeeze('time'); z = era5["z"].squeeze('time') / 10
            q.attrs['units']='g kg-1'

            # calculate potential temperature 
            th = mpcalc.potential_temperature(temp.isobaricInhPa, temp)
            th_cs = th.sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)

        elif inargs.data == 'n768':
            # remove unnecessary (time) dimension
            w = w_gl.squeeze('t'); u = u_gl.squeeze('t'); v = v_gl.squeeze('t')
            pv = pv_gl.squeeze('t'); q = q_gl.squeeze('t'); th = th_gl.squeeze('t')

            print(u)

            th_cs = th.sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           height_levels=slice(50, 13500) ).squeeze(var_dim)
        
        else: # SGT tool 
            w = w_sgt; u = u_sgt; v = v_sgt; ug = ug_sgt; vg = vg_sgt
            # calculate ageostrophic wind components (SGT tool)
            ua = u - ug; va = v - vg 
            # calculate ageostrophic wind components (N768 MetUM)
            ug_gl = ug_um; vg_gl = vg_um; ua_gl = u_gl - ug_gl; va_gl = v_gl - vg_gl
            # theta from N768 MetUM
            th_cs = th_gl.squeeze('t').sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           height_levels=slice(50, 13500) ).squeeze(var_dim)

        # temporary method for choosing variable to plot 
        if inargs.var == 'w':
            if inargs.data == '4p4':
                arr = omg.sel(latitude=slice(start[0],end[0]), 
                              longitude=slice(start[1],end[1]),
                              p=slice(950.0, 150.0) ).squeeze(var_dim)
                dl = 0.5; lmin = -15; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (Pa s-1)'
            elif inargs.data == 'era5':
                arr = omg.sel(latitude=slice(start[0],end[0]),
                              longitude=slice(start[1],end[1]),
                              isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
                dl = 0.5; lmin = -15; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (Pa s-1)'
            else: # N768 or SGT tool 
                arr = w.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            height_levels=slice(50, 13500) ).squeeze(var_dim) * 100.
                dl = 1.0; lmin = -20; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (cm s-1)'

        # add relative vorticity for N768 (08/07/21)
        elif inargs.var == 'vort':
            vort = mpcalc.vorticity(u, v, dx=None, dy=None) * 100000
            vort.attrs['units'] = '10-5 s-1'
            if inargs.data == '4p4':
                arr = vort.sel(latitude=slice(start[0],end[0]),
                               longitude=slice(start[1],end[1]),
                               p=slice(950.0, 150.0) ).squeeze(var_dim)            
            elif inargs.data == 'era5':
                arr = vort.sel(latitude=slice(start[0],end[0]),
                               longitude=slice(start[1],end[1]),
                               isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)                
            dl = 6; lmin = -180; lmax = -lmin + dl
            Cmap,norm,Levels = normalise_cmap(lmin,lmax,0,dl)
            cb_label = 'Relative vorticity (10-5 s-1)'

        elif inargs.var == 'q':
            if inargs.data == '4p4':
                arr = q.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            p=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = q.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)                
            else:
                arr = q.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; qmin = 1.0; qmax = 18.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
            cb_label = 'Specific humidity (kg kg-1)'

        # add relative humidity for N768 (08/07/21)
        elif inargs.var == 'rh':
            rh = mpcalc.relative_humidity_from_specific_humidity(q.p, temp, q) * 100.
            if inargs.data == '4p4':
                arr = rh.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             p=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = rh.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            dl = 5.0; rmin = 40.0; rmax = 100.0; Levels=np.arange(rmin,rmax+dl,dl); Cmap='BuPu'
            cb_label = 'Relative humidity (%)'

        elif inargs.var == 'u':
            if inargs.data == '4p4':
                arr = u.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            p=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = u.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            else: # N768 or SGT tool
                arr = u.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'Zonal wind (m s-1)'
            

        elif inargs.var == 'v':
            if inargs.data == '4p4':
                arr = v.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            p=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = v.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            else: # N768 or SGT tool
                arr = v.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'Meridional wind (m s-1)'

        elif inargs.var == 'ua':
            if inargs.data == 'sgt':
                arr = ua.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) ).squeeze(var_dim)
            elif inargs.data == 'n768':
                arr = ua_gl.sel(latitude=slice(start[0],end[0]),
                                longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'u_ageo (m s-1)'            

        elif inargs.var == 'va':
            if inargs.data == 'sgt':
                arr = va.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) ).squeeze(var_dim)
            elif inargs.data == 'n768':
                arr = va_gl.sel(latitude=slice(start[0],end[0]),
                                longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'v_ageo (m s-1)'

        elif inargs.var == 'ug':
            if inargs.data == 'sgt':
                        arr = ug.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) ).squeeze(var_dim)
            elif inargs.data == 'n768':
                arr = ug_gl.sel(latitude=slice(start[0],end[0]),
                                longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'u_geo (m s-1)'

        elif inargs.var == 'vg':
            if inargs.data == 'sgt':
                arr = vg.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) ).squeeze(var_dim)
            elif inargs.data =='n768':
                arr = vg_gl.sel(latitude=slice(start[0],end[0]),
                                longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) ).squeeze(var_dim)
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'v_geo (m s-1)'

        elif inargs.var == 'pv':
            if inargs.data == '4p4':
                pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
                arr = pv.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             p=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
                arr = pv.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            else:
                arr = pv.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) ).squeeze(var_dim)
            Levels = (-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0); Cmap='twilight'
            cb_label = 'Potential vorticity (PVU)'

        # date string 
        if inargs.data == '4p4':
            if inargs.var == 'u' or inargs.var == 'v' or inargs.var == 'vort':
                dstr = arr.t.dt.strftime("%Y%m%dT%H").values
            else:
                dstr = arr.t_1.dt.strftime("%Y%m%dT%H").values
        elif inargs.data == 'era5':
            dstr = arr.time.dt.strftime("%Y%m%dT%H").values
        elif inargs.data == 'n768':
            dstr = arr.t.dt.strftime("%Y%m%dT%H").values
        else:
            darr = data["u"].squeeze('t')
            dstr = darr.t.dt.strftime("%Y%m%dT%H").values

        # set up plot 
        fig = plt.figure(figsize=[9,6])
        ax = plt.axes()
        if inargs.data == 'sgt':
            fili='./vert_cs_{3}_{4}_{2}_{1}_{0}.png'.format(dstr,inargs.var,
                                                            pstr,inargs.data,inargs.sgt)
        else:
            fili='./vert_cs_{3}_{2}_{1}_{0}.png'.format(dstr, inargs.var, pstr, inargs.data)

        # filled contour plot (x-z)
        var_contour = plt.contourf(arr, levels=Levels, extend='max', cmap=Cmap)
        th_contour = plt.contour(th_cs, levels=np.arange(200, 500, 2), colors='black')
        var_cbar = fig.colorbar(var_contour)

        # tickmarks and labels
        ax.grid(True)
        xint = 1; yint = 1
        if inargs.plane == 'ns':
            ts = np.rint(arr.latitude[0].data); tf = np.rint(arr.latitude[-1].data)
            dim_size = len(arr.latitude)
            ax.set_xlabel('Latitude (degrees north)')
        else: 
            ts = np.rint(arr.longitude[0].data); tf = np.rint(arr.longitude[-1].data)
            dim_size = len(arr.longitude)
            ax.set_xlabel('Longitude (degrees east)')
            
        # x-axis tickmarks every 1º 
        ax.set_xlim(0, dim_size)
        if inargs.data == '4p4' or inargs.data == 'n768' or inargs.data == 'sgt':
            ax.set_xticks(np.arange(0, dim_size+1, 25) )
        elif inargs.data == 'era5':
            ax.set_xticks(np.arange(0, dim_size+1, 4) )
        ax.set_xticklabels(np.arange(ts, tf+1, yint) )

        # y-axis tickmarks and labels 
        if inargs.data == '4p4':
            plev_size = len(arr.p); ps = np.rint(arr.p[0].data); pf = np.rint(arr.p[-1].data)
            ax.set_yticks(np.arange(0, plev_size, 1) )
            ax.set_yticklabels(arr.p.data); ax.set_ylabel('Pressure (hPa)')
        elif inargs.data == 'era5':
            plev_size = len(arr.isobaricInhPa)
            ps = np.rint(arr.isobaricInhPa[0].data); pf = np.rint(arr.isobaricInhPa[-1].data)
            ax.set_yticks(np.arange(0, plev_size, 1) )
            ax.set_yticklabels(arr.isobaricInhPa.data); ax.set_ylabel('Pressure (hPa)')
        else:
            mlev_size = len(arr.height_levels)
            ax.set_yticks(np.arange(0, mlev_size, 4) )
            ax.set_yticklabels(arr.height_levels[::4].data); ax.set_ylabel('Height (m)')

        var_cbar.set_label(cb_label)
        plt.savefig(fili,dpi=200)

if __name__ == '__main__':
    description = 'Plot data from a 4.4 km MetUM forecast'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("input_file", type=str, help="MetUM input file path")
    parser.add_argument("hr", type=int, help="Validity time (e.g. T+24)") 
    parser.add_argument("plot_type", type=str, default="xy", help="Type of output plot")
    parser.add_argument("var", type=str, help="Variable for plotting (e.g. vort)")
    parser.add_argument("--data", type=str, help="Dataset to plot (e.g. 4p4, ERA5)")
    parser.add_argument("--plev", type=int, default=800, help="Pressure level")
    parser.add_argument("--mlev", type=float, default=1500, help="Model level height")
    parser.add_argument("--sgt", type=str, default="control", help="SGT tool RHS forcing")
    parser.add_argument("--wind", type=str, default="full", help="Wind vectors to overlay")
    parser.add_argument("--r0", type=float, default=3.0, help="Radius for circulation calc")
    parser.add_argument("--plane", type=str, default="ns", help="Plane for vert. cross-sec (NS/EW)")

    args = parser.parse_args()

    main(args)
