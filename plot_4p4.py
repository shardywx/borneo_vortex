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
from matplotlib.patches import Rectangle
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
    if inargs.data == '4p4':
        sstr, dstr, tstr, data_pc, data_pd = fp.open_file(inargs.input_file, inargs.hr, ftype=inargs.data)
    elif inargs.var == 'circ':
        sstr, dstr, tstr, data_pc, data_pd = fp.open_file(inargs.input_file, inargs.hr, ftype='4p4')
    else:
        sstr, dstr, tstr = fp.open_file(inargs.input_file, inargs.hr)

    # read in ERA5 reanalysis data (xarray)
    if inargs.data == 'era5':
        era_file = '/nobackup/earshar/borneo/bv_oct2018.grib'
        era5 = xr.open_dataset(era_file, engine="cfgrib").metpy.parse_cf()

    # subset the data     
    if inargs.hr == -12:
        nn = [103.0, 133.0, -3.0, 20.0]
    else:
        nn = [93.0, 123.0, -3.0, 20.0]

    # read in Kevin Hodges' Borneo vortex track data from text file
    df = pd.read_csv('/nobackup/earshar/borneo/bv_2018102112_track.csv',
                     na_filter=True,na_values="1.000000e+25")

    # convert time integers to datetime objects
    df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')

    # extract track information between 12Z on 21st and 26th October
    bv_lat = df.loc[0:20, "lat_vort"]; bv_lon = df.loc[0:20, "lon_vort"]
    bv_time = df.loc[0:20, "Time"]

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

    # read in Himawari data                                                                  
    if inargs.var == 'hima':
        hstr = dstr.strftime("%Y%m%d_%H00") # create string to read in data from single file 
        hima_nc = '/nobackup/earshar/borneo/himawari/himawari_10.4_{0}.4p4km.nc'.format(hstr)
        fili = './himawari_bt_{0}.png'.format(hstr)
        himawari = xr.open_dataset(hima_nc).metpy.parse_cf()
        himawari = himawari.sel(longitude=slice(nn[0],nn[1]),latitude=slice(nn[2],nn[3]) )
        t_bright = himawari.T_b

        # set up plot of Himawari data                                                           
        fig = plt.figure(figsize=(10,6))                                                         
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))              
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),edgecolor='blue',linewidth=1.0)
        plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

        # overlay lat/lon labels 
        xint = 5; yint = 3
        ns = np.rint(t_bright.longitude[0].data); nf = np.rint(t_bright.longitude[-1].data)
        ts = np.rint(t_bright.latitude[0].data); tf = np.rint(t_bright.latitude[-1].data)
        ax.set_xticks(np.arange(ns, nf+1, xint))
        ax.set_xticklabels(np.arange(ns, nf+1, xint))
        ax.set_yticks(np.arange(ts, tf+1, yint))
        ax.set_yticklabels(np.arange(ts, tf+1, yint))

        # filled contour plot                                                                    
        t_bright.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),                
                               cbar_kwargs={'label': 'K'},                                     
                               cmap='gist_yarg')

        # find index of vortex centre information that matches the chosen time 
        filter = bv_time==t_bright.coords['time'].data
        time_match = bv_time.where(filter).notna()
        ind = int(time_match.loc[time_match==True].index.values)

        # overlay vortex box position 
        ax.add_patch( Rectangle( (bv_lon[ind]-inargs.r0, bv_lat[ind]-inargs.r0),
                                 2*inargs.r0, 2*inargs.r0, linewidth=2,
                                 facecolor='none', edgecolor='c') )

        # save and exit
        plt.title('')
        plt.savefig(fili,dpi=200)
        exit()


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
        prcp_gpm = fp.acc_prcp(prcp_gpm, 'gpm', bv_lat, bv_lon, r0=inargs.r0, plt_prcp=False)
        prcp_4p4 = fp.acc_prcp(prcp_4p4, '4p4', bv_lat, bv_lon, r0=inargs.r0, plt_prcp=False)
        prcp_gl = fp.acc_prcp(prcp_gl, 'n768', bv_lat, bv_lon, r0=inargs.r0, plt_prcp=False)

        # set up plot
        fig, ax = plt.subplots(figsize=(10,6))
        fili = './acc_prcp_oct2018_{0}deg.png'.format(inargs.r0)
        # produce time series of accumulated rainfall 
        ax.plot(bv_time, np.zeros(21), color='k', alpha=0.0)
        ax.plot(bv_time[2:21:2], prcp_4p4, color='k', label='4.4 km MetUM')
        ax.plot(bv_time[4:21:2], prcp_gl, color='b', label='Global MetUM')
        ax.plot(bv_time[2:21:2], prcp_gpm, color='r', label='GPM-IMERG')
        # add details (grid, legend, labels)
        ax.set(xlabel='Time', ylabel='Accumulated rainfall (mm)',
               title='Accumulated rainfall following the vortex')
        ax.grid(True); ax.legend(loc='upper left')
        fig.savefig(fili,dpi=200)
        exit()

    # read in N768 data using xarray
    if inargs.var == 'circ' or inargs.var == 'ubar':
        # N768 file path 
        gl_pe='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pe*.nc'
        # need to manually specify the coordinate reference system (CRS; ?)
        data_pe=xr.open_mfdataset(gl_pe, combine='by_coords', chunks={"t": 5}).metpy.parse_cf()
        # subset data 
        gdata_pe=data_pe.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                              longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
        u_gl=gdata_pe.u; v_gl=gdata_pe.v; w_gl=gdata_pe.dz_dt; pv_gl=gdata_pe.field83

        # calculate and plot zonally-averaged zonal wind (u_bar)
        if inargs.var == 'ubar':
        
            # set up height coordinates
            ht_coords = gdata_pe.u['hybrid_ht_1'].data.astype('int32')
            # interpolate to new lat/lon grid 
            u_bar = u_gl.interp(longitude_1=v_gl["longitude"],
                                hybrid_ht_1=ht_coords,
                                method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                ht_coords)).swap_dims({"hybrid_ht_1":
                                                       "height_levels"})
            # calculate mean over longitude and time 
            u_bar = u_bar.sel(longitude=slice(95.0, 120.0), 
                              latitude=slice(0.0,
                    15.0) ).mean(dim=['longitude','t']).sel(height_levels=slice(50, 15000))

            # interpolate to new levels straight before plotting
            ht_coords = np.arange(0, 15000, 250)
            arr = u_bar.interp(height_levels=ht_coords,method="linear")

            # interpolate onto regular lat/lon grid before plotting 
            lat_dim = np.arange(np.rint(arr.latitude[0].data),
                                np.rint(arr.latitude[-1].data)+0.25, 0.25)
            arr = arr.interp(latitude=lat_dim, method="linear")

            # colour map and contour levels
            dl = 1.0; vmin = -20.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'Mean zonal wind (m s-1)'

            # set up plot and axes 
            fig = plt.figure(figsize=[9,6])
            ax = plt.axes()
            var_contour = plt.contourf(arr, levels=Levels, extend='max', cmap=Cmap)
            var_cbar = fig.colorbar(var_contour)

            # set up tickmarks and labels
            ax.grid(True)
            xint = 1; yint = 1
            ts = np.rint(arr.latitude[0].data); tf = np.rint(arr.latitude[-1].data)
            dim_size = len(arr.latitude); ax.set_xlabel('Latitude (degrees north)')
            ax.set_xticks(np.arange(0, dim_size+1, 4) )
            ax.set_xticklabels(np.arange(ts, tf+1, yint) )
            mlev_size = len(arr.height_levels)
            ax.set_yticks(np.arange(0, mlev_size, 4) )
            ax.set_yticklabels(arr.height_levels[::4].data); ax.set_ylabel('Height (m)')
            var_cbar.set_label(cb_label)
            fili='./ubar_n768.png'
            plt.savefig(fili,dpi=200)
            exit()

        # calculate and plot area-averaged relative vorticity (circulation)
        elif inargs.var == 'circ':

            era5 = fp.subset(era5, nn, var=inargs.var)
            data = fp.subset(data_pd, nn, var=inargs.var)

            circ_era  = fp.calc_circ(era5.u, era5.v, bv_lat, bv_lon, plev=inargs.plev, r0=inargs.r0)
            circ_4p4  = fp.calc_circ(data.u, data.v, bv_lat, bv_lon, plev=inargs.plev, r0=inargs.r0)
            circ_gl   = fp.calc_circ(u_gl, v_gl, bv_lat, bv_lon, mlev=inargs.mlev, r0=inargs.r0)

            # set up plot
            fig, ax = plt.subplots(figsize=(10,6))
            fili = './circ_oct2018_{0}deg.png'.format(inargs.r0)
            # produce time series of circulation
            ax.plot(bv_time, circ_4p4, color='k', label='4.4 km MetUM')
            ax.plot(bv_time[4:21:2], circ_gl, color='b', label='Global MetUM')
            ax.plot(bv_time, circ_era, color='r', label='ERA5 reanalysis')
            # add details (grid, legend, labels)
            var_str = 'Area-averaged relative vorticity'
            ax.set(xlabel='Time', 
               ylabel=r'Area-averaged relative vorticity ($\mathregular{10}^{-6}$ s$\mathregular{^{-1}}$)',
               title='Area-averaged relative vorticity following the vortex')
            ax.grid(True); ax.legend(loc='upper left')
            fig.savefig(fili,dpi=200)
            exit()

    # if not calculating area-averaged vorticity or zonally-averaged zonal wind 
    else:

        if inargs.data == '4p4':
            data_pc = fp.subset(data_pc, nn, var=inargs.var, vtime=dstr)
            data_pd = fp.subset(data_pd, nn, var=inargs.var, vtime=dstr)
        elif inargs.data == 'era5':
            era5 = fp.subset(era5, nn, var=inargs.var, vtime=dstr)
        else:
        # read in N768 and SGT tool data (deleted on 20/09/21 --> replaced from scratch)
            variabledict={}
            for name in output_names:
                var_name='{}'.format(name)
                diri='/nobackup/earshar/borneo/SGTool/N768/oct/{0}/filter_4_8/conv_g7x_v5/'.format(inargs.sgt)
                fn = '{0}/OUT_{1}_{2}_T{3:03d}.nc'.format(diri,name,sstr,inargs.hr)
                variabledict[name] = iris.load(fn)[0]
                variabledict[name].rename(name)

            # SGT tool 
            w_sgt = xr.DataArray.from_iris(variabledict['w'].extract('w')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            v_sgt = xr.DataArray.from_iris(variabledict['v'].extract('v')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            u_sgt = xr.DataArray.from_iris(variabledict['u'].extract('u')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            ug_sgt=xr.DataArray.from_iris(variabledict['ug'].extract('ug')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            vg_sgt=xr.DataArray.from_iris(variabledict['vg'].extract('vg')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            ug_um=xr.DataArray.from_iris(variabledict['ug_um'].extract('ug_um')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))
            vg_um=xr.DataArray.from_iris(variabledict['vg_um'].extract('vg_um')).sel(longitude=slice(nn[0],
                                        nn[1]),latitude=slice(nn[2],nn[3]))

            # N768 MetUM 
            Tp = int(inargs.hr)
            gl_pe='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pe{0:03d}.nc'.format(Tp-12)
            gl_pb='/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pb{0:03d}.nc'.format(Tp-12)
            data_pe=xr.open_dataset(gl_pe).metpy.assign_crs(grid_mapping_name='latitude_longitude',
                                                            earth_radius=6371229.0)
            data_pb=xr.open_dataset(gl_pb).metpy.assign_crs(grid_mapping_name='latitude_longitude',
                                                            earth_radius=6371229.0)
            gdata_pe=data_pe.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]),
                                  longitude_1=slice(nn[0], nn[1]), latitude_1=slice(nn[2], nn[3]) )
            gdata_pb=data_pb.sel( longitude=slice(nn[0], nn[1]), latitude=slice(nn[2], nn[3]) )


    # interpolate all data onto 4p4 grid 

    # N768 onto 4p4 
    if inargs.data == 'n768' or inargs.data == 'sgt':


        # create new array of height levels 
        ht_coords = np.arange(0, 17500, 250)
        if inargs.hr == 0 or inargs.hr == 12: # T+0 and T+12 (data slightly different than other times)
            ht_coords = gdata_pe.u['hybrid_ht'].data.astype('int32')
            # interpolate onto new grid (horizontal and vertical) 
            u_gl = gdata_pe.u.interp(hybrid_ht=ht_coords,
                                     method="linear").assign_coords(height_levels=("hybrid_ht",
                                    ht_coords)).swap_dims({"hybrid_ht":
                                    "height_levels"})
            v_gl = gdata_pe.v.interp(longitude_1=gdata_pe.u["longitude"],
                                     latitude_1=gdata_pe.u["latitude"],
                                     hybrid_ht=ht_coords,
                                     method="linear").assign_coords(height_levels=("hybrid_ht",
                                    ht_coords)).swap_dims({"hybrid_ht":
                                                           "height_levels"})
            pv_gl = gdata_pe.field83.interp(hybrid_ht_1=ht_coords,
                                            method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                            ht_coords)).swap_dims({"hybrid_ht_1":
                                                                   "height_levels"})
            w_gl  = gdata_pe.dz_dt.interp(hybrid_ht_1=ht_coords,
                                          method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                        ht_coords)).swap_dims({"hybrid_ht_1":
                                                               "height_levels"})

        else: # T+24 onwards 
            ht_coords = gdata_pe.u['hybrid_ht_1'].data.astype('int32')
            # interpolate onto new grid (horizontal and vertical)
            u_gl = gdata_pe.u.interp(longitude_1=gdata_pe.v["longitude"],
                                     hybrid_ht_1=ht_coords,
                                     method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                    ht_coords)).swap_dims({"hybrid_ht_1":
                                    "height_levels"})
            v_gl = gdata_pe.v.interp(latitude_1=gdata_pe.u["latitude"],
                                     hybrid_ht_1=ht_coords,
                                     method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                    ht_coords)).swap_dims({"hybrid_ht_1":
                                                           "height_levels"})
            pv_gl = gdata_pe.field83.interp(hybrid_ht=ht_coords,
                                            method="linear").assign_coords(height_levels=("hybrid_ht",
                                            ht_coords)).swap_dims({"hybrid_ht":
                                                                   "height_levels"})
            w_gl  = gdata_pe.dz_dt.interp(hybrid_ht=ht_coords,
                                          method="linear").assign_coords(height_levels=("hybrid_ht",
                                        ht_coords)).swap_dims({"hybrid_ht":
                                                               "height_levels"})

        if inargs.var != 'circ' and inargs.var != 'ubar':
            q_gl  = gdata_pb.q.interp(hybrid_ht=ht_coords,
                                      method="linear").assign_coords(height_levels=("hybrid_ht",
                                        ht_coords)).swap_dims({"hybrid_ht":
                                                               "height_levels"})
            th_gl = gdata_pb.theta.interp(hybrid_ht=ht_coords,
                                          method="linear").assign_coords(height_levels=("hybrid_ht",
                                        ht_coords)).swap_dims({"hybrid_ht":
                                                               "height_levels"})

            q_gl = q_gl * 1000.; q_gl.attrs['units'] = 'g kg-1'

        # SGT onto 4p4 (not for now)
        if inargs.hr == 0 or inargs.hr == 12:
            ht_sgt = gdata_pe.u['hybrid_ht'].data.astype('int32')
        else:
            ht_sgt = gdata_pe.u['hybrid_ht_1'].data.astype('int32')

        u_sgt = u_sgt.assign_coords(height_levels=("model_level_number",
                                    ht_sgt)).swap_dims({"model_level_number":
                                                        "height_levels"})
        v_sgt = v_sgt.assign_coords(height_levels=("model_level_number",
                                    ht_coords)).swap_dims({"model_level_number":
                                                           "height_levels"})
        w_sgt = w_sgt.assign_coords(height_levels=("model_level_number",
                                    ht_coords)).swap_dims({"model_level_number":
                                                           "height_levels"})
        ug_sgt = ug_sgt.assign_coords(height_levels=("model_level_number",
                                    ht_coords)).swap_dims({"model_level_number":
                                                           "height_levels"})
        vg_sgt = vg_sgt.assign_coords(height_levels=("model_level_number",
                                    ht_coords)).swap_dims({"model_level_number":
                                                           "height_levels"})
        ug_um = ug_um.assign_coords(height_levels=("model_level_number",
                                    ht_coords)).swap_dims({"model_level_number":
                                                           "height_levels"})
        vg_um = vg_um.assign_coords(height_levels=("model_level_number",
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

    # ERA5 onto 4p4 (not for now)
    if inargs.data == 'era5':
        u_era = era5.u; v_era = era5.v; w_era = era5.w
        q_era = era5.q * 1000.; q_era.attrs['units'] = 'g kg-1'
        z_era = era5.z; pv_era = era5.pv; temp_era = era5.t


    ### either produce x-y or x-z plot ###

    if inargs.plot_type == 'xy': # horizontal cross-section ('xy')

        # focus on single pressure level and remove 1D time dimension
        if inargs.data == '4p4':
            if inargs.var == 'pv':
                # read in data, combining two streams
                data = xr.combine_by_coords([data_pc.squeeze(['t', 't_1']), 
                                             data_pd.squeeze(['t', 't_1'])])
                data = data.reindex(p=data.p[::-1])
                u = data["u"]; v = data["v"]; temp = data["temp"]
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
                pv = pv.sel(p=int(inargs.plev)); u = u.sel(p=int(inargs.plev))
                v = v.sel(p=int(inargs.plev)) 
            else:
                data = xr.combine_by_coords([data_pc.squeeze(['t', 't_1']),
                                             data_pd.squeeze(['t', 't_1'])])
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
            if inargs.hr == 0 or inargs.hr == 12:
                u_gl=u_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                v_gl=v_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                w_gl=w_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                pv_gl=pv_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                q_gl=q_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                th_gl=th_gl.sel(height_levels=inargs.mlev,method="nearest").sel(t=dstr)
                ug_gl=ug_um.sel(height_levels=inargs.mlev,method="nearest")
                vg_gl=vg_um.sel(height_levels=inargs.mlev,method="nearest")
            else:
                u_gl=u_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                v_gl=v_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                w_gl=w_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                pv_gl=pv_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                q_gl=q_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                th_gl=th_gl.sel(height_levels=inargs.mlev,method="nearest").squeeze("t")
                ug_gl=ug_um.sel(height_levels=inargs.mlev,method="nearest")
                vg_gl=vg_um.sel(height_levels=inargs.mlev,method="nearest")

            # SGT tool 
            u_sgt=u_sgt.sel(height_levels=inargs.mlev,method="nearest")
            v_sgt=v_sgt.sel(height_levels=inargs.mlev,method="nearest")
            w_sgt=w_sgt.sel(height_levels=inargs.mlev,method="nearest")
            ug_sgt=ug_sgt.sel(height_levels=inargs.mlev,method="nearest")
            vg_sgt=vg_sgt.sel(height_levels=inargs.mlev,method="nearest")
            u_sgt.attrs['units'] = 'm/s'; v_sgt.attrs['units'] = 'm/s'
            ug_sgt.attrs['units'] = 'm/s'; vg_sgt.attrs['units'] = 'm/s'
            w_sgt.attrs['units'] = 'm/s' # will convert to cm/s within function (func_4p4) 

            # arrange in larger array (not a dictionary; what is it?)
            data_gl_sgt=[u_gl, v_gl, w_gl, pv_gl, q_gl, th_gl, ug_gl, vg_gl,
                         u_sgt, v_sgt, w_sgt, ug_sgt, vg_sgt]
           

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
        if inargs.var == 'pv' and inargs.data != 'n768':
            fig = fp.plot_pv(pv, u, v, inargs.data)
        else:
            if inargs.data == 'n768' or inargs.data == 'sgt':
                fig = fp.plot_xy(data_gl_sgt, inargs.data, inargs.var, df, 
                                 wind=inargs.wind, overlay=inargs.cs)
            else:
                fig = fp.plot_xy(data, inargs.data, inargs.var, df)

        if inargs.data == '4p4' or inargs.data == 'era5':
            fili = './{2}{1}_{3}_{0}.png'.format(tstr, inargs.plev, inargs.var, inargs.data)
        elif inargs.data == 'n768':
            fili = './{2}{1}_{3}_{4}_{5}_{0}.png'.format(tstr, inargs.mlev, inargs.var, 
                                                         inargs.data, inargs.wind, inargs.cs)
        else: # SGT 
            fili = './{2}{1}_{3}_{4}_{5}_{0}.png'.format(tstr, inargs.mlev, 
                                                         inargs.var, inargs.data, 
                                                         inargs.sgt, inargs.wind)
        fig.savefig(fili)

        print('output file created. moving onto next one...')

    # vertical cross section (inargs.plot_type == 'xz')
    else:

        #fig = fp.plot_xz(data, inargs.data, inargs.var, inargs.plane)
        if int(inargs.hr) == 24:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 112.0]; end = [12.0, 112.0]
                else:
                    start = [1.0, 112.0]; end = [12.0, 112.06] # 112º
            else:
                start = [4.0, 105.00]; end = [4.04, 118.00] # 3º
        elif int(inargs.hr) == 36:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 110.0]; end = [12.0, 110.0]
                else:
                    start = [1.0, 110.0]; end = [12.0, 110.06] # 110º
            else:
                start = [4.0, 103.00]; end = [4.04, 116.00] # 4º
        elif int(inargs.hr) == 48:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 108.0]; end = [12.0, 108.0]
                else:
                    start = [2.0, 108.0]; end = [12.0, 108.06] # 108º
            else:
                start = [7.0, 102.0]; end = [7.04, 114.0] # 6º
        elif int(inargs.hr) == 60:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 106.0]; end = [12.0, 106.0]
                else:
                    start = [2.0, 106.0]; end = [12.0, 106.06] # 106º
            else:
                start = [7.0, 101.0]; end = [7.04, 114.0] # 7º
        elif int(inargs.hr) == 72:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 102.0]; end = [12.0, 102.0]
                else:
                    start = [2.0, 104.0]; end = [12.0, 104.06] # 102º
            else:
                start = [8.0, 98.0]; end = [8.04, 111.0] # 10º

        if inargs.plane == 'ns':
            var_dim = 'longitude'
            ns = str(start[1]); ts = str(start[0]); tf = str(end[0])
            pstr = ns+'_'+ts+'_'+tf
            # avoid blank edges on x-z plot (troubleshooting)
            start[0] = start[0] - 0.5; end[0] = end[0] + 0.5
        else:
            var_dim = 'latitude'
            ts = str(start[0]); ns = str(start[1]); nf = str(end[1])
            pstr = ts+'_'+ns+'_'+nf
            # avoid blank edges on x-z plot (troubleshooting)
            start[1] = start[1]- 0.5; end[1] = end[1] + 0.5

        # remove unused dimensions 
        if inargs.data == '4p4':
            data = xr.combine_by_coords([data_pc.squeeze(['t', 't_1']),
                                         data_pd.squeeze(['t', 't_1'])])
            data = data.reindex(p=data.p[::-1])
            u = data["u"]; v = data["v"]; z = data["ht"]; temp = data["temp"]
            q = data["q"] * 1000.; q.attrs['units'] = 'g kg-1'; omg = data["omega"]

            # interpolate variables onto same grid 
            temp = temp.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")
            z = z.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")

            # calculate potential temperature
            th = mpcalc.potential_temperature(temp.p, temp)
            th_cs = th.sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           p=slice(1000.0, 150.0) ).squeeze(var_dim)

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

        else: # N768 or SGT tool

            ### N768 MetUM ###

            # remove unnecessary (time) dimension
            w_gl = w_gl.squeeze('t'); u_gl = u_gl.squeeze('t'); v_gl = v_gl.squeeze('t')
            pv_gl = pv_gl.squeeze('t') * 1000000.; q_gl = q_gl.squeeze('t'); th_gl = th_gl.squeeze('t')

            # interpolate geostrophic wind components (from SGT tool) onto N768 grid 
            ug_gl = ug_um.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            vg_gl = vg_um.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")

            # geostrophic and ageostrophic wind components
            ua_gl = u_gl - ug_gl; va_gl = v_gl - vg_gl

            # set up theta to overlay in all cases 
            if inargs.plane == 'ns':
                th = th_gl.sel(longitude=start[1], method="nearest")
                th_cs = th.sel(latitude=slice(start[0],end[0]),
                               height_levels=slice(50, 13500) )
            else:
                th = th_gl.sel(latitude=start[0], method="nearest")
                th_cs = th.sel(longitude=slice(start[1],end[1]),
                               height_levels=slice(50, 13500) )

            ### SGT tool ### 

            w_sg = w_sgt * 100.; u_sg = u_sgt; v_sg = v_sgt; ug_sg = ug_sgt; vg_sg = vg_sgt

            # interpolate to N768 grid 
            u_sg = u_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            v_sg = v_sg.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")
            w_sg = w_sg.interp(longitude=w_gl["longitude"],latitude=w_gl["latitude"],method="linear")
            ug_sg = ug_sg.interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            vg_sg = vg_sg.interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")

            # calculate ageostrophic wind components (SGT tool)
            ua_sg = u_sg - ug_sg; va_sg = v_sg - vg_sg 

            # relative vorticity
            vort_sg = mpcalc.vorticity(u_sg, v_sg, dx=None, dy=None) * 100000
            vort_sg.attrs['units'] = '10-5 s-1'            

            # set up vertical velocity to overlay if needed 
            if inargs.plane == 'ns':
                w = w_sg.sel(longitude=start[1], method="nearest")
                w_cs = w.sel(latitude=slice(start[0],end[0]),
                             height_levels=slice(50, 13500) )
                v = v_sg.sel(longitude=start[1], method="nearest")
                v_cs = v.sel(latitude=slice(start[0],end[0]),
                             height_levels=slice(50, 13500) )
            else:
                w = w_sg.sel(latitude=start[0], method="nearest")
                w_cs = w.sel(longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) )
                v = v_sg.sel(latitude=start[0], method="nearest")
                v_cs = v.sel(longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 13500) )

        # temporary method for choosing variable to plot 
        if inargs.var == 'w':
            if inargs.data == '4p4':
                arr = omg.sel(latitude=slice(start[0],end[0]), 
                              longitude=slice(start[1],end[1]),
                              p=slice(1000.0, 150.0) ).squeeze(var_dim)
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
                if inargs.plane == 'ns':
                    arr = w.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) ) * 100.
                else:
                    arr = w.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) ) * 100.
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
                               p=slice(1000.0, 150.0) ).squeeze(var_dim)            
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
                            p=slice(1000.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = q.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)                
            elif inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = q_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = q_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            else:
                if inargs.plane == 'ns':
                    arr = q_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = q_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            dl = 1.0; qmin = 1.0; qmax = 18.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
            cb_label = 'Specific humidity (kg kg-1)'

        # add relative humidity for N768 (08/07/21)
        elif inargs.var == 'rh':
            rh = mpcalc.relative_humidity_from_specific_humidity(q.p, temp, q) * 100.
            if inargs.data == '4p4':
                arr = rh.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             p=slice(1000.0, 150.0) ).squeeze(var_dim)
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
                            p=slice(1000.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = u.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = u_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = u_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) )
            else: # N768 
                if inargs.plane == 'ns':
                    arr = u_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = u_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) )
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'Zonal wind (m s-1)'
            

        elif inargs.var == 'v':
            if inargs.data == '4p4':
                arr = v.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            p=slice(1000.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                arr = v.sel(latitude=slice(start[0],end[0]),
                            longitude=slice(start[1],end[1]),
                            isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = v_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = v_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            else: # N768
                if inargs.plane == 'ns':
                    arr = v_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = v_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            dl = 1.0; vmin = -25.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'Meridional wind (m s-1)'


        elif inargs.var == 'ua':
            if inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = ua_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = ua_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
                dl = 1.0; vmin = -15.0; vmax = -vmin + dl
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
                cb_label = 'u_ageo (m s-1)'
            elif inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = ua_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else: 
                    arr = ua_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
                dl = 1.0; vmin = -30.0; vmax = -vmin + dl
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
                cb_label = 'u_ageo (m s-1)'            


        elif inargs.var == 'va':
            if inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = va_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = va_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
                dl = 1.0; vmin = -15.0; vmax = -vmin + dl
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
                cb_label = 'u_ageo (m s-1)'
            elif inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = va_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = va_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
                dl = 1.0; vmin = -30.0; vmax = -vmin + dl
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
                cb_label = 'u_ageo (m s-1)'


        elif inargs.var == 'ug':
            if inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = vg_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else: 
                    arr = vg_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            elif inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = ug_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = ug_gl.sel(latitude=start[0], method="nearest")                    
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'u_geo (m s-1)'


        elif inargs.var == 'vg':
            if inargs.data == 'sgt':
                if inargs.plane == 'ns':
                    arr = vg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = vg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            elif inargs.data =='n768':
                if inargs.plane == 'ns':
                    arr = vg_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = vg_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            dl = 1.0; vmin = -15.0; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl)
            cb_label = 'v_geo (m s-1)'


        elif inargs.var == 'pv':
            if inargs.data == '4p4':
                pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
                arr = pv.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             p=slice(1000.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'era5':
                pv = mpcalc.potential_vorticity_baroclinic(th, th.p, u, v) * 1000000.
                arr = pv.sel(latitude=slice(start[0],end[0]),
                             longitude=slice(start[1],end[1]),
                             isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
            else:
                if inargs.plane == 'ns':
                    arr = pv_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = pv_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
            Levels = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 
                      1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
            Cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("magma_r").colors[:250])
            Cmap.set_under('white')
            cb_label = 'Potential vorticity (PVU)'


        # interpolate to new levels straight before plotting (troubleshooting)
        ht_coords = np.arange(0, 13500, 250)
        prs_coords = np.arange(1000, 100, -50)

        if inargs.data != '4p4':
            arr = arr.interp(height_levels=ht_coords,method="linear")
            th_cs = th_cs.interp(height_levels=ht_coords,method="linear")
            w_cs = w_cs.interp(height_levels=ht_coords,method="linear")
            v_cs = v_cs.interp(height_levels=ht_coords,method="linear")
        else:
            arr = arr.interp(p=prs_coords,method="linear")
            th_cs = th_cs.interp(p=prs_coords,method="linear")

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
        else: # SGT tool
            dstr = u_gl.t.dt.strftime("%Y%m%dT%H").values[0]

        # set up plot 
        fig = plt.figure(figsize=[9,6])
        ax = plt.axes()
        if inargs.data == 'sgt':
            fili='./vert_cs_{3}_{4}_{2}_{1}_{5}_{0}.png'.format(dstr, inargs.var,
                                                                pstr, inargs.data,
                                                                inargs.sgt, inargs.cs)
        else:
            fili='./vert_cs_{3}_{2}_{1}_{4}_{0}.png'.format(dstr, inargs.var, pstr, 
                                                            inargs.data, inargs.cs)

        # filled contour plot (x-z)
        if inargs.data == 'n768' or inargs.data == 'sgt':
            if inargs.plane == 'ns':
                lat_dim = np.arange(np.rint(th_cs.latitude[0].data), 
                                    np.rint(th_cs.latitude[-1].data)+0.25, 0.25)
                arr = arr.interp(latitude=lat_dim, method="linear")            
                th_cs = th_cs.interp(latitude=lat_dim, method="linear")
                if inargs.data == 'n768' or inargs.data == 'sgt':
                    w_cs = w_cs.interp(latitude=lat_dim, method="linear")
                    v_cs = v_cs.interp(latitude=lat_dim, method="linear")
            else:
                lon_dim = np.arange(np.rint(th_cs.longitude[0].data),
                                    np.rint(th_cs.longitude[-1].data)+0.25, 0.25)
                arr = arr.interp(longitude=lon_dim, method="linear")
                th_cs = th_cs.interp(longitude=lon_dim, method="linear")
                if inargs.data == 'n768' or inargs.data == 'sgt':
                    w_cs = w_cs.interp(longitude=lon_dim, method="linear")
                    v_cs = v_cs.interp(longitude=lon_dim, method="linear")
                        
        var_contour = plt.contourf(arr, levels=Levels, extend='max', cmap=Cmap)
        if inargs.cs == 'th':
            th_contour = plt.contour(th_cs, levels=np.arange(200, 500, 2), colors=['black'])
        elif inargs.cs == 'w':
            w_contour = plt.contour(w_cs, levels=np.arange(1, 10, 1), colors=['deepskyblue'])
        elif inargs.cs == 'v':
            v_contour = plt.contour(v_cs, levels=np.arange(4, 20, 2), colors=['deepskyblue'])
        else: # 'vw'
            w_contour = plt.contour(w_cs, levels=np.arange(1, 10, 1), colors=['deepskyblue'])
            v_contour = plt.contour(v_cs, levels=np.arange(5, 10, 5), 
                                    colors=['slategray'], linestyles='dashed')
        var_cbar = fig.colorbar(var_contour)

        # tickmarks and labels
        ax.grid(True)
        xint = 1; yint = 2
        if inargs.plane == 'ns':
            ts = np.rint(arr.latitude[0].data); tf = np.rint(arr.latitude[-1].data)
            dim_size = len(arr.latitude); ax.set_xlabel('Latitude (degrees north)')
        else: 
            ts = np.rint(arr.longitude[0].data); tf = np.rint(arr.longitude[-1].data)
            dim_size = len(arr.longitude); ax.set_xlabel('Longitude (degrees east)')

        # x-axis tickmarks every 2º 
        ax.set_xlim(0, dim_size-1)
        if inargs.data == '4p4': # or inargs.data == 'n768' or inargs.data == 'sgt':
            ax.set_xticks(np.arange(0, dim_size+1, 50) )
        else: # ERA5, N768, SGT --> elif inargs.data == 'era5':
            ax.set_xticks(np.arange(0, dim_size+1, 8) )
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
    parser.add_argument("--cs", type=str, default="th", help="Variable to overlay onto cross-sec (th,w,v)")
    args = parser.parse_args()

    main(args)
