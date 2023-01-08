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
matplotlib.use('Agg')

import func_4p4 as fp

def main(inargs):
    """
    Produce plots of atmospheric variables using ERA5 reanalysis, MetUM forecast or satellite data
    """

    # FUNCTION 1 --> read in BV track data and extract track information
    VORTEX_PATH = '/nobackup/earshar/borneo/bv_2018102112_track.csv'
    bv_lat, bv_lon, bv_time = extract_vortex_info(VORTEX_PATH)


    # FUNCTION 2 --> subset the data
    bounds = define_plot_bounds(inargs.plot_type)


    # create date string using existing function --> need a less messy way of doing this
    METUM_4p4_PATH = '/nobackup/earshar/borneo/20181021T1200Z_SEA4_km4p4_ra1tld'
    _, date_str, _, _, _ = fp.open_file(METUM_4p4_PATH, inargs.hr, file_type='4p4')

    # FUNCTION 3 --> plot Himawari brightness temperature data
    if inargs.var == 'hima':
        single_date_str = date_str.strftime("%Y%m%d_%H00")
        himawari_path = '/nobackup/earshar/borneo/himawari/himawari_10.4_{0}.nc'.format(single_date_str)
        himawari_plot = plot_t_bright_himawari(single_date_str, bounds, himawari_path, VORTEX_PATH)
    # FUNCTION 4 --> produce time-series plot of accumulated precipitation
    elif inargs.var == 'prcp':
        prcp_time_series_plot = plot_prcp_time_series(bounds, VORTEX_PATH, inargs.r0)
    # FUNCTION 5 --> calculate and plot vbar, ubar or circ
    elif inargs.var == 'ubar':
        mean_uwind_plot = plot_mean_uwind(bounds, inargs.r0)
    elif inargs.var == 'node':
        mean_vwind_plot = plot_mean_vwind(bounds, inargs.r0, date_str)
    elif inargs.var == 'circ':
        circ_time_series_plot = plot_circ_time_series(bounds, inargs.hr, VORTEX_PATH)
    elif inargs.var == 'heating':
        total_heating_rate = read_n768_metum_heating(bounds, inargs.hr)
    # FUNCTION 6 --> general reading in of data
    else:
        if inargs.data == '4p4' or inargs.data == 'era5':
            data_4p4 = read_data(inargs.data, inargs.hr, inargs.sgt, bounds)
        elif inargs.data == 'era5':
            data_era5 = read_data(inargs.data, inargs.hr, inargs.sgt, bounds)
        else: # inargs.data == 'n768' or inargs.data == 'sgt'
            data_n768, data_sgt = read_data(inargs.data, inargs.hr, inargs.sgt, bounds)

    # FUNCTION 7 --> interpolate N768 MetUM and SGT tool data onto specified levels
    if inargs.data == 'n768' or inargs.data == 'sgt':
        data_n768, data_sgt = interp_to_evenly_spaced_levels(data_n768, data_sgt)

    # ERA5 onto 4p4 (not for now)
    if inargs.data == 'era5':
        u_era = era5.u; v_era = era5.v; w_era = era5.w
        q_era = era5.q * 1000.; q_era.attrs['units'] = 'g kg-1'
        z_era = era5.z; pv_era = era5.pv; temp_era = era5.t


    # SECTION (multiple functions) --> x-y plots

    if inargs.plot_type == 'xy': # horizontal cross-section ('xy')

        # FUNCTION 10 --> set up 4p4 data for x-y plot

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
            elif inargs.var == 'n2':
                # read in data, combining two streams 
                data = xr.combine_by_coords([data_pc.squeeze(['t', 't_1']),
                                             data_pd.squeeze(['t', 't_1'])])
                data = data.reindex(p=data.p[::-1])
                u = data["u"]; v = data["v"]; temp = data["temp"]
                u.attrs['units'] = 'm/s'; v.attrs['units'] = 'm/s'
                q = data["q"] #q.attrs['units'] = 'g kg-1'
                # interpolate onto same grid 
                temp=temp.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],
                                 method="linear")
                # calculate mixing ratio 
                mix = mpcalc.mixing_ratio_from_specific_humidity(q)
                # calculate potential temperature 
                th = mpcalc.potential_temperature(temp['p'], temp); th.attrs['units'] = 'K'
                # calculate density 
                rho = mpcalc.density(th.p, temp, mix)
                # next, calculate d(theta)/dp
                th_dp = mpcalc.first_derivative(th, axis=0) / 100.; th_dp.attrs['units'] = 'K/Pa'
                # then, calculate N^2 in pressure coordinates
                n2 = th_dp * -( (rho * np.square(9.81) ) / th)
                # finally, calculate effective PV gradient (q / N^2) 
                pv_grad = (q / n2)
                # select pressure level to analyse
                pv_grad = pv_grad.sel(p=int(inargs.plev)); u = u.sel(p=int(inargs.plev))
                v = v.sel(p=int(inargs.plev))

            else:
                data = xr.combine_by_coords([data_pc.squeeze(['t', 't_1']),
                                             data_pd.squeeze(['t', 't_1'])])
                data = data.sel(p=int(inargs.plev) )

        # FUNCTION 11 --> set up ERA5 data for x-y plot

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

        # FUNCTION 12 --> set up N768/SGT tool data for x-y plot 
 
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

        # FUNCTION 13 --> produce x-y plot, and save 

        # call the plotting function, and save the file 
        if inargs.var == 'pv' and inargs.data != 'n768':
            fig = fp.plot_pv(pv, u, v, df, inargs.data)
        elif inargs.var == 'n2' and inargs.data != 'n768':
            fig = fp.plot_n2(pv_grad, u, v, df, inargs.data)
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

    # SECTION (multiple functions) --> xz plots (inargs.plot_type == 'xz')

    else:

        """
        Tidy this part of the script up, and incorporate into external function 
        #fig = fp.plot_xz(data, inargs.data, inargs.var, inargs.plane)
        """

        # FUNCTION 14 --> set xz plot limits 

        if int(inargs.hr) == 0:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 112.0]; end = [12.0, 112.0]
                else:
                    start = [1.0, 112.0]; end = [12.0, 112.06] # 112º
            else: # E-W (lon-z)
                if inargs.data == '4p4':
                    start = [6.0, 89.00]; end = [6.0, 129.00] # 6º
                else:
                    start = [6.0, 89.00]; end = [6.04, 129.00] # 6º

        elif int(inargs.hr) == 12:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 112.0]; end = [12.0, 112.0]
                else:
                    start = [1.0, 112.0]; end = [12.0, 112.06] # 112º
            else: # E-W (lon-z)
                if inargs.data == '4p4':
                    start = [5.5, 89.00]; end = [5.5, 129.00] # 5.5º  
                else:
                    start = [5.5, 89.00]; end = [5.54, 129.00] # 5.5º 

        elif int(inargs.hr) == 24:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 112.0]; end = [12.0, 112.0]
                else:
                    start = [1.0, 112.0]; end = [12.0, 112.06] # 112º
            else: # E-W (lon-z)
                if inargs.data == '4p4':
                    start = [5.5, 89.00]; end = [5.50, 129.00] # 3º
                else:
                    start = [5.5, 89.00]; end = [5.54, 129.00] # 3º

        elif int(inargs.hr) == 36:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [1.0, 110.0]; end = [12.0, 110.0]
                else:
                    start = [1.0, 110.0]; end = [12.0, 110.06] # 110º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [5.5, 89.00]; end = [5.5, 129.00] # 4º
                else:
                    start = [5.5, 89.00]; end = [5.54, 129.00] # 4º 

        elif int(inargs.hr) == 48:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 108.0]; end = [12.0, 108.0]
                else:
                    start = [2.0, 108.0]; end = [12.0, 108.06] # 108º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    # when running with end = [6.0, ...], got the error below (different from T+24)
                    # IndexError: index 0 is out of bounds for axis 1 with size 0
                    # same with T+72 below --> probably a simple explanation, but currently not sure 
                    start = [6.0, 89.0]; end = [6.04, 129.0] # 6º
                else:
                    start = [6.0, 89.0]; end = [6.04, 129.0] # 6º 

        elif int(inargs.hr) == 60:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 106.0]; end = [12.0, 106.0]
                else:
                    start = [2.0, 106.0]; end = [12.0, 106.06] # 106º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [6.5, 89.00]; end = [6.5, 129.0] # 7º
                else:
                    start = [6.5, 89.00]; end = [6.54, 129.0] # 7º

        elif int(inargs.hr) == 72:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 102.0]; end = [12.0, 102.0]
                else:
                    start = [2.0, 104.0]; end = [12.0, 104.06] # 102º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [7.0, 89.00]; end = [7.04, 129.0] # 10º
                else:
                    start = [7.0, 89.00]; end = [7.04, 129.0] # 10º

        elif int(inargs.hr) == 84:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 102.0]; end = [12.0, 102.0]
                else:
                    start = [2.0, 104.0]; end = [12.0, 104.06] # 102º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [6.5, 89.00]; end = [6.5, 129.0] # 10º
                else:
                    start = [6.5, 89.00]; end = [6.54, 129.0] # 10º

        elif int(inargs.hr) == 96:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 102.0]; end = [12.0, 102.0]
                else:
                    start = [2.0, 104.0]; end = [12.0, 104.06] # 102º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [6.0, 89.00]; end = [6.0, 129.0] # 10º
                else:
                    start = [6.0, 89.00]; end = [6.04, 129.0] # 10º

        elif int(inargs.hr) == 108:
            if inargs.plane == 'ns':
                if inargs.data == 'era5':
                    start = [2.0, 102.0]; end = [12.0, 102.0]
                else:
                    start = [2.0, 104.0]; end = [12.0, 104.06] # 102º
            else: # E-W (lon-z) 
                if inargs.data == '4p4':
                    start = [6.0, 89.0]; end = [6.0, 129.0] # 10º
                else:
                    start = [6.0, 89.0]; end = [6.04, 129.0] # 10º

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

        # FUNCTION 15 --> set up 4p4 data for xz plot 

        # remove unused dimensions --> script won't run if you choose --cs='w' and --data='4p4'
        if inargs.data == '4p4':
            data = data_4p4
            data = data.reindex(p=data.p[::-1])
            u = data["u"]; v = data["v"]; z = data["ht"]; temp = data["temp"]
            q = data["q"] # * 1000.; q.attrs['units'] = 'g kg-1'
            omg = data["omega"]; rh = data["rh"]

            # interpolate variables onto same grid 
            temp = temp.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")
            z = z.interp(longitude_1=u["longitude"],latitude_1=u["latitude"],method="linear")

            # calculate potential temperature
            th = mpcalc.potential_temperature(temp.p, temp)
            th_cs = th.sel(latitude=slice(start[0],end[0]),
                           longitude=slice(start[1],end[1]),
                           p=slice(1000.0, 150.0) ).squeeze(var_dim)

            """
            calculate effective PV gradient (q / N^2)
            """

            # first, calculate mixing ratio 
            mix = mpcalc.mixing_ratio_from_specific_humidity(q)
            # then, calculate density (using mixing ratio)
            rho = mpcalc.density(th.p, temp, mix)

            # next, calculate d(theta)/dp
            th_dp = mpcalc.first_derivative(th, axis=0) / 100.; th_dp.attrs['units'] = 'K/Pa'
            # finally, calculate N^2 in pressure coordinates
            n2 = th_dp * -( (rho * np.square(9.81) ) / th)

            # now calculate effective PV gradient (q / N^2)
            pv_grad = (q / n2)

        # FUNCTION 16 --> set up ERA5 data for xz plot 

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

        # FUNCTION 17 --> set up N768 and SGT tool data for xz plot 

        else: # N768 or SGT tool

            ### N768 MetUM ###

            # first file contains 2 times --> select single time 
            if inargs.hr == 0 or inargs.hr == 12:
                u_gl = data_n768.u.sel(t=date_str)
                v_gl = data_n768.v.sel(t=date_str)
                w_gl = data_n768.dz_dt.sel(t=date_str)
                pv_gl = data_n768.field83.sel(t=date_str) * 1000000.
                q_gl = data_n768.q.sel(t=date_str) / 1000.; q_gl.attrs['units'] = 'kg kg-1'
                th_gl = data_n768.theta.sel(t=date_str)

            else:
                w_gl = data_n768.dz_dt; u_gl = data_n768.u; v_gl = data_n768.v
                pv_gl = data_n768.field83 * 1000000.
                q_gl = data_n768.q / 1000.; q_gl.attrs['units'] = 'kg kg-1'
                th_gl = data_n768.theta

            # interpolate geostrophic wind components (from SGT tool) onto N768 grid 
            ug_gl = data_sgt['ug_um'].interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            vg_gl = data_sgt['vg_um'].interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")

            # geostrophic and ageostrophic wind components
            ua_gl = u_gl - ug_gl; va_gl = v_gl - vg_gl

            # set up theta to overlay in all cases 
            if inargs.plane == 'ns':
                th = th_gl.sel(longitude=start[1], method="nearest")
                th_cs = th.sel(latitude=slice(start[0],end[0]),
                               height_levels=slice(50, 15000) )
            else:
                th = th_gl.sel(latitude=start[0], method="nearest")
                th_cs = th.sel(longitude=slice(start[1],end[1]),
                               height_levels=slice(50, 15000) )

            # calculate static stability
            # LOOK INTO THIS PART OF THE CODE (dimensions are wrong for the line below)
            th_dz = mpcalc.first_derivative(th_gl, axis=0)#; th_dz.attrs['units'] = 'K/km'
            th_dz = th_dz * (9.81 / th_gl)

            # now calculate effective PV gradient (q / N^2)
            pv_grad = (q_gl / th_dz) #/ 10000.

            # new moist stability calculation (January 2023)
            moist_stability = q_gl * th_dz * 1000000000.

            # interpolate to N768 grid 
            u_sg = data_sgt['u'].interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            v_sg = data_sgt['v'].interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")
            w_sg = data_sgt['w'].interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear") * 100.
            ug_sg = data_sgt['ug'].interp(longitude=u_gl["longitude"],latitude=u_gl["latitude"],method="linear")
            vg_sg = data_sgt['vg'].interp(longitude=v_gl["longitude"],latitude=v_gl["latitude"],method="linear")

            # calculate ageostrophic wind components (SGT tool)
            ua_sg = u_sg - ug_sg; va_sg = v_sg - vg_sg 

            # relative vorticity
            vort_sg = mpcalc.vorticity(u_sg, v_sg, dx=None, dy=None) * 100000
            vort_sg.attrs['units'] = '10-5 s-1'            


            # set up vertical velocity to overlay if needed 
            if inargs.plane == 'ns':
                w = w_sg.sel(longitude=start[1], method="nearest")
                w_cs = w.sel(latitude=slice(start[0],end[0]),
                             height_levels=slice(50, 15000) )
                v = v_sg.sel(longitude=start[1], method="nearest")
                v_cs = v.sel(latitude=slice(start[0],end[0]),
                             height_levels=slice(50, 15000) )
            else:
                w = w_sg.sel(latitude=start[0], method="nearest")
                w_cs = w.sel(longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 15000) )
                v = v_sg.sel(latitude=start[0], method="nearest")
                v_cs = v.sel(longitude=slice(start[1],end[1]),
                             height_levels=slice(50, 15000) )

        # FUNCTION 18 --> produce xz plot using chosen dataset and variable 

        # temporary method for choosing variable to plot 
        if inargs.var == 'w':
            if inargs.data == '4p4': # 4p4 MetUM
                arr = omg.sel(latitude=slice(start[0],end[0]), 
                              longitude=slice(start[1],end[1]),
                              p=slice(1000.0, 150.0) ).squeeze(var_dim)
                dl = 0.5; lmin = -15; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (Pa s-1)'
            elif inargs.data == 'era5': # ERA5
                arr = omg.sel(latitude=slice(start[0],end[0]),
                              longitude=slice(start[1],end[1]),
                              isobaricInhPa=slice(950.0, 150.0) ).squeeze(var_dim)
                dl = 0.5; lmin = -15; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (Pa s-1)'
            elif inargs.data == 'n768': # N768 MetUM
                if inargs.plane == 'ns':
                    arr = w_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) ) * 100.
                else:
                    arr = w_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) ) * 100.
                dl = 1.0; lmin = -20; lmax = -lmin + dl
                Levels = np.arange(lmin, lmax, dl); Cmap='seismic'
                cb_label = 'Vertical velocity (cm s-1)'
            else: # SGT tool 
                if inargs.plane == 'ns':
                    arr = w_sg.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = w_sg.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 13500) )
                dl = 1.0; lmin = -20; lmax = -lmin + dl # SCALE DIFFERENT HERE
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
            Cmap,norm,Levels = normalise_cmap(lmin,lmax,0,dl,'bwr')
            cb_label = 'Relative vorticity (10-5 s-1)'

        # potential temperature (troubleshooting)
        elif inargs.var == 'th':
            if inargs.data == '4p4':
                arr = th_dp.sel(latitude=slice(start[0],end[0]),
                                longitude=slice(start[1],end[1]),
                                p=slice(1000.0, 150.0) ).squeeze(var_dim)
            elif inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = th_gl.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 13500) )
                else:
                    arr = th_gl.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                height_levels=slice(50, 13500) )
            #dl = 2.0; qmin = 280.0; qmax = 370.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma'
            dl = 2.0; qmin = 280.0; qmax = 370.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma'
            cb_label = 'Potential temperature (K)'


        # effective PV gradient (r / N^2)
        elif inargs.var == 'n2':
            if inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = pv_grad.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 15000) )
                else:
                    arr = pv_grad.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 15000) )
                    dl = 5.0; qmin = 0.0; qmax = 100.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma_r'
                    cb_label=r'$q/N^2\,(s^{-2})$'
            elif inargs.data == '4p4':
                arr = pv_grad.sel(latitude=slice(start[0],end[0]),
                                  longitude=slice(start[1],end[1]),
                                  p=slice(1000.0, 150.0) ).squeeze(var_dim)
                dl = 5.0; qmin = 0.0; qmax = 100.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma_r'
                cb_label=r'$q/N^2\,(s^{-2})$'


        # moist stability (r * N^2) --> January 2023
        elif inargs.var == 'moist_stability':
            if inargs.data == 'n768':
                if inargs.plane == 'ns':
                    arr = moist_stability.sel(longitude=start[1], method="nearest")
                    arr = arr.sel(latitude=slice(start[0],end[0]),
                                  height_levels=slice(50, 15000) )
                else:
                    arr = moist_stability.sel(latitude=start[0], method="nearest")
                    arr = arr.sel(longitude=slice(start[1],end[1]),
                                  height_levels=slice(50, 15000) )
                    dl = 0.5; qmin = 0.0; qmax = 5.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma_r'
                    cb_label=r'$rN^2\,(s^{-2})$'
            elif inargs.data == '4p4':
                arr = moist_stability.sel(latitude=slice(start[0],end[0]),
                                  longitude=slice(start[1],end[1]),
                                  p=slice(1000.0, 150.0) ).squeeze(var_dim)
                dl = 0.5; qmin = 0.0; qmax = 5.0; Levels = np.arange(qmin,qmax+dl,dl); Cmap='plasma_r'
                cb_label=r'$rN^2\,(s^{-2})$'


        # specific humidity 
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
            dl = 5.0; rmin = 30.0; rmax = 100.0; Levels=np.arange(rmin,rmax+dl,dl); Cmap='plasma_r'
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
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
            cb_label=r'Meridional wind $\mathregular{(m\,s^{-1})}$'


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
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
                Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'bwr')
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

            dl = 0.2; vmin = -2.4; vmax = -vmin + dl
            Cmap,norm,Levels=normalise_cmap(vmin,vmax,0,dl,'PuOr_r')
            cb_label = 'Potential vorticity (PVU)'

            #Levels = (-2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1,
            #          0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
            #Cmap = 'PuOr'
            #cb_label = 'Potential vorticity (PVU)' 
            """
            Levels = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 
                      1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
            Cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("magma_r").colors[:250])
            Cmap.set_under('white')
            cb_label = 'Potential vorticity (PVU)'
            """

        # FUNCTION 19 --> interpolate to new levels before plotting 

        # interpolate to new levels straight before plotting (troubleshooting)
        ht_coords = np.arange(0, 15000, 250)
        prs_coords = np.arange(1000, 100, -50)

        if inargs.data != '4p4':
            arr = arr.interp(height_levels=ht_coords,method="linear")
            th_cs = th_cs.interp(height_levels=ht_coords,method="linear")
            w_cs = w_cs.interp(height_levels=ht_coords,method="linear")
            v_cs = v_cs.interp(height_levels=ht_coords,method="linear")
        else:
            arr = arr.interp(p=prs_coords,method="linear")
            th_cs = th_cs.interp(p=prs_coords,method="linear")

        # FUNCTION 20 --> create date string for plot title 

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
            dstr = u_gl.t.dt.strftime("%Y%m%dT%H").values#[0]

        # Plotting SECTION (multiple smaller functions)

        # FUNCTION 21 --> initialise plot object and produce filled contours 

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

        # plot filled contours
        var_contour = plt.contourf(arr, levels=Levels, extend='max', cmap=Cmap)

        # FUNCTION 22 --> overlay line contours onto filled contours 

        # overlay line contours 
        if inargs.cs == 'th':
            if inargs.data == '4p4':
                th_contour = plt.contour(th_cs, levels=np.arange(200, 500, 4), colors=['black'])
            else:
                th_contour = plt.contour(th_cs, levels=np.arange(200, 500, 2), colors=['black'])
        elif inargs.cs == 'w':
            w_contour = plt.contour(w_cs, levels=np.arange(-10, 10, 4), colors=['deepskyblue'])
        elif inargs.cs == 'v':
            v_contour = plt.contour(v_cs, levels=np.arange(4, 20, 2), colors=['deepskyblue'])
        elif inargs.cs == 'vw':
            w_contour = plt.contour(w_cs, levels=np.arange(1, 10, 1), colors=['deepskyblue'])
            v_contour = plt.contour(v_cs, levels=np.arange(5, 10, 5), 
                                    colors=['slategray'], linestyles='dashed')
        else:
            print('no additional contours overlaid on x-z plot')
        var_cbar = fig.colorbar(var_contour)

        # FUNCTION 23 --> overlay axis tickmarks and labels 

        # tickmarks and labels
        ax.grid(True)
        xint = 1; yint = 4
        if inargs.plane == 'ns':
            ts = np.rint(arr.latitude[0].data); tf = np.rint(arr.latitude[-1].data)
            dim_size = len(arr.latitude); ax.set_xlabel(r'Latitude ($\degree$N)', fontsize='large')
        else: 
            ts = np.rint(arr.longitude[0].data); tf = np.rint(arr.longitude[-1].data)
            dim_size = len(arr.longitude); ax.set_xlabel(r'Longitude ($\degree$E)', fontsize='large')

        # x-axis tickmarks every 2º 
        ax.set_xlim(0, dim_size-1)
        if inargs.data == '4p4': # or inargs.data == 'n768' or inargs.data == 'sgt':
            ax.set_xticks(np.arange(0, dim_size+1, 100) )
        else: # ERA5, N768, SGT --> elif inargs.data == 'era5':
            ax.set_xticks(np.arange(0, dim_size+1, 16) )
        ax.set_xticklabels(np.arange(int(ts), int(tf+1), yint) , fontsize='large')

        # y-axis tickmarks and labels 
        if inargs.data == '4p4':
            plev_size = len(arr.p); ps = np.rint(arr.p[0].data); pf = np.rint(arr.p[-1].data)
            ax.set_yticks(np.arange(0, plev_size, 1) )
            ax.set_yticklabels(arr.p.data); ax.set_ylabel('Pressure (hPa)', fontsize='large')
        elif inargs.data == 'era5':
            plev_size = len(arr.isobaricInhPa)
            ps = np.rint(arr.isobaricInhPa[0].data); pf = np.rint(arr.isobaricInhPa[-1].data)
            ax.set_yticks(np.arange(0, plev_size, 1) )
            ax.set_yticklabels(arr.isobaricInhPa.data); ax.set_ylabel('Pressure (hPa)', fontsize='large')
        else:
            mlev_size = len(arr.height_levels)
            ax.set_yticks(np.arange(0, mlev_size, 4) )
            ax.set_yticklabels(arr.height_levels[::4].data); ax.set_ylabel('Height (m)', fontsize='large')

        var_cbar.set_label(cb_label, fontsize='large')
        plt.savefig(fili,dpi=200)

def extract_vortex_info(VORTEX_PATH):

    # read in Kevin Hodges' Borneo vortex track data from text file
    vortex_df = pd.read_csv(VORTEX_PATH, na_filter=True, na_values="1.000000e+25")

    # convert time integers to datetime objects
    vortex_df['Time'] = pd.to_datetime(vortex_df['Time'], format='%Y%m%d%H')

    # extract track information between 12Z on 21st and 26th October
    vortex_lat = vortex_df.loc[0:20, "lat_vort"];
    vortex_lon = vortex_df.loc[0:20, "lon_vort"]
    vortex_time = vortex_df.loc[0:20, "Time"]

    return vortex_lat, vortex_lon, vortex_time


def define_plot_bounds(plot_type):

    if plot_type == 'xz':
        bounds = [87.0, 135.0, -3.0, 20.0]
    else:
        bounds = [88.0, 130.0, -6.0, 23.0]

    return bounds


def plot_t_bright_himawari(single_date_str, bounds, himawari_path, vortex_path):

    OUT_PATH = './himawari_t_bright_{0}.png'.format(single_date_str)
    himawari_data = xr.open_dataset(himawari_path).metpy.parse_cf()
    # himawari_data = himawari_data.sel(longitude=slice(bounds[0], bounds[1]),
    #                                   latitude=slice(bounds[2], bounds[3]))
    himawari_data = himawari_data.sel(lon=slice(bounds[0], bounds[1]),
                                      lat=slice(bounds[2], bounds[3]))
    t_bright = himawari_data.T_b

    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='blue', linewidth=1.0)
    plt.gca().gridlines(color='grey', linestyle='--', linewidth=0.5)

    # ax = overlay_lat_lon_labels(t_bright, ax)

    t_bright.plot.contourf(ax=ax, extend='max', transform=ccrs.PlateCarree(),
                           cbar_kwargs={'label': 'K'},
                           cmap='gist_yarg')

    # customise plot      
    ax.set_xticks([94, 99, 104, 109, 114, 119])
    ax.set_yticks([-2, 1, 4, 7, 10, 13, 16, 19])
    ax.set_xticklabels([94, 99, 104, 109, 114, 119], fontsize='large')
    ax.set_yticklabels([-2, 1, 4, 7, 10, 13, 16, 19], fontsize='large')
    ax.set_xlabel(r'longitude ($\degree$E)', fontsize='large')
    ax.set_ylabel(r'latitude ($\degree$N)', fontsize='large')
    ax.set_title('')

    # fig = overlay_vortex_position(ax, vortex_path, t_bright, fig)
    plt.savefig(OUT_PATH, dpi=200)

    return fig


def overlay_lat_lon_labels(variable_arr, ax):

    lon0, lon1 = np.rint([variable_arr.lon[0].data, variable_arr.lon[-1].data])
    lat0, lat1 = np.rint([variable_arr.lat[0].data, variable_arr.lat[-1].data])
    xint = 5; yint = 3

    ax.set_xticks(np.arange(lon0, lon1 + 1, xint))
    ax.set_xticklabels(np.arange(lon0, lon1 + 1, xint), fontsize='large')
    ax.set_yticks(np.arange(lat0, lat1 + 1, yint))
    ax.set_yticklabels(np.arange(lat0, lat1 + 1, yint), fontsize='large')
    ax.set_xlabel(r'longitude ($\degree$E)', fontsize='large')
    ax.set_ylabel(r'latitude ($\degree$N)', fontsize='large')

    return ax


def overlay_vortex_position(ax, vortex_path, variable_arr, fig):

    bv_lat, bv_lon, bv_time = extract_vortex_info(vortex_path)

    vort_box_radius = 3.0
    filter = bv_time == variable_arr.coords['time'].data
    time_match = bv_time.where(filter).notna()
    ind = int(time_match.loc[time_match == True].index.values)

    ax.add_patch(Rectangle((bv_lon[ind] - vort_box_radius, bv_lat[ind] - vort_box_radius),
                           2 * vort_box_radius, 2 * vort_box_radius, linewidth=2,
                           facecolor='none', edgecolor='c'))

    return fig


def plot_prcp_time_series(bounds, vortex_path, vortex_box_radius):

    METUM_4p4_PATH = '/nobackup/earshar/borneo/20181021T1200Z_SEA4_km4p4_ra1tld_pverb.pp'
    pcubes = iris.load(METUM_4p4_PATH)
    prcp_4p4_metum = xr.DataArray.from_iris(pcubes.extract('stratiform_rainfall_flux')[1])
    prcp_4p4_metum = prcp_4p4_metum.sel(longitude=slice(bounds[0], bounds[1]), latitude=slice(bounds[2], bounds[3]))

    GPM_PATH = '/nobackup/earshar/borneo/GPMHH_201810.nc'
    gpm_data = xr.open_dataset(GPM_PATH).sel(lon=slice(bounds[0], bounds[1]), lat=slice(bounds[2], bounds[3]))
    prcp_gpm = gpm_data.precipitationCal

    Tp = [24, 36, 48, 60, 72, 84, 96, 108, 120]
    prcp_n768_metum_dims = [len(Tp), prcp_4p4_metum.latitude.shape[0], prcp_4p4_metum.longitude.shape[0]]
    prcp_gpm_resample_12h = prcp_gpm.resample(time="12H").sum().sel(time=slice('2018-10-22T12',
                                                                               '2018-10-26T12'))

    prcp_n768_metum = xr.DataArray(np.ones(prcp_n768_metum_dims),
                           dims=["time", "latitude", "longitude"],
                           coords={
                               "time": prcp_gpm_resample_12h.time,
                               "latitude": prcp_4p4_metum.latitude,
                               "longitude": prcp_4p4_metum.longitude,
                           },
                           )

    for i, t in enumerate(Tp):
        METUM_N768_PATH = '/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pa{0:03d}.nc'.format(t - 12)
        # METUM_N768_PATH = '/nobackup/earshar/borneo/case_20181021T1200Z_N768_v2/umglaa_pa{0:03d}.nc'.format(t - 12)
        data_n768_metum = xr.open_dataset(METUM_N768_PATH)
        data_n768_metum=data_n768_metum["tot_precip"].squeeze('t_1').squeeze('surface').sel(longitude=slice(bounds[0],
                                                                                                            bounds[1]),
                                                                                            latitude=slice(bounds[2],
                                                                                                           bounds[3]))
        # interpolate N768 MetUM data onto 4p4 MetUM grid
        prcp_n768_metum[i, :, :] = data_n768_metum.interp(longitude=prcp_4p4_metum["longitude"],
                                                          latitude=prcp_4p4_metum["latitude"],
                                                          method="linear")

    # also interpolate GPM data onto 4p4 MetUM grid
    prcp_gpm = prcp_gpm.interp(lon=prcp_4p4_metum["longitude"],
                               lat=prcp_4p4_metum["latitude"],
                               method="linear")

    fig = calc_acc_prcp_time_series(vortex_path, prcp_gpm, prcp_4p4_metum, prcp_n768_metum, vortex_box_radius)

    return fig


def calc_acc_prcp_time_series(vortex_path, prcp_gpm, prcp_4p4_metum, prcp_n768_metum, vortex_box_radius):

    bv_lat, bv_lon, bv_time = extract_vortex_info(vortex_path)

    acc_prcp_gpm = fp.calc_area_ave_acc_prcp(prcp_gpm, 'gpm', bv_lat, bv_lon, r0=vortex_box_radius, plt_prcp=False)
    acc_prcp_4p4 = fp.calc_area_ave_acc_prcp(prcp_4p4_metum,'4p4',bv_lat,bv_lon,r0=vortex_box_radius,plt_prcp=False)
    acc_prcp_n768 = fp.calc_area_ave_acc_prcp(prcp_n768_metum,'n768',bv_lat,bv_lon,r0=vortex_box_radius,plt_prcp=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bv_time, np.zeros(21), color='k', alpha=0.0)
    ax.plot(bv_time[2:21:2], acc_prcp_4p4, color='k', label='4.4 km MetUM')
    ax.plot(bv_time[4:21:2], acc_prcp_n768, color='b', label='Global MetUM')
    ax.plot(bv_time[2:21:2], acc_prcp_gpm, color='r', label='GPM-IMERG')
    ax.set(xlabel='Time', ylabel='Accumulated rainfall (mm)',
           title='Time-series of area-averaged, accumulated rainfall')
    ax.grid(True)
    ax.legend(loc='upper left')

    OUT_PATH = './acc_prcp_oct2018_{0}deg.png'.format(vortex_box_radius)
    fig.savefig(OUT_PATH, dpi=200)

    return fig


def plot_mean_vwind(bounds, vortex_box_radius, date_str):

    data_n768_metum = read_all_n768_metum(bounds)
    uwind = data_n768_metum.u
    vwind = data_n768_metum.v
    n768_ht_coords = vwind['hybrid_ht_1'].data.astype('int32')

    # interpolate to new latitude grid and rename height coordinate
    vwind = vwind.interp(latitude_1=uwind["latitude"],
                         method="linear").assign_coords(height_levels=("hybrid_ht",
                                                                       n768_ht_coords)).swap_dims({"hybrid_ht":
                                                                        "height_levels"})

    mean_vwind = vwind.sel(longitude=slice(bounds[0], bounds[1]),
                           latitude=slice(2.0, 8.0)).mean(dim=['latitude']).sel(height_levels=slice(50, 15000))

    n768_ht_coords = np.arange(0, 15000, 250)
    mean_vwind = mean_vwind.interp(height_levels=n768_ht_coords, method="linear")

    lon_dimension_subset = np.arange(np.rint(mean_vwind.longitude[0].data),
                                     np.rint(mean_vwind.longitude[-1].data) + 0.25, 0.25)
    mean_vwind = mean_vwind.interp(longitude=lon_dimension_subset, method="linear")


    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0,0].plot(mean_vwind.longitude, mean_vwind.sel(height_levels=2000, t=date_str),
                 color='k', label='Meridional wind at 2 km')
    ax[0,1].plot(mean_vwind.longitude, mean_vwind.sel(height_levels=7000, t=date_str),
                 color='b', label='Meridional wind at 7 km')
    ax.grid(True)
    ax.legend(loc='upper left')


    OUT_PATH = './mean_vwind_n768_{0}deg.png'.format(vortex_box_radius)
    fig.savefig(OUT_PATH, dpi=200)

    return fig


def plot_mean_uwind(bounds, vortex_box_radius):

    data_n768_metum = read_all_n768_metum(bounds)
    uwind = data_n768_metum.u
    vwind = data_n768_metum.v
    n768_ht_coords = uwind['hybrid_ht_1'].data.astype('int32')

    # interpolate to new longitude grid and rename height coordinate
    uwind = uwind.interp(hybrid_ht=n768_ht_coords,
                         method="linear").assign_coords(height_levels=("hybrid_ht",
                                                        n768_ht_coords)).swap_dims({"hybrid_ht":
                                                                                    "height_levels"})

    uwind = uwind.interp(longitude_1=vwind["longitude"],
                         method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                                        n768_ht_coords)).swap_dims({"hybrid_ht_1":
                                                                                    "height_levels"})

    mean_uwind = uwind.sel(longitude=slice(95.0, 120.0),
                           latitude=slice(0.0, 15.0)).mean(dim=['longitude', 't']).sel(height_levels=slice(10, 15000))

    # START FROM HERE ('interp' function causing issues, both with height and latitude)
    n768_ht_coords = np.arange(0, 15000, 250)
    mean_uwind = mean_uwind.interp(height_levels=n768_ht_coords, method="linear")

    lat_dimension_subset = np.arange(np.rint(mean_uwind.latitude[0].data),
                                     np.rint(mean_uwind.latitude[-1].data) + 0.25, 0.25)
    mean_uwind = mean_uwind.interp(latitude=lat_dimension_subset, method="linear")

    dl = 1.0; vmin = -20.0; vmax = -vmin + dl
    Colourmap, _, Levels = normalise_cmap(vmin, vmax, 0, dl, 'bwr')
    colourbar_label = r'Mean zonal wind $\mathregular{(m\,s^{-1})}$'

    ht_coords = np.arange(0, 13500, 250)
    mean_uwind = mean_uwind.interp(height_levels=ht_coords,method="linear")

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_uwind_plot = plt.contourf(mean_uwind, levels=Levels, extend='max', cmap=Colourmap)
    mean_uwind_cbar = fig.colorbar(mean_uwind_plot)


    lat0, lat1 = np.rint([mean_uwind.latitude[0].data, mean_uwind.latitude[-1].data])
    mean_uwind_lat_size = len(mean_uwind.latitude)

    ax.set_xlabel(r'Latitude ($\degree$N)', fontsize='large')
    ax.set_xticks(np.arange(0, mean_uwind_lat_size + 1, 4))
    ax.set_xticklabels(np.arange(lat0, lat1 + 1, 1), fontsize='large')

    mean_uwind_hgt_size = len(mean_uwind.height_levels)
    ax.set_yticks(np.arange(0, mean_uwind_hgt_size, 4))
    ax.set_yticklabels(mean_uwind.height_levels[::4].data)
    ax.set_ylabel('Height (m)', fontsize='large')

    mean_uwind_cbar.set_label(colourbar_label)
    ax.grid(True)


    OUT_PATH = './mean_uwind_n768_{0}deg.png'.format(vortex_box_radius)
    fig.savefig(OUT_PATH, dpi=200)

    return fig


def plot_circ_time_series(bounds, output_time, vortex_path):

    ERA5_PATH = '/nobackup/earshar/borneo/bv_oct2018.grib'
    era5 = xr.open_dataset(ERA5_PATH, engine="cfgrib").metpy.parse_cf()
    data_era5 = fp.subset(era5, bounds, var="circ")

    METUM_4p4_PATH = '/nobackup/earshar/borneo/20181021T1200Z_SEA4_km4p4_ra1tld'
    _, date_str, _, data_pc, data_pd = fp.open_file(METUM_4p4_PATH, output_time, file_type='4p4')
    data_4p4_metum = fp.subset(data_pd, bounds, var="circ")

    data_n768_metum = read_all_n768_metum(bounds)
    uwind = data_n768_metum.u
    vwind = data_n768_metum.v

    bv_lat, bv_lon, bv_time = extract_vortex_info(vortex_path)

    circ_era5 = fp.calc_circ(data_era5.u, data_era5.v, bv_lat, bv_lon, plev=800, r0=3.0)
    circ_4p4 = fp.calc_circ(data_4p4_metum.u, data_4p4_metum.v, bv_lat, bv_lon, plev=800, r0=3.0)
    circ_n768 = fp.calc_circ(uwind, vwind, bv_lat, bv_lon, mlev=2000, r0=3.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bv_time, circ_4p4, color='k', label='4.4 km MetUM')
    ax.plot(bv_time[4:21:2], circ_n768, color='b', label='Global MetUM')
    ax.plot(bv_time, circ_era5, color='r', label='ERA5 reanalysis')

    variable_str = 'Area-averaged relative vorticity'
    ax.set(xlabel='Time',
           ylabel=r'{0} ($\mathregular{10}^{-6}$ s$\mathregular{^{-1}}$)'.format(variable_str),
           title=f'{variable_str} following the vortex')
    ax.grid(True)
    ax.legend(loc='upper left')

    OUT_PATH = './circ_oct2018_{0}deg.png'.format(inargs.r0)
    fig.savefig(OUT_PATH, dpi=200)

    return fig


def read_n768_metum_heating(bounds, output_time):

    if output_time == 0:
        time_str = int(output_time)
    else:
        time_str = int(output_time) - 12

    METUM_N768_PATH = f'/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pc{time_str:03d}.nc'
    # METUM_N768_PATH = f'/nobackup/earshar/borneo/case_20181021T1200Z_N768_v2/umglaa_pc{time_str:03d}.nc'

    data_n768_pc = xr.open_dataset(METUM_N768_PATH,drop_variables=['unspecified_5','unspecified_6',
                                                                   'unspecified_9','unspecified_10'])
    data_n768_pc = data_n768_pc.metpy.assign_crs(grid_mapping_name='latitude_longitude',
                                                 earth_radius=6371229.0).sel(longitude=slice(bounds[0], bounds[1]),
                                                                             latitude=slice(bounds[2], bounds[3]))

    heating_rate = 0
    vars_list = ['unspecified', 'unspecified_1', 'unspecified_2', 'unspecified_3',
                 'unspecified_4', 'unspecified_7','unspecified_8']
    for var in vars_list:
        heating_rate = heating_rate + data_n768_pc[var]

    return heating_rate


def read_all_n768_metum(bounds):

    METUM_N768_PATH = '/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa_pe*.nc'
    # METUM_N768_PATH = '/nobackup/earshar/borneo/case_20181021T1200Z_N768_v2/umglaa_pe*.nc'
    data_n768_metum = xr.open_mfdataset(METUM_N768_PATH, combine='by_coords', chunks={"t": 5}).metpy.parse_cf()
    data_n768_metum = data_n768_metum.sel(longitude=slice(bounds[0], bounds[1]),
                                          latitude=slice(bounds[2], bounds[3]),
                                          longitude_1=slice(bounds[0], bounds[1]),
                                          latitude_1=slice(bounds[2], bounds[3])
                                          )

    return data_n768_metum


def open_era5(validity_time):

    if validity_time < 0:
        ERA5_PATH = '/nobackup/earshar/borneo/bv_oct2018_early.grib'
    else:
        ERA5_PATH = '/nobackup/earshar/borneo/bv_oct2018.grib'
    era5 = xr.open_dataset(ERA5_PATH, engine="cfgrib").metpy.parse_cf()

    return era5


def read_data(data_type, output_time, sgt_run, bounds):

    METUM_4p4_PATH = f'/nobackup/earshar/borneo/20181021T1200Z_SEA4_km4p4_ra1tld'
    ERA5_PATH = f'/nobackup/earshar/borneo/bv_oct2018.grib'
    SGT_TOOL_PATH = f'/nobackup/earshar/borneo/SGTool/N768/oct/{sgt_run}/filter_4_8/conv_g7x_v5/'
    METUM_N768_PATH = f'/nobackup/earshar/borneo/case_20181021T1200Z_N768/nc/umglaa'
    # METUM_N768_PATH = f'/nobackup/earshar/borneo/case_20181021T1200Z_N768_v2/umglaa'

    if data_type == '4p4':
        start_fcst_str, date_str, tstr, data_pc, data_pd = fp.open_file(METUM_4p4_PATH, output_time, file_type='4p4')
        data_4p4_pc = subset(data_pc, bounds, validity_time=date_str)
        data_4p4_pd = subset(data_pd, bounds, validity_time=date_str)

        data_4p4 = xr.combine_by_coords([data_4p4_pc.squeeze(['t', 't_1']),
                                         data_4p4_pd.squeeze(['t', 't_1'])])

    elif data_type == 'era5':
        era5 = xr.open_dataset(ERA5_PATH, engine="cfgrib").metpy.parse_cf()
        _, date_str, _ = fp.open_file(METUM_4p4_PATH, output_time, file_type='era5')
        data_era5 = subset(era5, bounds, validity_time=date_str)

    else: # data_type == 'sgt' or data_type == 'n768'

        # SGT tool
        variabledict={}
        data_sgt = {}
        for name in output_names:
            if output_time == 0:
                time_str = int(output_time) + 12
            else:
                time_str = int(output_time)

            start_fcst_str, date_str, _ = fp.open_file(METUM_4p4_PATH, output_time, file_type='sgt')
            SGT_FILE_NAME = f'{SGT_TOOL_PATH}/OUT_{name}_{start_fcst_str}_T{time_str:03d}.nc'
            variabledict[name] = iris.load(SGT_FILE_NAME)[0]
            variabledict[name].rename(name)
            data_sgt[name] = xr.DataArray.from_iris(variabledict[name].extract(name)).sel(longitude=slice(bounds[0],
                                          bounds[1]),latitude=slice(bounds[2],bounds[3]))

        # N768 MetUM
        if output_time == 0:
            time_str = int(output_time)
        else:
            time_str = int(output_time) - 12
        data_n768_pe=f'{METUM_N768_PATH}_pe{time_str:03d}.nc'
        data_n768_pb=f'{METUM_N768_PATH}_pb{time_str:03d}.nc'
        data_768_pe=xr.open_dataset(data_n768_pe).metpy.assign_crs(grid_mapping_name='latitude_longitude',
                                                                   earth_radius=6371229.0)
        data_768_pb=xr.open_dataset(data_n768_pb).metpy.assign_crs(grid_mapping_name='latitude_longitude',
                                                                   earth_radius=6371229.0)
        data_pe=data_768_pe.sel(longitude=slice(bounds[0], bounds[1]), latitude=slice(bounds[2], bounds[3]),
                                longitude_1=slice(bounds[0], bounds[1]), latitude_1=slice(bounds[2], bounds[3]) )
        data_pb=data_768_pb.sel(longitude=slice(bounds[0], bounds[1]), latitude=slice(bounds[2], bounds[3]) )

        if time_str == 0:
            data_n768 = xr.combine_by_coords([data_pe, data_pb])
        else:
            data_n768 = xr.combine_by_coords([data_pe.squeeze('t'),
                                              data_pb.squeeze('t')])

    if data_type == '4p4':
        return data_4p4
    elif data_type == 'era5':
        return data_era5
    else: # data_type == 'n768' or data_type == 'sgt'
        return data_n768, data_sgt


def subset(data, bounds, validity_time=[]):
    """
    create subset of gridded data

    Args:
      data (xarray data array): multi-dimensional data array
      bounds (list): latitude / longitude bounds for subset

    Kwargs:
      var (str): variable to plot
      vtime (datetime object): validity time (e.g. 12Z on 23rd Oct 2018)
    """

    coords = data.coords.dims
    for c in coords:
        if c == 'p':
            dataset = '4p4'
        elif c == 'isobaricInhPa':
            dataset = 'era5'
        elif c == 'hybrid_ht_1':
            dataset = 'n768'

    if dataset == '4p4':
        data = data.sel(t_1=slice(validity_time, validity_time),
                        t=slice(validity_time, validity_time),
                        longitude=slice(bounds[0], bounds[1]),
                        latitude=slice(bounds[2], bounds[3]),
                        longitude_1=slice(bounds[0], bounds[1]),
                        latitude_1=slice(bounds[2], bounds[3]))
    else: # dataset == 'era5'
        data = data.reindex(latitude=data.latitude[::-1])  # rearrange to S --> N
        data = data.sel(time=slice(validity_time, validity_time),
                        longitude=slice(bounds[0], bounds[1]),
                        latitude=slice(bounds[2], bounds[3]))

    return data


def interp_to_evenly_spaced_levels(data_n768, data_sgt):

    # new_ht_levels = np.arange(0, 17500, 250)
    new_ht_levels = data_n768.theta['hybrid_ht'].data.astype('int32')

    data_n768 = data_n768.interp(hybrid_ht=new_ht_levels,
                                 method="linear").assign_coords(height_levels=("hybrid_ht",
                                    new_ht_levels)).swap_dims({"hybrid_ht":
                                                               "height_levels"
                                                       }
                                    )

    data_n768 = data_n768.interp(longitude_1=data_n768.theta["longitude"],
                                 latitude_1=data_n768.theta["latitude"],
                                 hybrid_ht_1=new_ht_levels,
                                 method="linear").assign_coords(height_levels=("hybrid_ht_1",
                                    new_ht_levels)).swap_dims({"hybrid_ht_1":
                                                              "height_levels"
                                                       }
                                    )

    new_ht_levels = np.arange(0, 17500, 250)
    var_names = {'u', 'v', 'w', 'ug', 'vg', 'ug_um', 'vg_um'}
    for name in var_names:
        data_sgt[name] = data_sgt[name].assign_coords(height_levels=("model_level_number",
                                                new_ht_levels)).swap_dims({"model_level_number":
                                                                           "height_levels"})
        data_sgt[name] = data_sgt[name].interp(height_levels=new_ht_levels,method="linear")
        data_sgt[name].attrs['units'] = 'm/s'

    return data_n768, data_sgt


if __name__ == '__main__':
    description = 'Plot atmospheric variables from ERA5 reanalysis, MetUM forecast or satellite data'
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
