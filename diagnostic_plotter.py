import numpy as np
import sys
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.maths as imath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd 
import xarray as xr
import diagnosticSGfunctions as SG

from general_eqns import vorticity
from datetime import datetime, timedelta
from custom_cmap import *
from windspharm.iris import VectorWind
from diagnosticSGsetup import output_names
from diagnosticSGfunctions import calc_f, exner_to_pres, reverse_lat, order_lat
from iris.cube import Cube
from iris.time import PartialDateTime
import cmocean

'''
Useful IRIS/matplotlib/numpy resources: 
(1) https://www.ecmwf.int/sites/default/files/elibrary/2015/13340-
iris-python-package-analysis-and-visualisation-meteorological-data.pdf
(2) https://scitools.org.uk/iris/docs/latest/userguide/index.html
(3) https://scitools.org.uk/iris/docs/latest/iris/iris.html
(4) https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.figure.Figure.html
(5) https://scitools.org.uk/iris/docs/latest/userguide/interpolation_and_regridding.html 
(6) https://scitools.org.uk/iris/docs/latest/iris/iris/cube.html#iris.cube.Cube.regrid
(7) https://ajdawson.github.io/windspharm/latest/
(8) https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.html
'''

'''
Useful scripts on GitHub (mostly John Ashcroft so far...):
(1) https://github.com/johnash92/tc_analysis/blob/
ddac47195521e9727832801aa1b01253b575d7d6/large_scale_flow/vrt_slp_geop.py
(2) https://github.com/johnash92/tc_analysis/blob/
ddac47195521e9727832801aa1b01253b575d7d6/large_scale_flow/streamlines.py
(3) https://www.sciencedirect.com/topics/earth-and-planetary-sciences/geostrophic-wind
'''

def round_int(n):
        '''
        round the given integer to a whole number ending in zero
        '''
        # smaller multiple
        a = (n // 10) * 10
        # larger multiple 
        b = a + 10
        # return the closest of the two 
        return (b if n-a > b-n else a)

def area_ave(cube): 
        '''
        calculate the area average of a 2D array over a fixed region 
        '''
        coords = ('longitude','latitude')
        cube_ave = cube.collapsed(coords,iris.analysis.MEAN)
        return cube_ave

def area_sum(cube):
        ''' 
        calculate the area average of a 2D array over a fixed region 
        '''
        coords = ('longitude','latitude')
        cube_sum = cube.collapsed(coords,iris.analysis.SUM)
        return cube_sum

def diagnostic_plotter(ddir, fcst, Tp, res, sim, md, wind, var, out, size, plane, mn):

        print('\n======= Running diagnostic_plotter for {0} T+{1}'.format(fcst, Tp))

        # define variable dictionary (data structure that maps one value to another)
	# print('Loading...') --> loading all output variables from the SGT tool
        variabledict={}
        for name in output_names:  # import this information from 'diagnosticSGsetup'
                var_name='{}'.format(name)
                print(var_name)
                fn = '{0}/SGTool/{4}/{5}/{6}/filter_4_8/conv_g7x_v5/OUT_{1}_{2}_T{3:03d}.nc'.format(ddir,name,
                                                                                                    fcst,Tp,res,
                                                                                                    mn,sim)
                variabledict[name] = iris.load(fn)[0]
                variabledict[name].rename(name)

        # define grid subset to reduce memory required to run script 
        if fcst == '20181021T1200Z' or fcst == '20181211T1200Z':
                if (size == 'zoom'):
                        #lon0 = 101; lon1 = 119; lat0 = 1; lat1 = 14
                        lon0 = 93; lon1 = 123; lat0 = -3; lat1 = 20
                        #lon0 = 142; lon1 = 162; lat0 = 3; lat1 = 18;
                elif (size == 'ext'):
                        lon0 = 75; lon1 = 155; lat0 = -15; lat1 = 45;
                        # lon0 = 76; lon1 = 179; lat0 = -15; lat1 = 55;
                else: # reg
                        if var == 'prcp':
                                lon0 = 91; lon1 = 139; lat0 = -14; lat1 = 24;
                        else:
                                lon0 = 93; lon1 = 123; lat0 = -3; lat1 = 20
                                #lon0 = 91; lon1 = 159; lat0 = -14; lat1 = 25;
        else:
                lon0 = -60; lon1 = 0; lat0 = 30; lat1 = 70; 

	# use shorthand notation to define this region
	# use of 'subset' below was returning '<No cubes>' errors, 
        # b/c lat/lon names were incorrect
        subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

        # # read in global time-mean data (30-day)
        # mdiri  = '/nobackup/earshar/borneo'
        # mfili  = '{0}/op_gl_um_2018octnov_mean.nc'.format(mdiri)
        # mcubes = iris.load(mfili)
        # mean_u = mcubes.extract_strict('eastward_wind')
        # mean_v = mcubes.extract_strict('northward_wind')

        # # read in waves data (Kelvin, R1, WMRG waves)
        # k_fili = '{0}/uz_Kelvin_rm_clim_6h_k2-40_p2-30_2018.nc'.format(mdiri)
        # r1_fili= '{0}/uvz_R1_rm_clim_6h_kn2-40_p2-30_2018.nc'.format(mdiri)
        # wm_fili= '{0}/uvz_WMRG_rm_clim_6h_kn2-40_p2-30_2018.nc'.format(mdiri) 
        # kcubes = iris.load(k_fili); rcubes = iris.load(r1_fili); wcubes = iris.load(wm_fili)

	# plot filled contours (w, vort, ...)
        if (var == 'blh'):
                levs = [14]
        else:
                levs = [16]#,23,24,25,26,27,28,29,30]

	# loop over vertical levels
        for lev in levs:

                # read in PV from MetUM files ('pe' stream) --> edited 13/10/20
                fnames = '{0}/case_{3}_{2}/umglaa_pe{1:03d}'.format(ddir, Tp-12, res, fcst)
                cubes = iris.load(fnames)
                pv = cubes.extract_strict('potential_vorticity_of_atmosphere_layer')[lev,:,:]
                pv.rename('PV'); pv.units = '1e6 PVU'; pv.convert_units('PVU')

                # retrieve sigma information
                sigma = cubes.extract_strict('potential_vorticity_of_atmosphere_layer').coord('sigma')

                # also read in horizontal and vertical wind components from MetUM files
                wt = cubes.extract_strict('upward_air_velocity')[lev,:,:]
                ut = cubes.extract_strict('x_wind')[lev,:,:]
                vt = cubes.extract_strict('y_wind')[lev,:,:]

                wt.rename('vertical velocity')
                ut.rename('zonal wind component')
                vt.rename('meridional wind component')

                # # read in GPM-IMERG precipitation data (30 min interval)
                # pdiri  = '/nobackup/earshar/borneo/'
                # pfili  = '{0}/GPMHH_201810.nc'.format(pdiri)
                # pcubes = iris.load(pfili); gpm_prcp = pcubes.extract_strict('precipitationCal')

                # choose single time for analysis 
                if Tp == 12:
                        pdt1 = PartialDateTime(year=2018, month=10, day=21, hour=22, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=2, minute=00)
                elif Tp == 15:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=1, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=5, minute=00)
                elif Tp == 18:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=4, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=8, minute=00)
                elif Tp == 21:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=7, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=11, minute=00)
                elif Tp == 24:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=10, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=14, minute=00)
                elif Tp == 27:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=13, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=17, minute=00)
                elif Tp == 30:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=16, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=20, minute=00)
                elif Tp == 33:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=19, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=22, hour=23, minute=00)
                elif Tp == 36:
                        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=22, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=2, minute=00)
                elif Tp == 39:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=1, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=5, minute=00)
                elif Tp == 42:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=4, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=8, minute=00)
                elif Tp == 45:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=7, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=11, minute=00)
                elif Tp == 48:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=10, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=14, minute=00)
                elif Tp == 51:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=13, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=17, minute=00)
                elif Tp == 54:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=16, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=20, minute=00)
                elif Tp == 57:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=19, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=23, hour=23, minute=00)
                elif Tp == 60:
                        pdt1 = PartialDateTime(year=2018, month=10, day=23, hour=22, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=2, minute=00)
                elif Tp == 63:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=1, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=5, minute=00)
                elif Tp == 66:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=4, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=8, minute=00)
                elif Tp == 69:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=7, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=11, minute=00)
                elif Tp == 72:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=10, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=14, minute=00)
                elif Tp == 75:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=13, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=17, minute=00)
                elif Tp == 78:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=16, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=20, minute=00)
                elif Tp == 81:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=19, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=24, hour=23, minute=00)
                elif Tp == 84:
                        pdt1 = PartialDateTime(year=2018, month=10, day=24, hour=22, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=2, minute=00)
                elif Tp == 87:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=1, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=5, minute=00)
                elif Tp == 90:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=4, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=8, minute=00)
                elif Tp == 93:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=7, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=11, minute=00)
                elif Tp == 96:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=10, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=14, minute=00)
                else:
                        pdt1 = PartialDateTime(year=2018, month=10, day=26, hour=10, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=26, hour=14, minute=00)

                # calculate accumulated precipitation
                if md == 'gpm':
                        acc_time = iris.Constraint(time=lambda t: pdt1 <= t.point < pdt2)
                        gpm_prcp = gpm_prcp.extract(acc_time)
                        gpm_prcp = gpm_prcp.collapsed('time', iris.analysis.SUM)

                # read in Borneo vortex track data from text file (Hodges)                         
                df = pd.read_csv('/nobackup/earshar/borneo/bv_2018102112_track.csv',
                                 na_filter=True,na_values="1.000000e+25")
                # convert time integers to datetime objects                                        
                df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')
                # extract track information between 12Z on 21st and 26th October                   
                bv_lat = df.loc[0:20, "lat_vort"]
                bv_lon = df.loc[0:20, "lon_vort"]
                bv_time = df.loc[0:20, "Time"]

                # # read in precipitation data from 4.4 km forecast
                # pdiri  = '/nobackup/earshar/borneo/'
                # pfili  = '{0}/20181021T1200Z_SEA4_km4p4_ra1tld_pverb.pp'.format(pdiri)
                # pcubes = iris.load(pfili)
                # prcp   = pcubes.extract('stratiform_rainfall_flux')[1]
                # # calculate units, using frequency of input data (either every 1 h or 15 min)
                # ntimes = prcp.shape[0]
                # if ntimes == 120:
                #         prcp = prcp * 3600
                # else:
                #         prcp = prcp * 900
                # prcp.units = 'mm hr**-1'
                # # focus on subset of times  
                # # EDIT --> MetUM precip data look strange, even though max/min are sensible
                # um_time = iris.Constraint(time = lambda t0: pdt1 <= t0.point <= pdt2)
                # prcp = prcp.extract(um_time) 
                # # calculate accumulated precipitation 
                # prcp = prcp.collapsed('time', iris.analysis.SUM)
                # prcp.rename('accumulated rainfall')

                # also read in 'pb' stream data
                bnames = '{0}/case_{3}_{2}/umglaa_pb{1:03d}'.format(ddir, Tp-12, res, fcst)
                cubes  = iris.load(bnames)

                th0 = cubes.extract_strict('air_potential_temperature')[lev,:,:]
                rh0 = cubes.extract_strict('relative_humidity')[lev,:,:]
                q0  = cubes.extract_strict('specific_humidity')[lev,:,:]
                rho = cubes.extract_strict('air_density')[lev,:,:]

                # regrid 'u' onto 'v' grid 
                ut = ut.regrid(vt,iris.analysis.Linear())

                # # regrid time-mean onto MetUM grid 
                # mean_u.coord(axis='x').coord_system = ut.coord(axis='x').coord_system
                # mean_u.coord(axis='y').coord_system = ut.coord(axis='y').coord_system
                # mean_v.coord(axis='x').coord_system = ut.coord(axis='x').coord_system
                # mean_v.coord(axis='y').coord_system = ut.coord(axis='y').coord_system
                # mean_u = mean_u.regrid(ut,iris.analysis.Linear())
                # mean_v = mean_v.regrid(ut,iris.analysis.Linear())

                # model level array 
                ht_levs = ut.coord('level_height') 

                # customise output figure 
                fig = plt.figure(figsize=(9, 6))

                # figure axes [left, bottom, width, height]
                ax = fig.add_axes([0.05, 0.05, 0.78, 0.85], projection=ccrs.PlateCarree())

                # domain extent
                ax.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], ccrs.PlateCarree())
                # gridlines
                ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
                # lat/lon tickmarks
                x_ticks = np.arange(round_int(lon0), round_int(lon1)+5, 5.0)
                y_ticks = np.arange(round_int(lat0), round_int(lat1)+5, 5.0)
                ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
                # coastlines
                ax.coastlines(resolution='10m', color='k', linewidth=1)
                VT = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=Tp))
                if (md == 'mean'):
                        ax.set_title('', loc='right')
                else:
                        ax.set_title(VT.strftime('%HZ %a %d %b %Y [T+{}]'.format(Tp)), loc='right')

                # read in 3D windspeed data from MetUM (or GPM-IMERG)
                if md == 'um' or md == 'gpm':
                        if (Tp == 12):
                                ua = ut[1,lev]; va = vt[1,lev]; wa = wt[1,lev]
                        else:
                                ua = ut; va = vt; wa = wt * 100; wa.units = 'cm s**-1'
                        # geostrophic wind components
                        ug = variabledict['ug_um'][lev]; vg = variabledict['vg_um'][lev]
                        ug.units = 'm s**-1'; vg.units = 'm s**-1'
                        # calculate relative vorticity
                        V  = VectorWind(ua, va); vort = V.vorticity() * 100000
                        vort.rename('relative_vorticity')
                        vort.units = '10-5 s**-1'

                        # regrid variables onto SGT grid (test)
                        wa = wa.regrid(ug,iris.analysis.Linear())
                        va = va.regrid(ug,iris.analysis.Linear())
                        ua = ua.regrid(ug,iris.analysis.Linear())
                        vort = vort.regrid(ug,iris.analysis.Linear())

                        # regrid output from SGT tool onto MetUM grid
                        #ug = ug.regrid(ua,iris.analysis.Linear())
                        #vg = vg.regrid(va,iris.analysis.Linear())

                        # calculate ageostrophic wind using geostrophic wind
                        uaa = ua - ug; vaa = va - vg
                        # add metadata
                        uaa.rename('ageostrophic_wind_ucomp'); vaa.rename('ageostrophic_wind_vcomp')
                        
                # read in 3D wind field from SGT tool 
                elif (md == 'sgt'):
                        va = variabledict['v'][lev]; ua = variabledict['u'][lev]
                        wa = variabledict['w'][lev] * 100
                        wa.units = 'cm s**-1'
                        # calculate relative vorticity
                        V  = VectorWind(ua, va); vort = V.vorticity() * 100000
                        vort.rename('relative_vorticity')
                        vort.units = '10-5 s**-1'
                        # read in geostrophic wind components
                        ug = variabledict['ug'][lev]; vg = variabledict['vg'][lev]
                        # calculate ageostrophic wind components 
                        uaa = ua - ug; vaa = va - vg
                        # boundary layer height (index)
                        blh_ind = variabledict['blh']
                        # balanced heating 
                        heat1 = variabledict['balheat1']
                        heat2 = variabledict['balheat2']
                        '''
                        # regrid onto MetUM grid 
                        ua = ua.regrid(ut,iris.analysis.Linear())
                        va = va.regrid(vt,iris.analysis.Linear())
                        wa = wa.regrid(wt,iris.analysis.Linear()) 
                        vort = vort.regrid(ut,iris.analysis.Linear())
                        ug = ug.regrid(ut,iris.analysis.Linear())
                        vg = vg.regrid(ut,iris.analysis.Linear())
                        uaa = uaa.regrid(ut,iris.analysis.Linear())
                        vaa = vaa.regrid(ut,iris.analysis.Linear())
                        '''
                # read in both datasets and create difference plots
                else:
                        v_sg = variabledict['v'][lev]; u_sg = variabledict['u'][lev]
                        w_sg = variabledict['w'][lev]
                        u_sg.units = 'm s**-1'; v_sg.units = 'm s**-1'; w_sg.units = 'm s**-1'
                        # regrid SGT tool output onto MetUM grid 
                        u_sg = u_sg.regrid(ut,iris.analysis.Linear()); 
                        v_sg = v_sg.regrid(vt,iris.analysis.Linear())
                        w_sg = w_sg.regrid(wt,iris.analysis.Linear())
                        # calculate speed (SGT tool)
                        spd_sg = (u_sg * u_sg + v_sg * v_sg) ** 0.5
                        # calculate speed (MetUM)
                        spd_um = (ut * ut + vt * vt) ** 0.5
                        # calculate difference fields
                        ua = ut - u_sg; va = vt - v_sg; wa = wt - w_sg * 100
                        wa.units = 'cm s**-1'
                        # calculate relative vorticity from difference fields 
                        V  = VectorWind(ua, va); vort = V.vorticity() * 100000
                        vort.rename('relative_vorticity')
                        vort.units = '10-5 s**-1'
                        '''
                        # add code to plot geostrophic and ageostrophic wind components (14/10/20)
                        '''

                # define lat/lon grids (both MetUM and SGT tool)
                x0 = ua.coord('longitude').points; y0 = ua.coord('latitude').points

                # add options for vorticity on pressure levels (filled contours)
                dl = 1; vrtmin = -20; vrtmax = -vrtmin + dl
                Cmap_vort, norm_vort, vortLevels = normalise_cmap(vrtmin,vrtmax,0,dl,'bwr')
                cb_label = r'Relative vorticity $\mathregular{(s^{-1})}$'

                # difference (relative vorticity)
                dl_dv = 3; dvmin = -24; dvmax = -dvmin + dl_dv
                Cmap_dv, norm_dv, dvLevels = normalise_cmap(dvmin,dvmax,0,dl_dv,'bwr')
                cb_label = r'Relative vorticity $\mathregular{(s^{-1})}$'

                # same for horizontal wind 
                dl_v = 2; spdmin = -30; spdmax = -spdmin + dl_v
                Cmap_spd, norm_spd, spdLevels = normalise_cmap(spdmin,spdmax,0,dl_v,'bwr')
                cb_label = r'Horizontal wind $\mathregular{(m\,s^{-1})}$'

                # ageostrophic wind (SGT tool)
                dl_ua = 1; uamin = -15; uamax = -uamin + dl_ua
                Cmap_ua, norm_ua, uaLevels = normalise_cmap(uamin,uamax,0,dl_ua,'bwr') 
                cb_label = r'Ageostrophic wind $\mathregular{(m\,s^{-1})}$'

                # difference (horizontal wind)
                dl_diff = 2; diffmin = -20; diffmax = -diffmin + dl_diff
                Cmap_diff, norm_diff, diffLevels = normalise_cmap(diffmin,diffmax,0,dl_diff,'bwr')
                cb_label = r'Horizontal wind $\mathregular{(m\,s^{-1})}$'

                # contour levels for time-mean wind 
                dl = 1; v_min = -10; v_max = -v_min + dl
                Cmap_v, norm_v, vLevels = normalise_cmap(v_min,v_max,0,dl,'bwr')
                cb_label = r'Time-mean wind $\mathregular{(m\,s^{-1})}$'
                
                # create colour scale for vertical velocity 
                dl_w = 0.25; wmin = -15.0; wmax = -wmin + dl_w
                Cmap_w, norm_w, wLevels = normalise_cmap(wmin,wmax,0,dl_w,'bwr')
                cb_label = r'Vertical velocity $\mathregular{(cm\,s^{-1})}$'

                # calculate circulation around vortex, if required
                if (var == 'circ'):
                        ii = int(Tp/6)
                        lonS = bv_lon[ii]-4.0; lonF = bv_lon[ii]+4.0
                        latS = bv_lat[ii]-4.0; latF = bv_lat[ii]+4.0
                        vort0 = vort.extract(iris.Constraint(latitude=lambda z: latS<z<latF,
                                                             longitude=lambda z: lonS<z<lonF) )
                        circ = area_sum(vort0) # * (100000)
                        print(circ.data)
                        exit()
                # relative vorticity
                elif (var == 'vort'):
                        print('Starting contour plotting...')
                        if (md == 'diff'):
                                cf = iplt.contourf(vort[:, :], axes=ax, levels=dvLevels, 
                                                   cmap=Cmap_dv)
                        elif (md == 'sgt'):
                                cf = iplt.contourf(vort[:, :], axes=ax, 
                                                   levels=vortLevels, cmap=Cmap_vort)
                        else: # md == 'um'
                                cf = iplt.contourf(vort[:, :], axes=ax, 
                                                   levels=vortLevels, cmap=Cmap_vort)
                        print('Finished contour plotting...')
                # vertical velocity 
                elif (var == 'w'):
                        print('Starting contour plotting...')
                        if (res == 'N768'):
                                if (md == 'diff'): # Residual 
                                        cf = iplt.contourf(wa, axes=ax, 
                                                           levels=np.arange(-0.5, 0.5, 0.05), 
                                                           cmap='RdBu_r')
                                elif (md == 'um'):
                                        cf = iplt.contourf(wa,axes=ax,levels=wLevels,cmap=Cmap_w)
                                else: # SGT or MetUM 
                                        cf = iplt.contourf(wa,axes=ax,levels=wLevels,cmap=Cmap_w)
                        print('Finished contour plotting...')
                # zonal wind
                elif (var == 'u'):
                        print('Starting contour plotting...')
                        if (md == 'diff'):
                                cf = iplt.contourf(ua, axes=ax, levels=diffLevels, 
                                                   cmap=Cmap_diff) # cmap='RdBu_r'
                        elif (md == 'mean'):
                                cf = iplt.contourf(mean_u, axes=ax, levels=vLevels, cmap=Cmap_v)
                        else:
                                cf = iplt.contourf(ua, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # meridional wind
                elif (var == 'v'):
                        print('Starting contour plotting...')
                        if (md == 'diff'):
                                cf = iplt.contourf(va, axes=ax, levels=diffLevels, 
                                                   cmap=Cmap_diff) # cmap='RdBu_r'
                        elif (md == 'mean'):
                                cf = iplt.contourf(mean_v, axes=ax, levels=vLevels, 
                                                   cmap=Cmap_v)
                        else:
                                #cf = iplt.contourf(va, axes=ax, levels=np.arange(-50., 50., 5), 
                                #cmap='seismic')
                                cf = iplt.contourf(va, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # vector wind
                elif (var == 'spd'):
                        print('Starting contour plotting...')
                        if (md == 'diff'):
                                spd = spd_um - spd_sg
                                cf = iplt.contourf(spd, axes=ax, levels=np.arange(-20., 20., 2), 
                                                   cmap='RdBu_r')
                        else:
                                spd = (ua * ua + va * va) ** 0.5
                                cf = iplt.contourf(spd, axes=ax, 
                                                   levels=np.arange(3., 30., 3), cmap='BuPu')
                        print('Finished contour plotting...')                        
                # precipitation rate 
                elif (var == 'prcp'):
                        print('Starting contour plotting...')
                        if md == 'um':
                                cf = iplt.contourf(prcp, axes=ax, 
                                                   levels=[0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0], 
                                                   cmap=cmocean.cm.haline_r)
                        else: # GPM-IMERG
                                cf = iplt.contourf(gpm_prcp, axes=ax, 
                                                   levels=[0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0], 
                                                   cmap=cmocean.cm.haline_r)
                        print('Finished contour plotting...')                        
                # relative humidity 
                elif (var == 'rh'):
                        print('Starting contour plotting...')
                        dl = 5.0; rhmin = 40.0; rhmax = 100.0 
                        levs=np.arange(rhmin,rhmax+dl,dl); Cmap='BuPu'
                        cf = iplt.contourf(rh0, axes=ax, levels=levs, cmap=Cmap)
                        print('Finished contour plotting...')
                # specific humidity 
                elif (var == 'q'):
                        print('Starting contour plotting...')
                        dl = 0.2; rhmin = 0.0; rhmax = 2.0 
                        levs=np.arange(rhmin,rhmax+dl,dl); Cmap='BuPu'
                        cf = iplt.contourf(q0, axes=ax, levels=levs, cmap=Cmap)
                        print('Finished contour plotting...')
                # potential vorticity 
                elif (var == 'pv'):
                        print('Starting contour plotting...')
                        cf = iplt.contourf(pv, axes=ax, levels=[0.05, 0.1, 0.2, 0.3, 0.4, 
                                                                0.5, 0.6, 0.8, 1.0, 1.2, 1.5], 
                                           cmap='BuPu')
                        print('Finished contour plotting...')
                # geostrophic wind (u)
                elif (var == 'ug'):
                        print('Starting contour plotting...')
                        cf = iplt.contourf(ug, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # geostrophic wind (v)
                elif (var == 'vg'):
                        print('Starting contour plotting...')
                        cf = iplt.contourf(vg, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # geostrophic vector wind
                elif (var == 'spd_g'): 
                        print('Starting contour plotting...')
                        spd = (ug * ug + vg * vg) ** 0.5
                        cf = iplt.contourf(spd, axes=ax, levels=np.arange(3., 30., 3), cmap='BuPu')
                        print('Finished contour plotting...')
                # ageostrophic wind (u)
                elif (var == 'ua'):
                        print('Starting contour plotting...')
                        if md == 'sgt':
                                cf = iplt.contourf(uaa, axes=ax, levels=uaLevels, cmap=Cmap_ua)
                        else: # N768
                                cf = iplt.contourf(uaa, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # ageostrophic wind (v)
                elif (var == 'va'):
                        print('Starting contour plotting...')
                        if md == 'sgt':
                                cf = iplt.contourf(vaa, axes=ax, levels=uaLevels, cmap=Cmap_ua)
                        else: # N768
                                cf = iplt.contourf(vaa, axes=ax, levels=spdLevels, cmap=Cmap_spd)
                        print('Finished contour plotting...')
                # boundary layer height 
                elif (var == 'blh'):
                        print('Starting contour plotting...')
                        cf = iplt.contourf(blh_ind, axes=ax)#, levels=np.arange(2., 9., 1), cmap='BuPu')
                        print('Finished contour plotting...')

                # now calculate and overlay wind vectors --> reduce number of grid points before plotting
                if (size == 'zoom'): 
                        skip = 1
                        vScale = 300
                        rvec = 10
                else:
                        if size == 'reg' and var == 'w':
                                if md != 'um':
                                        if sim == 'diab':
                                                skip = 2; vScale = 35; rvec = 2 # SGT (rvec = 5)
                                        elif sim == 'geo':
                                                skip = 2; vScale = 70; rvec = 5 # SGT (rvec = 10) 
                                        else: # control (full forcing)
                                                skip = 2; vScale = 150; rvec = 10 #
                                else:
                                        skip = 2; vScale = 200; rvec = 10 # MetUM (rvec = 20)
                        else:
                                skip = 4; vScale = 200; rvec = 10

                # overlay wind vectors (+ reference vector)
                print('Adding wind vectors to the plot...')
                if (wind == 'full'):
                        if (md == 'mean'):
                                q = iplt.quiver(mean_u[::skip, ::skip], mean_v[::skip, ::skip], 
                                                angles='xy', scale=400)
                                plt.quiverkey(q, 0.07, 0.82, rvec, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                              labelpos='E', coordinates='figure')
                        else: # (SGT or MetUM)
                                if size == 'reg':
                                        q = iplt.quiver(ua[::skip, ::skip], va[::skip, ::skip], 
                                                        angles='xy', scale=vScale)
                                        plt.quiverkey(q, 0.10, 0.92, rvec, 
                                                      r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                      labelpos='E', coordinates='figure')
                                elif size == 'ext':
                                        q = iplt.quiver(ua[::skip, ::skip], va[::skip, ::skip], 
                                                        angles='xy', scale=vScale)
                                        plt.quiverkey(q, 0.09, 0.90, rvec, 
                                                      r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                      labelpos='E', coordinates='figure')
                                else: # zoom
                                        q = iplt.quiver(ua[::skip, ::skip], va[::skip, ::skip], 
                                                        angles='xy', scale=vScale)
                                        plt.quiverkey(q, 0.08, 0.90, rvec, 
                                                      r'$20\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                      labelpos='E', coordinates='figure')
                elif (wind == 'geo'):
                        q = iplt.quiver(ug[::skip, ::skip], vg[::skip, ::skip], angles='xy', scale=600)
                elif (wind == 'ageo'):
                        if size == 'reg':
                                q = iplt.quiver(uaa[::skip, ::skip], vaa[::skip, ::skip], 
                                                angles='xy', scale=vScale)
                                if md != 'um':
                                        if sim == 'diab':
                                                plt.quiverkey(q, 0.09, 0.90, rvec, 
                                                              r'$2\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                              labelpos='E', coordinates='figure')
                                        else:
                                                plt.quiverkey(q, 0.10, 0.90, rvec, 
                                                              r'$5\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                              labelpos='E', coordinates='figure')
                                else:
                                        plt.quiverkey(q, 0.09, 0.82, rvec, 
                                                      r'$20\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                                      labelpos='E', coordinates='figure')
                        else:
                                vScale = 400; rvec = 10
                                q = iplt.quiver(uaa[::skip, ::skip], vaa[::skip, ::skip], 
                                                angles='xy', scale=vScale)
                                plt.quiverkey(q, 0.09, 0.90, rvec, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                              labelpos='E', coordinates='figure')

                '''
                if (md == 'um'):
                        plt.quiverkey(q, 0.09, 0.90, 10, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                      labelpos='E', coordinates='figure')
                elif (md == 'mean'):
                        plt.quiverkey(q, 0.07, 0.82, 10, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                      labelpos='E', coordinates='figure')                        
                else: # (md == 'sgt')
                        if (sim == 'diab'):
                                plt.quiverkey(q, 0.09, 0.90, 10, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                              labelpos='E', coordinates='figure')
                        else:
                                plt.quiverkey(q, 0.09, 0.90, 10, r'$10\ \mathrm{m}\ \mathrm{s}^{-1}$',
                                              labelpos='E', coordinates='figure')
                '''

                print('Have added wind vector arrows to the plot...')
                                                        
                # add colourbar
                if var != 'circ':
                        if (size == 'reg'):
                                plt.colorbar(cf, fraction=0.025, 
                                             pad=0.06, extend='both').set_label(cb_label)
                        else:
                                plt.colorbar(cf, fraction=0.032, 
                                             pad=0.06, extend='both').set_label(cb_label)
                        
                # output the plot to X11 window
                if (out == 'x11'):
                        plt.show()
                        exit()

                # produce output plot 
                if (md == 'sgt'):
                        if (size == 'zoom'):
                                fili='./{2}_sgt_{3}_{5}_{4}_lev{0}_{7}_T{1}.{6}'.format(lev,Tp,res,
                                                                                        sim,var,wind,
                                                                                        out,size)
                        else:
                                fili='./{2}_sgt_{3}_{5}_{4}_lev{0}_T{1}.{6}'.format(lev,Tp,res,sim,
                                                                                    var,wind,out)
                elif (md == 'um'):
                        if (size == 'zoom'):
                                fili='./{2}_metum_control_{5}_{4}_lev{0}_{7}_T{1}.{6}'.format(lev,Tp,res,
                                                                                              sim,var,wind,
                                                                                              out,size)
                        else:
                                fili='./{2}_metum_control_{5}_{4}_lev{0}_T{1}.{6}'.format(lev,Tp,res,
                                                                                          sim,var,wind,out)
                elif (md == 'gpm'):
                        fili='./{2}_gpm_{5}_{4}_lev{0}_T{1}.{6}'.format(lev,Tp,res,sim,
                                                                        var,wind,out)
                else:
                        if (size == 'zoom'):
                                fili='./{2}_diff_{5}_{4}_lev{0}_{7}_T{1}.{6}'.format(lev,Tp,res,
                                                                                     sim,var,
                                                                                     wind,out,size)
                        else:
                                fili='./{2}_diff_{5}_{4}_lev{0}_T{1}.{6}'.format(lev,Tp,res,sim,
                                                                                 var,wind,out)
                                                                
                fig.savefig(fili)
                                                                
                print('Output file created. Moving onto next model level...')

if __name__ == '__main__':
        # execute only if run as a script 
        main()
