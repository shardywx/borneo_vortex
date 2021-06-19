import numpy as np
import sys
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.maths as imath
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import metpy.calc as mpcalc
import xarray as xr

import diagnosticSGfunctions as SG
from general_eqns import vorticity
from datetime import datetime, timedelta
from iris.time import PartialDateTime
from custom_cmap import *
from windspharm.iris import VectorWind
from diagnosticSGsetup import output_names
from diagnosticSGfunctions import calc_f, exner_to_pres, reverse_lat, order_lat
from iris.cube import Cube

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

def hov_plotter(ddir, sim, md, var, out, mn):

        # define some useful variables 
        if mn == 'oct':
                fcst='20181021T1200Z'
        else:
                fcst='20181211T1200Z'
        res='N768'

        print('\n======= Running hov_plotter for {0}'.format(fcst))

        # load in global time-mean data (30-day)
        mdiri  = '/nobackup/earshar/borneo'
        mfili  = '{0}/op_gl_um_2018octnov_mean.nc'.format(mdiri)
        mcubes = iris.load(mfili)
        mean_u = mcubes.extract_strict('eastward_wind'); mean_v = mcubes.extract_strict('northward_wind') 

        # load in SGT tool data (clunky)
        level = iris.Constraint(model_level_number=14)
        fili_ug = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_ug_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_vg = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_vg_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_u  = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_u_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_v  = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_v_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_w  = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_w_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_um_u = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_ug_um_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        fili_um_v = '{0}/SGTool/N768/{1}/{2}/filter_4_8/conv_g7x_v5/OUT_vg_um_20181021T1200Z_T*.nc'.format(ddir,mn,sim)
        ug = iris.load(fili_ug,level); vg = iris.load(fili_vg,level); u = iris.load(fili_u,level); v = iris.load(fili_v,level)
        ug_um = iris.load(fili_um_u,level); vg_um = iris.load(fili_um_v,level); w = iris.load(fili_w,level)
 
        # throw out analysis at T+12, T+36, etc
        ug = ug[1:11:2]; vg = vg[1:11:2]; u = u[1:11:2]; v = v[1:11:2]
        ug_um = ug_um[1:11:2]; vg_um = vg_um[1:11:2]; w = w[1:11:2]

        # read in waves data (Kelvin, R1, WMRG waves)
        k_fili = '{0}/uz_Kelvin_rm_clim_6h_k2-40_p2-30_2018.nc'.format(mdiri)
        r1_fili= '{0}/uvz_R1_rm_clim_6h_kn2-40_p2-30_2018.nc'.format(mdiri)
        wm_fili= '{0}/uvz_WMRG_rm_clim_6h_kn2-40_p2-30_2018.nc'.format(mdiri)
        kcubes = iris.load(k_fili); rcubes = iris.load(r1_fili); wcubes = iris.load(wm_fili)

        # extract 850 hPa u and v 
        lv = 1
        r1_ut = rcubes.extract_strict('eastward_wind')[:,lv,:,:]; r1_vt = rcubes.extract_strict('northward_wind')[:,lv,:,:]
        wm_ut = wcubes.extract_strict('eastward_wind')[:,lv,:,:]; wm_vt = wcubes.extract_strict('northward_wind')[:,lv,:,:]

        # extract data only for October (for Hovmoller)
        #pdt1 = PartialDateTime(year=2018, month=10, day=15); pdt2 = PartialDateTime(year=2018, month=11, day=6)
        pdt1 = PartialDateTime(year=2018, month=10, day=22, hour=12); pdt2 = PartialDateTime(year=2018, month=10, day=26, hour=18)
        hov_time = iris.Constraint(time=lambda t: pdt1 <= t.point < pdt2)
        r1_ut = r1_ut.extract(hov_time); r1_vt = r1_vt.extract(hov_time)
        wm_ut = wm_ut.extract(hov_time); wm_vt = wm_vt.extract(hov_time)

	# plot filled contours (w, vort, ...)
        if (var == 'blh'):
                levs = [14]
        else:
                levs = [31]

	# loop over vertical levels
        for lev in levs:

                # read in PV from MetUM files ('pe' stream) --> edited 13/10/20
                fnames = '{0}/case_{2}_{1}/umglaa_pe*'.format(ddir, res, fcst)
                cubes = iris.load(fnames)
                pv = cubes.extract_strict('potential_vorticity_of_atmosphere_layer')[2:12:2,lev,:,:]
                pv.rename('PV')
                pv.units = '1e6 PVU'
                pv.convert_units('PVU')

                # also read in horizontal and vertical wind components from MetUM files (T+24, ..., T+120)
                wt = cubes.extract_strict('upward_air_velocity')[2:12:2,lev,:,:]
                ut = cubes.extract_strict('x_wind')[2:12:2,lev,:,:]
                vt = cubes.extract_strict('y_wind')[2:12:2,lev,:,:]
                
                wt.rename('vertical velocity')
                ut.rename('zonal wind component')
                vt.rename('meridional wind component')

                # define grid subset to produce Hovmöller plot
                if (md == 'wm'):
                        lon0 = 61; lon1 = 179; lat0 = 4; lat1 = 12
                else:
                        lon0 = 61; lon1 = 179; lat0 = -6; lat1 = 6
                ll_subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

                # focus on grid subset (waves data)
                r1_ut = r1_ut.extract(ll_subset); r1_vt = r1_vt.extract(ll_subset)
                wm_ut = wm_ut.extract(ll_subset); wm_vt = wm_vt.extract(ll_subset)

                # calculate mean over latitude (waves data)
                r1_ut = r1_ut.collapsed('latitude', iris.analysis.MEAN); r1_vt = r1_vt.collapsed('latitude', iris.analysis.MEAN)
                wm_ut = wm_ut.collapsed('latitude', iris.analysis.MEAN); wm_vt = wm_vt.collapsed('latitude', iris.analysis.MEAN)

                # regrid 'u' onto 'v' grid 
                ut = ut.regrid(vt,iris.analysis.Linear())
                wt = wt.regrid(vt,iris.analysis.Linear())

                # rename SGT tool variables 
                for n in range(0, 5):
                        w[n].rename('vertical velocity')
                        u[n].rename('SGT zonal wind')
                        v[n].rename('SGT meridional wind')
                        ug[n].rename('SGT geostrophic zonal wind')
                        vg[n].rename('SGT geostrophic meridional wind')
                        ug_um[n].rename('geostrophic zonal wind')
                        vg_um[n].rename('geostrophic meridional wind')

                # add time coordinate to SGT tool cubes 
                time_points = vt.coord('time').points; time_units = vt.coord('time').units
                time_name   = vt.coord('time').standard_name
                time_coord  = iris.coords.DimCoord(time_points, standard_name=time_name, units=time_units)

                # loop over cubes
                for n in range(0, 5):
                        w[n].add_aux_coord(time_coord[n])
                        u[n].add_aux_coord(time_coord[n])
                        v[n].add_aux_coord(time_coord[n])
                        ug[n].add_aux_coord(time_coord[n])
                        vg[n].add_aux_coord(time_coord[n])
                        ug_um[n].add_aux_coord(time_coord[n])
                        vg_um[n].add_aux_coord(time_coord[n])

                # merge cubes into single cube 
                u = u.merge_cube(); v = v.merge_cube(); w = w.merge_cube(); ug = ug.merge_cube()
                vg = vg.merge_cube(); ug_um = ug_um.merge_cube(); vg_um = vg_um.merge_cube()

                # add metadata (units)
                u.units = ut.units; v.units = vt.units; w.units = wt.units

                # regrid SGT tool variables onto MetUM grid 
                w = w.regrid(wt,iris.analysis.Linear()); v = v.regrid(wt,iris.analysis.Linear())
                u = u.regrid(wt,iris.analysis.Linear()); ug = ug.regrid(wt,iris.analysis.Linear())
                vg = vg.regrid(wt,iris.analysis.Linear()); ug_um = ug_um.regrid(wt,iris.analysis.Linear())
                vg_um = vg_um.regrid(wt,iris.analysis.Linear())

                # regrid global mean variables onto MetUM grid 
                mean_u.coord(axis='x').coord_system = ut[0,:,:].coord(axis='x').coord_system
                mean_u.coord(axis='y').coord_system = ut[0,:,:].coord(axis='y').coord_system
                mean_v.coord(axis='x').coord_system = ut[0,:,:].coord(axis='x').coord_system
                mean_v.coord(axis='y').coord_system = ut[0,:,:].coord(axis='y').coord_system
                mean_u = mean_u.regrid(ut[0,:,:],iris.analysis.Linear()); mean_v = mean_v.regrid(ut[0,:,:],iris.analysis.Linear())

                # focus on grid subset (MetUM)
                wt = wt.extract(ll_subset); vt = vt.extract(ll_subset); ut = ut.extract(ll_subset); pv = pv.extract(ll_subset)

                # calculate mean over latitude (MetUM)
                ut = ut.collapsed('latitude', iris.analysis.MEAN); vt = vt.collapsed('latitude', iris.analysis.MEAN)
                wt = wt.collapsed('latitude', iris.analysis.MEAN); pv = pv.collapsed('latitude', iris.analysis.MEAN)

                # focus on grid subset (SGT)
                w = w.extract(ll_subset); v = v.extract(ll_subset); u = u.extract(ll_subset); ug = ug.extract(ll_subset)
                vg = vg.extract(ll_subset); ug_um = ug_um.extract(ll_subset); vg_um = vg_um.extract(ll_subset) 

                # calculate mean over latitude (SGT)
                w = w.collapsed('latitude', iris.analysis.MEAN); u = u.collapsed('latitude', iris.analysis.MEAN)
                v = v.collapsed('latitude', iris.analysis.MEAN); ug = ug.collapsed('latitude', iris.analysis.MEAN)
                vg = vg.collapsed('latitude', iris.analysis.MEAN); ug_um = ug_um.collapsed('latitude', iris.analysis.MEAN)
                vg_um = vg_um.collapsed('latitude', iris.analysis.MEAN)

                # focus on grid subset (time-mean data)
                mean_u = mean_u.extract(ll_subset); mean_v = mean_v.extract(ll_subset)

                # calculate mean over latitude (time-mean data)
                mean_u = mean_u.collapsed('latitude', iris.analysis.MEAN); mean_v = mean_v.collapsed('latitude', iris.analysis.MEAN)

                # tidy up metadata before cube operations
                ut.remove_coord('forecast_period'); vt.remove_coord('forecast_period'); wt.remove_coord('forecast_period')
                ut.remove_coord('forecast_reference_time'); vt.remove_coord('forecast_reference_time')
                wt.remove_coord('forecast_reference_time')

                # subtract time-mean data from MetUM
                ut = ut - mean_u.data; vt = vt - mean_v.data

                # subtract time-mean from SGT
                u = u - mean_u.data; v = v - mean_v.data

                # calculate residual (MetUM - SGT) 
                ur = ut - u; vr = vt - v; wr = wt - w
                ur.rename('residual zonal wind'); vr.rename('residual meridional wind'); wr.rename('residual vertical velocity')

                # plotting resources 
                dl = 1; v_min = -10; v_max = -v_min + dl
                Cmap_v, norm_v, vLevels = normalise_cmap(v_min,v_max,0,dl)

                # WMRG wave plotting resources (smaller range)
                dl_wm = 0.5; wm_min = -4.0; wm_max = -wm_min + dl_wm
                Cmap_wm, norm_wm, wmLevels = normalise_cmap(wm_min, wm_max, 0, dl_wm)

                # customise output figure 
                fig = plt.figure(figsize=(10, 13))

                # setup figure axes [left, bottom, width, height] and invert y-axis
                ax = fig.add_axes([0.10, 0.05, 0.78, 0.85])
                ax.invert_yaxis()

                # create time strings for plot
                y_labels = ['','','','','']
                values = [24, 48, 72, 96, 120]
                for i, n in enumerate(values):
                        VT = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=n))
                        y_labels[i] = VT.strftime('%HZ %d %b [T+{}]'.format(n) )

                # plot the data 
                if (md == 'um'): # MetUM
                        if (var == 'u'):
                                cf = iplt.contourf(ut, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(vt, axes=ax, levels=vLevels, cmap=Cmap_v)
                        else:
                                cf = iplt.contourf(wt, axes=ax, levels=vLevels, cmap=Cmap_v)
                elif (md == 'sgt'): # SGT 
                        if (var == 'u'):
                                cf = iplt.contourf(u, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(v, axes=ax, levels=vLevels, cmap=Cmap_v)
                        else:
                                cf = iplt.contourf(w, axes=ax, levels=vLevels, cmap=Cmap_v)
                elif (md == 'diff'): # residual (MetUM - SGT)
                        if (var == 'u'):
                                cf = iplt.contourf(ur, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(vr, axes=ax, levels=vLevels, cmap=Cmap_v)
                        else:
                                cf = iplt.contourf(wr, axes=ax, levels=vLevels, cmap=Cmap_v)
                elif (md == 'r1'): # R1 wave data 
                        if (var == 'u'):
                                cf = iplt.contourf(r1_ut, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(r1_vt, axes=ax, levels=vLevels, cmap=Cmap_v)
                elif (md == 'wm'): # WMRG wave data 
                        if (var == 'u'):
                                cf = iplt.contourf(wm_ut, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(wm_vt, axes=ax, levels=vLevels, cmap=Cmap_v)
                else: # Kelvin wave data 
                        if (var == 'u'):
                                cf = iplt.contourf(k_ut, axes=ax, levels=vLevels, cmap=Cmap_v)
                        elif (var == 'v'):
                                cf = iplt.contourf(k_vt, axes=ax, levels=vLevels, cmap=Cmap_v)                        

                # ticks and tick labels 
                #ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
                #ax.set_yticklabels(y_labels)

                # colourbar 
                plt.colorbar(cf, orientation='horizontal', pad=0.04)
                plt.ylabel('Time (h)')
                plt.axis('tight')

                fili_ts='./hov_lev{2}_{0}_{1}.png'.format(md,var,lev)
                fig.savefig(fili_ts)

                print('Output file created. Moving onto next model level...')

if __name__ == '__main__':
        # execute only if run as a script
        main()
