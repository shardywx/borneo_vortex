import numpy as np
import sys
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.maths as imath
import iris.coords
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import metpy.calc as mpcalc 
from xarray import DataArray
from metpy.interpolate import cross_section

import diagnosticSGfunctions as SG
from general_eqns import vorticity
from datetime import datetime, timedelta
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

def cs_plotter(ddir, fcst, Tp, res, sim, md, wind, var, out, size, plane, mn):

        print('\n======= Running diagnostic_plotter for {0} T+{1}'.format(fcst, Tp))

        # define grid subset defined above (surely way to tidy up this step --> 13/10/20)  
        if (fcst == '20181021T1200Z'):
                if (size == 'zoom'):
                        lon0 = 100; lon1 = 120; lat0 = 0; lat1 = 15;
                elif (size == 'ext'):
                        lon0 = 76; lon1 = 160; lat0 = -12; lat1 = 50;
                else:
                        lon0 = 94; lon1 = 160; lat0 = -16; lat1 = 21;
        else:
                lon0 = -60; lon1 = 0; lat0 = 30; lat1 = 70;
        # constrain by focusing on lower troposphere + single latitude
        ht_sub = iris.Constraint(model_level_number=lambda m: 0<m<=34)
        subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

        # define variable dictionary (data structure that maps one value to another)
        # print('Loading...') --> loading all output variables from the SGT model  
        variabledict={}
        for name in output_names:  # import this information from 'diagnosticSGsetup'
                print('   {}'.format(name))
                fn = '{0}/SGTool/{4}/{5}/{6}/filter_4_8/conv_g7x_v5/OUT_{1}_{2}_T{3:03d}.nc'.format(ddir,name,fcst,Tp,res,mn,sim)
                variabledict[name] = iris.load(fn,subset)[0]
                variabledict[name].rename(name)

        # read in PV from MetUM files ('pe' stream)
        fnames = '{0}/case_{3}_{2}/umglaa_pe{1:03d}'.format(ddir, Tp-12, res, fcst)
        cubes = iris.load(fnames,subset & ht_sub)

        # also read in horizontal and vertical wind components from MetUM files 
        pv0 = cubes.extract_strict('potential_vorticity_of_atmosphere_layer')
        pv0.rename('PV'); pv0.units = '1e6 PVU'; pv0.convert_units('PVU')
        wt0 = cubes.extract_strict('upward_air_velocity')
        ut0 = cubes.extract_strict('x_wind'); vt0 = cubes.extract_strict('y_wind')
        wt0.rename('vertical velocity'); ut0.rename('zonal wind component')
        vt0.rename('meridional wind component')

        # also read in 'pb' stream data 
        bnames = '{0}/case_{3}_{2}/umglaa_pb{1:03d}'.format(ddir, Tp-12, res, fcst)
        cubes  = iris.load(bnames,subset & ht_sub)
        th0 = cubes.extract_strict('air_potential_temperature')
        rh0 = cubes.extract_strict('relative_humidity')
        q0 = cubes.extract_strict('specific_humidity') * 1000
        q0.units = 'g kg**-1'

	# regrid 'u' onto 'v' grid 
        ut0 = ut0.regrid(vt0,iris.analysis.Linear())

        # customise output figure 
        fig = plt.figure(figsize=(9, 6))

        # read in 3D windspeed data from MetUM
        if (md == 'um'):
                ua = ut0[:]; va = vt0[:]; wa = wt0[:] * 100; wa.units = 'cm s**-1'
                # put all components on same grid 
                ua = ua.regrid(wa,iris.analysis.Linear()); va = va.regrid(wa,iris.analysis.Linear())
                # geostrophic wind components
                ug = variabledict['ug_um'][:]; vg = variabledict['vg_um'][:]
                ug.units = 'm s**-1'; vg.units = 'm s**-1'
                # calculate relative vorticity
                if (var == 'vort'):
                        V  = VectorWind(ua, va); vort = V.vorticity()
                # regrid output from SGT tool onto MetUM grid
                ug = ug.regrid(ua,iris.analysis.Linear()); vg = vg.regrid(va,iris.analysis.Linear())

        # read in 3D wind field from SGT tool
        else:
                wa = variabledict['w'][:] * 100; wa.units = 'cm s**-1'
                va = variabledict['v'][:]; ua = variabledict['u'][:]
                V  = VectorWind(ua, va); vort = V.vorticity()
                # calculate geostrophic and ageostrophic wind components
                ug = variabledict['ug'][:]; vg = variabledict['vg'][:]
                uaa = ua - ug; vaa = va - vg
                # regrid onto MetUM grid 
                ua = ua.regrid(ut0,iris.analysis.Linear()); va = va.regrid(ut0,iris.analysis.Linear())
                wa = wa.regrid(ut0,iris.analysis.Linear()); vort = vort.regrid(ut0,iris.analysis.Linear())

        # constrain further by focusing on single latitude / longitude
        if (Tp == 72):
                lt = 7.0; ln = 105.0
        elif (Tp == 48):
                lt = 6.0; ln = 108.0
        else:
                print('vortex not prominent at this time. exiting...')
                exit()

        # customise cross section slice --> need to update 
        lat_sub = iris.Constraint(latitude=lambda l: lt-0.06 < l < lt+0.08)
        lon_sub = iris.Constraint(longitude=lambda l: ln-0.1 < l < ln+0.12)

        # what kind of cross-section do we want? 
        cs0 = 'WE'

        # choose direction of cross-section
        if (var == 'u'):
                plane = 'lon'; xy_sub = lon_sub; xy_coord = 'latitude'
        elif (var == 'v'):
                plane = 'lat'; xy_sub = lat_sub; xy_coord = 'longitude'
        else: # var == 'rh' or 'q'
                if (cs0 == 'WE'):
                        plane = 'lat'; xy_sub = lat_sub; xy_coord = 'longitude'
                else:
                        plane = 'lon'; xy_sub = lon_sub; xy_coord = 'latitude'


        # narrow down further to single latitude / longitude
        wa = wa.extract(xy_sub); va = va.extract(xy_sub); ua = ua.extract(xy_sub); pv0 = pv0.extract(xy_sub)
        th0 = th0.extract(xy_sub); rh0 = rh0.extract(xy_sub); q0 = q0.extract(xy_sub)

        # choose variable to plot 
        if (var == 'v'):
                cs = va
                dl = 1.0; vmin = -15.0; vmax = -vmin + dl
                Cmap, norm, levs = normalise_cmap(vmin,vmax,0,dl)
        elif (var == 'u'): 
                cs = ua
                dl = 1.0; vmin = -15.0; vmax = -vmin + dl
                Cmap, norm, levs = normalise_cmap(vmin,vmax,0,dl)
        elif (var == 'rh'):
                cs = rh0
                dl = 5.0; rmin = 40.0; rmax = 100.0
                levs=np.arange(rmin,rmax+dl,dl); Cmap='BuPu'
        elif (var == 'q'):
                cs = q0
                dl = 1.0; qmin = 1.0; qmax = 18.0
                levs=np.arange(qmin,qmax+dl,dl); Cmap='BuPu'
        elif (var == 'pv'):
                cs = pv0
                levs=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]; Cmap='twilight'
        elif (var == 'spd'):
                cs = (ua * ua + va * va) ** 0.5
                dl = 2.0; vmin = -20.0; vmax = -vmin + dl
                Cmap, norm, levs = normalise_cmap(vmin,vmax,0,dl)
        elif (var == 'w'):
                cs = wa
                dl = 5.0; wmin = -20.0; wmax = -wmin + dl
                Cmap, norm, levs = normalise_cmap(wmin,wmax,0,dl)
        else: # var == 'vort'
                vort = vort.extract(subset & ht_sub & xy_sub); vort = vort * 100000
                cs = vort; levs = np.arange(-10., 10., 1.0); Cmap = 'RdBu_r'

        # set up cross-section to cover small region
        cf = iplt.contourf(cs, coords=[xy_coord, 'level_height'], levels=levs, cmap=Cmap)
        #cf = iplt.contourf(cs, coords=[xy_coord, 'level_height'], cmap='BuPu')

        # overlay line contours of theta
        contours = iplt.contour(th0, coords=[xy_coord, 'level_height'], 
                                levels=np.arange(200, 440, 2), colors='black')
        plt.clabel(contours, inline=True, fontsize=8)

        '''
        # overlay selected vertical velocity contours 
        # NEED TO FIND WAY TO SMOOTH VERTICAL VELOCITY BEFORE THIS STEP CAN BE VALID 
        w_cont = iplt.contour(wa, coords=[xy_coord, 'level_height'], levels=[20,40], colors='gray')
        '''

        # add colourbar 
        plt.colorbar(cf, fraction=0.032, pad=0.06)

        # output the plot to X11 window 
        if (out == 'x11'):
                plt.show()
                exit()

        # produce output plot 
        if (md == 'sgt'):
                if (plane == 'lon'):
                        fili_cs = './{1}_sgt_{2}_cs_{4}{5}_{3}_T{0}.{6}'.format(Tp,res,sim,var,plane,ln,out)
                else:
                        fili_cs = './{1}_sgt_{2}_cs_{4}{5}_{3}_T{0}.{6}'.format(Tp,res,sim,var,plane,lt,out)
        else:
                if (plane == 'lon'):
                        fili_cs = './{1}_metum_control_cs_{4}{5}_{3}_T{0}.{6}'.format(Tp,res,sim,
                                                                                      var,plane,ln,out)
                else:
                        fili_cs = './{1}_metum_control_cs_{4}{5}_{3}_T{0}.{6}'.format(Tp,res,sim,
                                                                                      var,plane,lt,out)
        fig.savefig(fili_cs)
        print('output file created. Moving onto next model level...')

if __name__ == '__main__':
        # execute only if run as a script 
        main()
