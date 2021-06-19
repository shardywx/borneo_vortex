import numpy as np
import sys
import iris
import pandas as pd
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.maths as imath
import iris.pandas as ipd
import xarray as xr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

def prcp_plotter(ddir, fcst, Tp, res, sim, md, wind, var, out, size, plane, mn):

        print('\n======= Running prcp_plotter for {0} T+{1}'.format(fcst, Tp))

        # define grid subset to reduce memory required to run script 
        if fcst == '20181021T1200Z' or fcst == '20181211T1200Z':
                if (size == 'zoom'):
                        lon0 = 101; lon1 = 119; lat0 = 1; lat1 = 14;
                elif (size == 'ext'):
                        lon0 = 76; lon1 = 160; lat0 = -12; lat1 = 50;
                else: # reg
                        lon0 = 96; lon1 = 124; lat0 = -4; lat1 = 19;

	# use shorthand notation to define this region
	# use of 'subset' below was returning '<No cubes>' errors, 
        # b/c lat/lon names were incorrect
        subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

	# loop over vertical levels
        levs = [12]
        for lev in levs:

                # read in GPM-IMERG precipitation data (30 min interval)
                pdiri  = '/nobackup/earshar/borneo/'
                pfili  = '{0}/GPMHH_201810.nc'.format(pdiri)
                pcubes = iris.load(pfili); gpm_prcp = pcubes.extract_strict('precipitationCal')

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
                elif Tp == 99:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=13, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=17, minute=00)
                elif Tp == 102:
                        pdt1 = PartialDateTime(year=2018, month=10, day=25, hour=16, minute=30)
                        pdt2 = PartialDateTime(year=2018, month=10, day=25, hour=20, minute=00)

                # calculate accumulated precipitation
                if md == 'gpm':
                        acc_time = iris.Constraint(time=lambda t: pdt1 <= t.point < pdt2)
                        gpm_prcp = gpm_prcp.extract(acc_time)
                        gpm_prcp = gpm_prcp.collapsed('time', iris.analysis.SUM)

                # read in precipitation data from 4.4 km forecast
                pdiri  = '/nobackup/earshar/borneo/'
                pfili  = '{0}/20181021T1200Z_SEA4_km4p4_ra1tld_pverb.pp'.format(pdiri)
                pcubes = iris.load(pfili)
                prcp   = pcubes.extract('stratiform_rainfall_flux')[1]
                # calculate units, using frequency of input data (either every 1 h or 15 min)
                ntimes = prcp.shape[0]
                if ntimes == 120:
                        prcp = prcp * 3600
                else:
                        prcp = prcp * 900
                prcp.units = 'mm hr**-1'

                # focus on subset of times  
                if Tp != 120:
                        um_time = iris.Constraint(time = lambda t0: pdt1 <= t0.point <= pdt2)
                        prcp = prcp.extract(um_time) 

                # calculate accumulated precipitation 
                prcp = prcp.collapsed('time', iris.analysis.SUM)
                prcp.rename('accumulated rainfall'); prcp.units = 'mm'

                # read in global precipitation data from N768 forecast using xarray
                if md == 'um_gl':
                        nfili = '{0}/case_{3}_{2}/umglaa_pa{1:03d}.nc'.format(ddir, Tp-12, res, fcst)
                        data = xr.open_dataset(nfili)
                        prcp_ls = data["lsrain_1"].squeeze('t_2').squeeze('surface')
                        prcp_cv = data["cvrain_1"].squeeze('t_2').squeeze('surface')
                        # calculate total precipitation (large-scale + convective)
                        prcp_gl = (prcp_ls + prcp_cv) * 10800; prcp_gl.attrs['units'] = 'mm hr**-1'
                        
                # read in Borneo vortex track data from text file (Hodges)
                df = pd.read_csv('/nobackup/earshar/borneo/bv_2018102112_track.csv',
                                 na_filter=True,na_values="1.000000e+25")
                # convert time integers to datetime objects 
                df['Time'] = pd.to_datetime(df['Time'], format='%Y%m%d%H')
                # extract track information between 12Z on 21st and 26th October
                bv_lat = df.loc[0:20, "lat_vort"]
                bv_lon = df.loc[0:20, "lon_vort"]
                bv_time = df.loc[0:20, "Time"]

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
                # date string for plot label
                VT = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=Tp))
                if md == 'mean' or md == 'um_gl':
                        ax.set_title('', loc='right')
                else:
                        ax.set_title(VT.strftime('%HZ %a %d %b %Y [T+{}]'.format(Tp)), loc='right')


                # plot precipitation data (filled contours)
                print('Starting contour plotting...')

                # define levels
                if Tp == 120:
                        Levels=[2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0, 
                                200.0, 250.0, 300.0, 400.0, 500.0, 600.0]
                else:
                        Levels=[1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
                
                # plot data 
                if md == 'um':
                        cf = iplt.contourf(prcp, axes=ax, 
                                           levels=Levels,cmap=cmocean.cm.haline_r)
                elif md == 'um_gl':
                        prcp_gl.plot.contourf(ax=ax, levels=Levels, extend='max', 
                                              transform=ccrs.PlateCarree(),
                                              cbar_kwargs={'label': prcp_gl.units},
                                              cmap=cmocean.cm.haline_r)
                else: # GPM-IMERG
                        cf = iplt.contourf(gpm_prcp, axes=ax, 
                                           levels=Levels,cmap=cmocean.cm.haline_r)
                print('Finished contour plotting...')                        

                # overlay Borneo vortex track (ERA-Interim)
                if Tp == 120:
                        ax.plot(bv_lon, bv_lat, 'bo--', markersize=3)
                else:
                        ii = int(Tp/6)
                        ax.plot(bv_lon[ii], bv_lat[ii], 'bD--', markersize=5)

                # add colourbar
                if md != 'um_gl':
                        if size == 'reg':
                                plt.colorbar(cf, fraction=0.025, pad=0.06, label=prcp.units)
                        else:
                                plt.colorbar(cf, fraction=0.032, pad=0.06, label=prcp.units)
                        

                # output the plot to X11 window
                if (out == 'x11'):
                        plt.show()
                        exit()

                # produce output plot 
                if md == 'um':
                        mstr='4p4'
                elif md == 'um_gl':
                        mstr='N768'

                if md == 'um' or md == 'um_gl':
                        if (size == 'zoom'):
                                fili='./{1}_metum_control_{4}_{3}_{6}_T{0}.{5}'.format(Tp,mstr,
                                                                                       sim,var,
                                                                                       wind,out,
                                                                                       size)
                        else:
                                fili='./{1}_metum_control_{4}_{3}_T{0}.{5}'.format(Tp,mstr,
                                                                                   sim,var,
                                                                                   wind,out)
                else: # md == 'gpm'
                        fili='./{1}_gpm_{4}_{3}_T{0}.{5}'.format(Tp,res,sim,
                                                                 var,wind,out)

                                                                
                fig.savefig(fili)
                                                                
                print('output file created. Moving onto next output time...')

if __name__ == '__main__':
        # execute only if run as a script 
        main()
