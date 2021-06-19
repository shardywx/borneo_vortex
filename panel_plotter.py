import numpy as np
import sys
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.maths as imath
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
from diagnosticSGfunctions import calc_f, exner_to_pres, geo_wind, reverse_lat, order_lat
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

def panel_plotter(ddir, pdir, fcst, Tp, res, sim, init, md, wind, var, out):

        print('\n======= Running panel_plotter for {0} T+{1}'.format(fcst, Tp))

        # define variable dictionary (data structure that maps one value to another)
	# print('Loading...') --> loading all output variables from the SGT model 
        variabledict={}
        for name in output_names:  # import this information from 'diagnosticSGsetup'
                print('   {}'.format(name))
                fn = '{0}/SGTool/{4}/{5}/OUT_{1}_{2}_T{3:03d}.nc'.format(ddir,name,fcst,
                                                                         Tp,res,sim)
                variabledict[name] = iris.load(fn)[0]
                variabledict[name].rename(name)
                
        # define grid subset to reduce memory required to run script 
        if (init == '20181021T1200Z'):
                lon0 = 94; lon1 = 146; lat0 = -16; lat1 = 21;
                lon0 = -60; lon1 = 0; lat0 = 30; lat1 = 70;
        else:
                lon0 = -60; lon1 = 0; lat0 = 30; lat1 = 70; 
                lon0 = 94; lon1 = 146; lat0 = -16; lat1 = 21;

	# use shorthand notation to define this region
	# use of 'subset' below was returning '<No cubes>' errors, b/c lat/lon names were incorrect
        subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

        # read in PV from MetUM files ('pe' stream)
        fnames = '{0}/case_{3}_{2}/umglaa_pe{1:03d}'.format(ddir, Tp+12, res, init)
        cubes = iris.load(fnames)
        pv = cubes.extract_strict('potential_vorticity_of_atmosphere_layer')
        pv.rename('PV')
        pv.units = '1e6 PVU'
        pv.convert_units('PVU')

        # also read in horizontal and vertical wind components from MetUM files 
        wt = cubes.extract_strict('upward_air_velocity')
        ut = cubes.extract_strict('x_wind')
        vt = cubes.extract_strict('y_wind')

        wt.rename('vertical velocity')
        ut.rename('zonal wind component')
        vt.rename('meridional wind component')

        # read in data from other data streams ('pa')
        anames = '{0}/case_{3}_{2}/umglaa_pa{1:03d}'.format(ddir, Tp-12, res, init)
        cubes  = iris.load(anames)
        pmsl   = cubes.extract_strict('air_pressure_at_sea_level')

	# also read in 'pb' stream data 
        bnames = '{0}/case_{3}_{2}/umglaa_pb{1:03d}'.format(ddir, Tp-12, res, init)
        cubes  = iris.load(bnames)
        exn = cubes.extract_strict('dimensionless_exner_function')
        rho = cubes.extract_strict('air_density')

	# regrid 'u' onto 'v' grid 
        ut = ut.regrid(vt,iris.analysis.Linear())
        rho = rho.regrid(vt,iris.analysis.Linear())

	# plot filled contours (w, vort, ...)
        lev = 30

        # extract lat/lon information from diagnostic array
        xlon = ut[0].coord('longitude')
        ylat = ut[0].coord('latitude')

        # read in 3D wind field from SGT model (on single model level)
        w = variabledict['w'][lev]
        v = variabledict['v'][lev]
        u = variabledict['u'][lev]

        # open figure
        plt.figure(figsize=(9, 6), dpi=100)

        # add options for vorticity on pressure levels (filled contours)
        vrtmax = 30e-5
        vrtmin = -30e-5
        dl = 3e-5
        cmap, norm, vortLevels = normalise_cmap(vrtmin,vrtmax,0,dl)

        # reduce the number of grid points before plotting 
        skip = 2
        ua = ut[lev][::skip, ::skip]
        va = vt[lev][::skip, ::skip]
        wa = wt[lev][::skip, ::skip]
        
        # calculate pressure from Exner pressure   
        '''
        (1) https://github.com/Ba-So/coarse_grain/blob/
        # 0f4f41b694c5bcaa289b98852bde701005b7679a/modules/phys_mod.py
        (2) http://esc24.github.io/iris/examples/graphics/deriving_phenomena.html
        '''
        pres = exner_to_pres(exn[lev])
        
        # calculate Coriolis parameter and pressure gradient
        if (var == 'f') or (var == 'geo') or (var == 'pgf'):
                V  = VectorWind(ut[lev], vt[lev])
                pres = pres.regrid(ut[lev],iris.analysis.Linear())
                px, py = V.gradient(pres,truncation=21)

        # calculate geostrophic wind 

        # set constants
        r_e = 6371000.
        pi = 3.14159265
        # get grid in radians
        xn = pres.coord('longitude').points * (pi / 180.)
        yt = pres.coord('latitude').points * (pi / 180.)
        # calculate spacing in radians between grid points
        nx = pres.shape[1]
        ny = pres.shape[0]
        dlon = 4. * pi / nx
        dlat = 2. * pi / ny
        # calculate cosine of latitude
        coslat = np.cos(yt)
        coslat = coslat.reshape(1, ny, 1)
        # call function to calculate gradient 
        px0 = SG.ddlambda_2d(pres.data, r_e, dlon, coslat)
        py0 = SG.ddphi_2d(pres.data, r_e, dlat)
        # reshape arrays
        px0 = px0.reshape(ny,nx)
        py0 = py0.reshape(ny,nx)
        # create new Iris cubes containing pressure gradient data
        # https://groups.google.com/forum/#!msg/scitools-iris/4cTGbiPHPPY/jEUCqkOk5h4J;context-place=forum/scitools-iris
        pgf_u = pres.copy(data=px0)
        pgf_v = pres.copy(data=py0)
        # calculate Coriolis term in geostrophic wind components
        f0    = calc_f(ut[lev])
        f     = ut[lev].copy(data=f0)
        f2    = imath.exponentiate(f,2)
        cor_v = imath.divide(f,f2*rho[lev])
        cor_u = imath.multiply(imath.divide(f,f2*rho[lev]),-1)
        # calculate geostrophic wind components
        ug = pgf_u * cor_u
        vg = pgf_v * cor_v
        # add metadata (units)
        ug.units = 'm s**-1'
        vg.units = 'm s**-1'

        #VT = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=Tp))
        #ax.set_title(VT.strftime('%HZ %a %d %b %Y [T+{}]'.format(Tp)), loc='right')
                        
        # panel plot of chosen terms/diagnostics
        print('starting contour plotting...')
                        
        # first panel (create subplot; domain extent; gridlines)
        ax1 = plt.subplot(2,3,1, projection=ccrs.PlateCarree())
        ax1.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':') 
        # set contour levels
        #contour_levels = np.linspace(np.min(py.data),np.max(py.data),20)
        contour_levels = np.linspace(-0.01, 0.01, 21)
        # filled contour plot 
        cf = iplt.contourf(pgf_u, levels=contour_levels, cmap='seismic')
        # add coastlines (get the current Axes instance matching the given keyword arguments)
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)



        # second panel (create subplot; domain extent; gridlines)
        ax2 = plt.subplot(2,3,4, projection=ccrs.PlateCarree())
        ax2.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
        # set contour levels
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)
        contour_levels = np.linspace(-0.01, 0.01, 21)
        cf = iplt.contourf(pgf_v, levels=contour_levels, cmap='seismic')



        # make an axes to put shared colourbar in ('gcf' = get the current figure) 
        colorbar_axes = plt.gcf().add_axes([0.13, 0.06, 0.21, 0.03])
        colorbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal', extend='both', extendrect=False)
        colorbar.set_label('%s' % '10$^{-3}$ Pa m$^{-1}$')
        # limit the colourbar to a set number of tickmarks
        import matplotlib.ticker
        colorbar.locator = matplotlib.ticker.MaxNLocator(11)


        # change units before plotting 
        cor_u = cor_u / 10000.
        cor_v = cor_v / 10000.

        # third panel (create subplot; domain extent; gridlines)
        ax3 = plt.subplot(2,3,2, projection=ccrs.PlateCarree())
        ax3.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax3.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
        # set contour levels
        contour_levels = np.linspace(-50.,50.,21)
        cf = iplt.contourf(cor_u, levels=contour_levels, cmap='seismic')
        # add coastlines
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)


        
        # fourth panel (create subplot; domain extent; gridlines)
        ax4 = plt.subplot(2,3,5, projection=ccrs.PlateCarree())
        ax4.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax4.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
        # set contour levels 
        contour_levels = np.linspace(-50.,50.,21)
        cf = iplt.contourf(cor_v, levels=contour_levels, cmap='seismic')
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)


        
        # add second colourbar 
        colorbar_axes = plt.gcf().add_axes([0.40, 0.06, 0.21, 0.03])
        colorbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal', extend='both', extendrect=False)
        colorbar.set_label('%s' % '10$^{-3}$ Pa m$^{-1}$')
        # limit the colourbar to a set number of tickmarks  
        import matplotlib.ticker
        colorbar.locator = matplotlib.ticker.MaxNLocator(11)



        # fifth panel
        ax3 = plt.subplot(2,3,3, projection=ccrs.PlateCarree())
        ax3.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax3.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
        # set contour levels 
        contour_levels = np.linspace(-100., 100., 21)
        cf = iplt.contourf(ug, levels=contour_levels, cmap='seismic')
        # geostrophic wind vectors
        iplt.quiver(vg[::skip, ::skip], ug[::skip, ::skip], angles='xy', scale=800)
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)



        # sixth panel
        ax4 = plt.subplot(2,3,6, projection=ccrs.PlateCarree())
        ax4.set_extent([lon0-1, lon1+1, lat0-1, lat1+1], crs=ccrs.PlateCarree())
        ax4.gridlines(crs=ccrs.PlateCarree(), linewidth=0.75, color='k', linestyle=':')
        # set contour levels 
        contour_levels = np.linspace(-100., 100., 21)
        cf = iplt.contourf(vg, levels=contour_levels, cmap='seismic')
        iplt.quiver(vg[::skip, ::skip], ug[::skip, ::skip], angles='xy', scale=800)
        plt.gca().coastlines(resolution='10m', color='k', linewidth=1)



        # colourbar for geostrophic wind 
        colorbar_axes = plt.gcf().add_axes([0.68, 0.06, 0.21, 0.03])
        colorbar = plt.colorbar(cf, colorbar_axes, orientation='horizontal', extend='both', extendrect=False)#, format='%.1e')
        colorbar.set_label('%s' % 'm s$^{-1}$')
        # limit the colourbar to a set number of tickmarks 
        import matplotlib.ticker
        colorbar.locator = matplotlib.ticker.MaxNLocator(11)

        # output the plot to X11 window
        print('finished contour plotting...')
        plt.show()

if __name__ == '__main__':
        # execute only if run as a script 
        main()
