'''
quickplot of a 3D cube on a map
run using python plot_slp.py 2018 10 21 12 png (only one possible set of values for now)
sys.argv[1]: year (2018)
sys.argv[2]: month (10)
sys.argv[3]: day (21)
sys.argv[4]: initialisation time (12)
sys.argv[5]: output file format (png, pdf, ...)  
'''

from datetime import datetime, timedelta
import tc_functions as tc
import plotting as plot 

import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import iris 
import iris.plot as iplt 
import iris.quickplot as qplt
import iris.coords as icoords
import numpy as np 

import matplotlib 
matplotlib.use('TkAgg')

def main():

    # define file paths 
    yr = int(sys.argv[1]); mn = int(sys.argv[2]); dy = int(sys.argv[3]); init = sys.argv[4]
    bname = '/nobackup/earshar/borneo/SEA4_km4p4_ra1t/{0}{1}{2}T{3}00Z_SEA4_km4p4_ra1tld_pverb.pp'.format(yr, mn, dy, init)
    dname = '/nobackup/earshar/borneo/SEA4_km4p4_ra1t/{0}{1}{2}T{3}00Z_SEA4_km4p4_ra1tld_pverd.pp'.format(yr, mn, dy, init)

    # define grid subset
#    lon0 = 94; lon1 = 146; lat0 = -16; lat1 = 21
    lon0 = 94; lon1 = 116; lat0 = -4; lat1 = 12
    subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

    # read in data 
    cubes_sfc = iris.load(bname,subset); cubes_p = iris.load(dname,subset)

    # specific variables 
    u0 = cubes_p.extract_strict('x_wind'); v0 = cubes_p.extract_strict('y_wind')
    z0 = cubes_p.extract_strict('geopotential_height')
    w0 = cubes_p.extract_strict('lagrangian_tendency_of_air_pressure')

    # define start/end time indices (T+24 --> T+36 for now)
    times = np.arange(8,13)

    # loop over times 
    for i, it in enumerate(times):

        # specify pressure level and time 
        u = u0[it,7,:,:]; v = v0[it,7,:,:]; z = z0[it-1,7,:,:]
        w = w0[it-1,7,:,:] 

        # convert units of geopotential height 
        z.convert_units('dam')

        # extract forecast initialisation time for output file string
        # https://github.com/johnash92/pv_tendency/blob/d5b3c4da9feb1652565cf560ac1f7ef07271d398/test_metrics.py   
        tc0 = u.coord('forecast_reference_time')
        hh  = tc0.units.num2date(tc0.points)[0].hour
        dd  = tc0.units.num2date(tc0.points)[0].day
        mm  = tc0.units.num2date(tc0.points)[0].month
        yy  = tc0.units.num2date(tc0.points)[0].year 

        # extract forecast reference time (T+...)
        tt = int(u.coord('forecast_period').points)
        print('Working on time: T+{0}'.format(tt))

        # calculate vector wind speed 
        spd = (u * u + v * v) ** 0.5
        spd.name('horizontal_windspeed')

        # quick plot 
        dstr = '{0}{1:02d}{2:02d}T{3:02d}00Z'.format(yy,mm,dd,hh)
        fili = '/nobackup/earshar/borneo/trunk_BJH/700_{0}_T{2:02d}.{1}'.format(dstr, sys.argv[5], tt)
        plot.qplot(z,w,fili,x11=0)

if __name__ == '__main__':
    main()
