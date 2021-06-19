'''
quickplot of a 3D cube on a map
run using python plot_rain.py 2018 10 21 12 png (only one possible set of values for now)
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

    # define grid subset
    lon0 = 94; lon1 = 116; lat0 = -4; lat1 = 12
    subset = iris.Constraint(latitude=lambda z: lat0<z<lat1,longitude=lambda z: lon0<z<lon1)

    # read in data 
    cubes = iris.load(bname,subset)

    # specific variables (customise method of reading in data)
    slp = cubes[0][:,:,:] #.extract_strict('air_pressure_at_sea_level')
    rain = cubes[5][:,:,:] #.extract_strict('stratiform_rainfall_flux')

    # define start/end time indices (T+24 --> T+36 for now)
    times = np.arange(24,36)

    # loop over times 
    for i, it in enumerate(times):

        # read in data 
        slp = cubes[0][it,:,:]
        rain = cubes[5][it,:,:]

        print(slp)
        print(rain)
        exit()

        # EDIT FROM HERE (rain valid at HH:30; pmsl valid at HH:00)
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
        fili = '/nobackup/earshar/borneo/trunk_BJH/850_{0}_T{2:02d}.{1}'.format(dstr, sys.argv[5], tt)
        plot.qplot(z,w,fili,x11=0)

if __name__ == '__main__':
    main()
