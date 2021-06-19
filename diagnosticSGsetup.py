import iris
from datetime import datetime, timedelta
import numpy as np
import glob

stash_codes = {
	# Instantaneous fields
	'exner'      : 'm01s00i255',
	'thetavd'    : 'm01s00i388',
	'rhod'       : 'm01s00i389',
	'theta'      : 'm01s00i004',
	'q'          : 'm01s00i010',
	'rh'         : 'm01s09i229',
	
	# 1 hour accumulations
	'sw'         : 'm01s01i181',
	'lw'         : 'm01s02i181',
	'bl'         : 'm01s03i181',
	'pc2'        : 'm01s04i141',
	'lsp'        : 'm01s04i181',
	'con'        : 'm01s05i181',
	'gwt'        : 'm01s06i181',
	'cmtu'       : 'm01s05i185',
	'cmtv'       : 'm01s05i186',
	'gwu'        : 'm01s06i185',
	'gwv'        : 'm01s06i186',
	
	# 1 hour averages
	'km'         : 'm01s03i503',
	'km1'        : 'm01s03i505',
	'km2'        : 'm01s03i507',
	#'cf'         : 'm01s09i231',
	'dbdz'       : 'm01s03i469',
	'lcl'        : 'm01s05i273',
	
	# Validation
	'u'          : 'm01s00i002',
	'v'          : 'm01s00i003',
	'w'          : 'm01s00i150',
	}

output_names = ['ug', 'vg', 'dpidt', 'dugdt', 'dvgdt', 'dthetavdt',
                'u', 'v', 'w', 'balheat1', 'balheat2',
                'ug_um', 'vg_um', 'blh']

def regrid(cube):
    '''
    interpolate to N80 grid using an area-weighted regrid method 
    (1) create new latitude/longitude arrays 
    (2) add coordinate system information from the existing cube
    (3) add bounds to the coordinates using the 'guess_bounds' method (need these to do area-weighted regridding)
    (4) create cube with the N80 lat/lon grids 
    '''
    newlons = np.linspace(0, 360, 361)[:-1] # newlons = np.linspace(0, 360, 161)[:-1]
    newlats = np.linspace(-90, 90, 361)[1::2] # newlats = np.linspace(-90, 90, 241)[1::2]
    cs = cube.coord_system()
    lonc = iris.coords.DimCoord(newlons, standard_name='longitude',
                                units='degrees_east', coord_system=cs)
    lonc.guess_bounds()
    latc = iris.coords.DimCoord(newlats, standard_name='latitude',
                                units='degrees_north', coord_system=cs)
    latc.guess_bounds()
    dc_and_d = [(lonc, 0), (latc, 1)]
    template_cube = iris.cube.Cube(np.zeros([len(newlons), len(newlats)]),
                                   dim_coords_and_dims=dc_and_d)
    print('Regridding {} (new_dx={}, new_dy={})...'.\
            format(cube.name(), newlons[1]-newlons[0], newlats[1]-newlats[0]))
    for coname in ['longitude', 'latitude']:
        if cube.coord(coname).bounds is None:
            cube.coord(coname).guess_bounds()
    return cube.regrid(template_cube, iris.analysis.AreaWeighted())

def diagnosticSGsetup(fnames, ddir, fcst, Tp, orog_fn=None):
    """
    """
    print('\n======= Running diagnosticSGsetup for {0} T+{1}'.\
            format(fcst, Tp))

    # Load all cubes for a single time
    print('Loading...')
    print('\n'.join(glob.glob(fnames)))
    VT = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=Tp))
    VTm1 = (datetime.strptime(fcst, '%Y%m%dT%H%MZ') + timedelta(hours=(Tp-1)))
    tcon = iris.Constraint(time=lambda t: t.point <= VT and t.point >= VTm1)
    if orog_fn:
      print('Using orography from:\n{}'.format(orog_fn))
      cubes = iris.load([fnames, orog_fn])
      cubes = cubes.extract(tcon)
    else:
      cubes = iris.load(fnames, constraints=tcon)
    
    '''
    Extract what's needed - load and regrid one at a time 
    Get memory for N768 data if we try and load all at once 
    '''
    print('Extracting...')
    variabledict = {}
    for varname, sc in stash_codes.items():
        print('   {}'.format(varname))
        variabledict[varname] = cubes.extract_strict(iris.AttributeConstraint(STASH=sc))

        '''
        variabledict[varname] = regrid(cube)
            cubes.extract_strict(iris.AttributeConstraint(STASH=sc))
        '''

    '''
    # Interpolate to N80 grid
    sample_points = [('longitude', np.linspace(0, 357.75, 160)),
                     ('latitude',  np.linspace(-89.25, 89.25, 120))]
    '''
    sample_points = [('longitude', np.linspace(0, 359., 360)),
                     ('latitude',  np.linspace(-89.5, 89.5, 180))]
    print('Regridding (new_dx={}, new_dy={})...'.\
            format(sample_points[0][1][1]-sample_points[0][1][0],
                   sample_points[1][1][1]-sample_points[1][1][0]))
    for varname in stash_codes:
        print('   {}'.format(varname))
        variabledict[varname] = variabledict[varname].\
			interpolate(sample_points, iris.analysis.Linear())

    # Save (one file per variable)
    print('Saving...')
    for varname, sc in stash_codes.items():
        fnout = '{0}/IN_{1}_{2}_T{3:03d}.nc'.\
            format(ddir, varname, fcst, Tp)
        print('   {}'.format(fnout.split('/')[-1]))
        iris.save(variabledict[varname], fnout)


