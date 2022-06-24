import iris
import numpy as np
import diagnostic_core as SGD
import diagnosticSGtools as SG
from diagnosticSGsetup import stash_codes, output_names


def diagnostic_interface(ddir, fcst, Tp):
	"""
    Reads in fcst initialization time as YYYYmmddTHHMMZ string and Tp as
    integer of forecast lead time.
    """

	print('\n======= Running diagnostic_interface for {0} T+{1}'.\
			  format(fcst, Tp))
	
	print('Loading...')
	variabledict={}
	for name in stash_codes:
		print('   {}'.format(name))
		fn = '{0}/IN_{1}_{2}_T{3:03d}.nc'.format(ddir, name, fcst, Tp)
		variabledict[name] = iris.load(fn)[0]

    # Extract and scale data for increments
	print('Extracting increments...')
	tsc = 3600.
	sw = variabledict['sw'].data / tsc
	lw = variabledict['lw'].data / tsc
	bl = variabledict['bl'].data / tsc
	PC2 = variabledict['pc2'].data / tsc
	lsp = variabledict['lsp'].data / tsc
	con = variabledict['con'].data / tsc
	cmtu = variabledict['cmtu'].data / tsc
	cmtv = variabledict['cmtv'].data / tsc
	gwt = variabledict['gwt'].data / tsc
	gwu = variabledict['gwu'].data / tsc
	gwv = variabledict['gwv'].data / tsc

	# Combine mixing coefficients
	km = variabledict['km'].data
	km1 = variabledict['km1'].data
	km2 = variabledict['km2'].data
	km = np.maximum(km, km1 + km2)
	kmt_cube = SG.newcube(km, variabledict['km'])
        # kh = variabledict['kh'].data
	# kh1 = variabledict['kh1'].data
	# kh2 = variabledict['kh2'].data
	# kh = np.maximum(kh, kh1 + kh2)

	# Save cube formats for later use
	exner=variabledict['exner'].data
	exner_cube = variabledict['exner']
	thetavd_cube = variabledict['thetavd']

        # Get information on dimension sizes from model 
	nx=exner.shape[2]
	ny=exner.shape[1]
	nlev=exner.shape[0]

	# Create lcl cube if not available from UM data
	# will also need to modify other statements 
	# lcl = SG.lcl(variabledict['rh'].data)
	# exner_slice = exner_cube.extract(iris.Constraint(model_level_number=1))
	# lcl_cube = SG.newcube2d(lcl,exner_slice)

	# Set switches for use of moist stability and artificial heating
	cloud = 0
	heating = 0
	if cloud > 0:
		lcl = variabledict['lcl'].data
		nx = sw.shape[2]
		ny = sw.shape[1]
		nlevels = sw.shape[0]
		tinc = sw + lw + gwt
		for j in range(0, ny):
			for i in range(0, nx):
				k = int(lcl[j, i])
				tinc[0:k-1, j, i] = tinc[0:k-1, j, i] + \
					(bl[0:k-1, j, i] + lsp[0:k-1, j, i] + PC2[0:k-1, j, i])
		uinc = gwu
		vinc = gwv
	else:
		#tinc = bl + sw + lw + PC2 + lsp + con + gwt
		tinc = np.zeros( (nlev, ny, nx) )
		uinc = cmtu + gwu
		vinc = cmtv + gwv

	# Convert increments to cubes
	uinc_cube = SG.newcube(uinc, exner_cube)
	vinc_cube = SG.newcube(vinc, exner_cube)
	tinc_cube = SG.newcube(tinc, exner_cube)

	# Run SG tool
	print('Calling diagnostic_core...')
	output = SGD.diagnostic_core(
		exner_cube, variabledict['theta'],
		variabledict['thetavd'], variabledict['rhod'],
		variabledict['q'], variabledict['rh'],
		kmt_cube, variabledict['dbdz'],
		tinc_cube, uinc_cube, vinc_cube,
		variabledict['lcl'], cloud, heating)

	# Save output (one file per variable)
	print('Saving output...')
	for cube, name in zip(output, output_names):
		fnout = '{0}/OUT_{1}_{2}_T{3:03d}.nc'.format(ddir, name, fcst, Tp)
		iris.save(cube, fnout)
