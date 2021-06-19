# deleted code from other scripts 

import numpy as np
import math 
import iris 
import pandas as pd 

'''                                                                                           
# from prcp_plotter.py --> converting Pandas DataFrame to Iris cube before overlaying on contour plot 
bv_lat = ipd.as_cube(bv_lat); bv_lon = ipd.as_cube(bv_lon)
bv_lat.rename('latitude'); bv_lat.units = 'degrees'
bv_lon.rename('longitude'); bv_lon.units = 'degrees'
'''

'''
# had trouble reading in N768 precipitation data using Iris --> now using Xarray 
nfili = '{0}/case_{3}_{2}/umglaa_pa{1:03d}'.format(ddir, Tp-12, res, fcst)
ncubes = iris.load(nfili); prcp_gl = ncubes.extract('stratiform_rainfall_flux')
#prcp_gl = prcp_gl * 3600
print(prcp_gl.shape)
exit()
'''

# from plot_4p4.py

print(gdata)
exit()
dims = gdata.dims
u = gdata.u
print(u)
exit()

"""                                                                                  
ds_fix = xr.Dataset(coords={'z': np.arange(dims['hybrid_ht']),                       
'y': np.arange(dims['latitude']),                        
'x': np.arange(dims['longitude']),                       
't': np.arange(dims['t'])})                              
"""

print(ds_fix)
exit()
