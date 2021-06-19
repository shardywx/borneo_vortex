import sys
from hov_plotter import hov_plotter
'''
run from command line using: python run_hov.py oct um u control png
sys.argv[1] = December ('dec') or October ('oct') case study
sys.argv[2] = model ('sgt' or 'um')
sys.argv[3] = diagnostic for filled contour plot ('u' or 'v')
sys.argv[4] = forcing terms on RHS of SGT model ('diab', 'geo', 'control')
sys.argv[5] = output file type (png, pdf, ...)
'''

# assign variable names to user arguments 
mn = sys.argv[1]; md = sys.argv[2]

# print information to screen
if mn == 'oct':
        print('Analysing October 2018 case...')
        init = '20181021T1200Z'
else:
        print('Analysing December 2018 case...')
        init = '20181211T1200Z'
res = 'N768'

# which model to analyse in 'diagnostic_plotter'
if md == 'sgt':
        sim = sys.argv[4]
else:
        sim = 'control'

# choose variable for filled contour plot 
var  = sys.argv[3]

# output file type (for final plot)
out  = sys.argv[5]

# set up directories
ddir = '/nobackup/earshar/borneo'
ddir_in = '{0}/case_{2}_{1}'.format(ddir,res,init)

# make plots 
hov_plotter(ddir, sim, md, var, out, mn)
