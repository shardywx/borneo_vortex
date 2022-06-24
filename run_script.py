import sys
import glob
from diagnosticSGsetup import diagnosticSGsetup
from diagnostic_interface import diagnostic_interface
from diagnostic_plotter import diagnostic_plotter
from cs_plotter import cs_plotter
from hov_plotter import hov_plotter
from prcp_plotter import prcp_plotter

'''
run from command line using: python run_script.py 2018 dec um full u 36 control png N768 plt ext xy
sys.argv[1] = case study (2016 = NAWDEX; 2018 = FORSEA BV case study)
sys.argv[2] = December ('dec') or October ('oct') case study
sys.argv[3] = model ('sgt' or 'um')
sys.argv[4] = horizontal wind ('full', 'geo' or 'ageo')
sys.argv[5] = diagnostic for filled contour plot ('w', 'vort', 'u' or 'v')
sys.argv[6] = analysis time (T+... '24', '36', '48', ...)
sys.argv[7] = forcing terms on RHS of SGT model ('diab', 'geo', 'control')
sys.argv[8] = output file type (png, pdf, ...)
sys.argv[9] = horizontal resolution of input MetUM forecast data (N768 or N96)
sys.argv[10] = run main code (core), or just produce plots (plt) 
sys.argv[11] = zoom into smaller region when plotting (e.g. for detailed vortex structure)
'''

# Decide which case to look at 
if sys.argv[1] == '2016': 
        print('Analysing 2016 case...')
        init = '20160915T1200Z'
        res = 'nawdex'
else:
        if sys.argv[2] == 'oct':
                print('Analysing October 2018 case...')
                init = '20181021T1200Z'
        else:
                print('Analysing December 2018 case...')
                init = '20181211T1200Z'
        res = sys.argv[9]

# which model to analyse in 'diagnostic_plotter'
if sys.argv[3] == 'sgt':
        sim = sys.argv[7]
else:
        sim = 'control'

# month string 
mn = sys.argv[2]

# model string 
md = sys.argv[3]

# whether to plot the total, geostrophic or ageostrophic wind 
wind = sys.argv[4]

# choose variable for filled contour plot 
var  = sys.argv[5]

# output file type (for final plot)
out  = sys.argv[8]

# optional argument for diagnostic_plotter (whether to zoom into small region)
size = sys.argv[11]

# produce x-y or x-z plot in diagnostic_plotter
plane = sys.argv[12]

# add input options when running these scripts 
fcsts={
	init : [int(sys.argv[6])],
	}

for fcst in fcsts:
    # set up directories
        ddir = '/nobackup/earshar/borneo'
        ddir_in = '{0}/case_{2}_{1}'.format(ddir,res,init)
        if sys.argv[10] == 'plt':
                ddir_out = '{0}/SGTool/{1}/{2}/{3}/filter_3_6/conv_g7x'.format(ddir,res,mn,sim)
        else:
                ddir_out = '{0}/SGTool/{1}/{2}/{3}'.format(ddir,res,mn,sim)

        # loop over fcst leadtime
        for Tp in fcsts[fcst]:

                # run diagnosticSGsetup to extract required variables ('pb' to 'pe' stream)   
                fnames = '{0}/umglaa_p[!a]{1:03d}'.format(ddir_in, Tp-12)
                # also extract 'pa' stream data                                             
                orog_fn = '{0}/umglaa_pa{1:03d}'.format(ddir_in, 0)
                # run the diagnostic core
                if sys.argv[10] == 'core': 
                        # call 'diagnosticSGsetup'                                                  
                        diagnosticSGsetup(fnames, ddir_out, fcst, Tp, orog_fn=orog_fn)
                        # run diagnostic_interface                                                  
                        diagnostic_interface(ddir_out, fcst, Tp)
                # or produce output plots
                else:
                        # make some plots 
                        if var == 'prcp':
                                prcp_plotter(ddir, fcst, Tp, res, sim, md, wind, 
                                             var, out, size, plane, mn)
                        else:
                                if plane == 'xy':
                                        diagnostic_plotter(ddir, fcst, Tp, res, sim, md, wind, 
                                                           var, out, size, plane, mn)
                                elif plane == 'xz':
                                        cs_plotter(ddir, fcst, Tp, res, sim, md, wind, 
                                                   var, out, size, plane, mn)
                                else:
                                        hov_plotter(ddir, fcst, Tp, res, sim, init, md, wind, 
                                                    var, out, size, plane, mn)
