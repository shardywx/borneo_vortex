######################################################################
#  Generic functions to customise colourmaps for plotting.           #
#                                       John Ashcroft, November 2016 #
######################################################################

#### Contents:   #####
#
## normalise_cmap()
# inputs a max, min, midpoint and spacing (dl). The colourmap is then
# centred on the midpoint. Used in particular for diverging colourmaps
# so that the white colour is equal to the value of 0.
# Returns cmap, norm and levels (i.e. an example of the plot after using
# function: ax.contourf(X,Y,DATA,norm=norm,levels=levels,cmap=cmap)
#
## OOMFormatter()
# Used in scripts to format how scientific numbers are represented. 
# Ensures that everything is neat and tidy with the correct power abover
# the colourbar. For example, see vorticity.py
#
## highResPoints()
# Increases the amount of data points between two points. i.e. inserts 
# 10* the number of points in an array ([0,1] --> [0,0.1,0.2,...,1]
# Used to colour lines in a scatter plot using a cmap. 

import os
from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker

def normalise_cmap(vmin, vmax, midpoint, dl):
    levels=np.arange(vmin,vmax,dl)
    midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
    vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
    colors = plt.cm.bwr(vals)
    cmap, norm = from_levels_and_colors(levels, colors)
    return cmap, norm, levels

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)


def highResPoints(x,y,factor=10):
    '''
    Take points listed in two vectors and return them at a higher
    resolution. Create at least factor*len(x) new points that include the
    original points and those spaced in between.
    Returns new x and y arrays as a tuple (x,y).
    '''
    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    dr = rtot[-1]/(NPOINTS*RESFACT-1)
    xmod=[x[0]]
    ymod=[y[0]]
    rPos = 0 # current point on walk along data
    rcount = 1 
    while rPos < r.sum():
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        dpos = rPos-rtot[rcount] 
        theta = np.arctan2((x2-x1),(y2-y1))
        rx = np.sin(theta)*dpos+x1
        ry = np.cos(theta)*dpos+y1
        xmod.append(rx)
        ymod.append(ry)
        rPos+=dr
        while rPos > rtot[rcount+1]:
            rPos = rtot[rcount+1]
            rcount+=1
            if rcount>rtot[-1]:
                break

    return xmod,ymod
