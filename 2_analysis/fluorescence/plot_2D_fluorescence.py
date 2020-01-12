import numpy as np
import scipy.interpolate
import os
import math
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import rcParams
'''
This plots the 2D fluorescence spectrum from data we computed in
compute-fluorescence.py that should have been dumped to
./data/fluorescence.pickle
This uses a fluorescence shift that should be computed in the
output of plot-time-resolved.py and saved to ./data/fl_shift.txt
'''

def interpolate_fluorescence(tgrid, wgrid, fgrid, ngrid=1000):
    ''' 
    Use cubic interpolation to place the computed fluorescence grid on a finer mesh.
    ngrid is the grid size for the new interpolated grid.
    '''
    # Unravel the fluorescence grid 
    ts = []
    ws = []
    fs = []
    for i, t in enumerate(tgrid):
        for j, w in enumerate(wgrid):
            ts.append(t)
            ws.append(w)
            fs.append(fgrid[i,j])
    # Expand the time and wavelength grids
    tgrid2 = np.linspace(tgrid[0], tgrid[-1], ngrid)
    wgrid2 = np.linspace(wgrid[0], wgrid[-1], ngrid)
    # Place the expanded grids on a mesh.
    tgrid3, wgrid3 = np.meshgrid(tgrid2, wgrid2)
    # Interpolate the fluorescence grid onto the mesh.
    fgrid3 = scipy.interpolate.griddata( (ts, ws), fs, (tgrid3, wgrid3), method='cubic', fill_value=0)
    # Normalize the interpolated grid.
    fgrid3 = fgrid3 / np.max(fgrid3)

    return tgrid3, wgrid3, fgrid3

def plot_fluorescence(fgrid, wgrid, tgrid, figname='2D_fluorescence', fl_shift=0, lineout_wls=[]):
    '''
    Here, we think about time as x and energy as y. It turns out that in
    the fluorescence 2D array, each row corresponds to a specific time 
    value, so the first index (rows) gives the time and the second 
    (column) index is energy. The energy grid is interchangeable with the
    wavelength grid. 
    First, we place the fluorescence grid on a finer mesh and then we plot
    the 2D spectrum as a contour plot. Then we plot it with a custom 
    color scheme. 
    '''

    if fl_shift > 0:
        # Keep in mind that you're going to want to manually set the plotting ranges
        # for the wavelength if you use the energy shift.
        print('Shifting the theory spectrum to match experiment.')
        egrid = [(1240./x - fl_shift) for x in wgrid]
        wgrid = [int(1240./x) for x in egrid]

    print('Interpolating the fluorescence grid.')
    tgrid3, wgrid3, fgrid3 = interpolate_fluorescence(tgrid, wgrid, fgrid)

    print('Plotting the fluorescence grid.')
    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    # Construct a custom truncated jet colormap.
    ngrid_cm = 51
    colors = [ (40, 80, 250), (15, 160, 240), (25, 200, 225), (100, 250, 190), (180, 240, 150), (220, 210, 120), (255, 165, 90), (255, 100, 50), (255, 0, 0) ]
    colors = [ (x[0]/255., x[1]/255., x[2]/255.) for x in colors]
    cmap_name = 'special'
    cm = cl.LinearSegmentedColormap.from_list(cmap_name, colors, N=ngrid_cm)
    # cm = plt.get_cmap('jet') # if you want to use the standard jet colormap.
    levels = np.linspace(np.min(fgrid3), np.max(fgrid3), ngrid_cm)
    fl_im = plt.contourf(tgrid3, wgrid3, fgrid3, cmap=cm, levels=levels, vmin=np.min(fgrid3), vmax=np.max(fgrid3))
    cb = fig.colorbar(fl_im, format='%.1f')
    cb.set_ticks( [ x for x in np.arange( np.min(fgrid3), np.max(fgrid3)+0.1, 0.1) ] )
    # cb.set_label('Fluorescence Intensity', rotation=270, fontsize=labelsize)

    # Plot lineouts if requested by giving a list of lineout wavelengths
    if len(lineout_wls) > 0:
        for wl in lineout_wls:
            plt.plot([tgrid[0], tgrid[-1]], [wl]*2, linewidth=1.5, linestyle='--', color='silver')

    plt.xlabel('Time [fs]',fontsize=labelsize)
    plt.ylabel('Wavelength [nm]',fontsize=labelsize)
    plt.tight_layout()

    print('Saving the fluorescence figure to ./figures/%s.pdf' %figname)
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/%s.pdf' %figname, dpi=300)
    plt.close()

if os.path.isfile('./data/fl_shift.txt'):
    fl_shift = np.loadtxt('./data/fl_shift.txt')
else: fl_shift = 0
fl_data = pickle.load(open('./data/fluorescence.pickle', 'rb'))
wgrid = fl_data['wgrid']
tgrid = fl_data['tgrid']
fl    = fl_data['fluorescence']
figname = '2D_fluorescence'
lineout_wls = [400, 600]
plot_fluorescence(fl, wgrid, tgrid, figname=figname, fl_shift=fl_shift, lineout_wls=lineout_wls)
