import numpy as np
import os
import math
import sys
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cl
'''
This plots the 2D fluorescence spectrum from data we computed in
compute-fluorescence.py that should have been dumped to
./data/fluorescence.pickle
This uses a fluorescence shift that should be computed in the
output of plot-time-resolved.py and saved to ./data/fl_shift.txt
'''

def plot_fluorescence(fl_data, fl_shift):
    '''
    Here, we think about time as x and energy as y. It turns out that in
    the fluorescence 2D array, each row corresponds to a specific time 
    value, so the first index (rows) gives the time and the second 
    (column) index is energy. This means that time is on the y-axis and 
    energy is on the x-axis if we use the array as a plot. So we need
    to transpose this array. 
    '''
    wgrid = fl_data['wgrid']
    egrid = fl_data['egrid']
    tgrid = fl_data['tgrid']
    ngrid = len(tgrid)
    fl    = fl_data['fluorescence']

    # egrid = egrid - fl_shift
    # wgrid = np.array([1240./x for x in egrid])

    t0 = np.argmin(np.abs(0 - tgrid))
    xticks = np.arange(t0, ngrid, ngrid//5)
    xlabels = ['%d' %int(k) for k in tgrid[xticks]]
    yticks = np.arange(0, ngrid, ngrid//5)
    ylabels = ['%d' %int(k) for k in wgrid[yticks]]

    fl2 = fl.T

    ''' 
    Custom color map taken by extracting RGB values from Schmidt 2005 figure 1
    in order to match the scale and colors. 
    '''
    colors = [(101, 0, 145), (37, 0, 250), (0, 61, 255), (0, 162, 255), (0, 246, 255), (0, 255, 175), (1, 255, 0), (187, 255, 0), (255, 234, 0), (255, 162, 0), (255, 88, 0), (255, 13, 0) ] 
    colors = [ (x[0]/255., x[1]/255., x[2]/255.) for x in colors]
    
    cmap_name = 'special'
    cm = cl.LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    legendsize = 12

    plt.imshow(fl2, origin='lower', cmap=cm)

    # Plot line outs -- if you don't have single wavelength signal to highlight,
    # you can comment this block out
    wl650 = np.argmin(np.abs(egrid - (1240./650. + fl_shift)))
    wl800 = np.argmin(np.abs(egrid - (1240./800. + fl_shift)))
    print(wgrid[wl650], wgrid[wl800])
    plt.plot(np.arange(ngrid), [wl650]*len(tgrid), linewidth=2.0, linestyle='-', color='silver')
    plt.plot(np.arange(ngrid), [wl800]*len(tgrid), linewidth=2.0, linestyle='-', color='silver')

    plt.xticks(xticks, xlabels, fontsize=ticksize)
    plt.yticks(yticks, ylabels, fontsize=ticksize)
    plt.xlabel('Time [fs]',fontsize=labelsize)
    plt.ylabel('Wavelength [nm]',fontsize=labelsize)
    plt.colorbar()
    # fig.autofmt_xdate()
    plt.tight_layout()
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/2D-fluorescence.pdf')
    plt.close()

fl_data = pickle.load(open('./data/fluorescence.pickle', 'rb'))
fl_shift = np.loadtxt('./data/fl_shift.txt')
plot_fluorescence(fl_data, fl_shift)
