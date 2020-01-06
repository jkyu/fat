import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')

def exp_func(x, b):
    return 1. * np.exp(-b * x)

def exp_fit(x, y):

    popt, pcov = curve_fit(exp_func, x, y, absolute_sigma=False)
    sigma = [ pcov[i,i]**.5 for i in range(0,len(pcov[0,:])) ]

    return popt, sigma

def process_expt(data):

    # Uses spline interpolation between points to generate a smoother plot
    x = data[:,0]
    y = data[:,1]
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(np.min(x), np.max(x), (np.max(x)-np.min(x))/500.)
    xnew_eV = [ 1240./z for z in xnew ]
    ynew = interpolate.splev(xnew, tck, der=0)
    ynew = ynew / np.max(ynew)

    return xnew_eV, ynew 

def process_expt_trf(data):
    
    # Reading in time resolved fluorescence.
    x = data[:,0]
    y = data[:,1]
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(-0.1, 0.6, 1001)
    xnew_fs = xnew*1000.
    ynew = interpolate.splev(xnew, tck, der=0)
    ynew = ynew / np.max(ynew)

    return xnew_fs, ynew 

def find_max(spectrum, grid):

    max_ind = np.argmax(spectrum)
    intensity_max = spectrum[max_ind]
    grid_max = grid[max_ind]

    return intensity_max, grid_max

def integrate_fluorescence_time(fl, wgrid, tgrid):
    '''
    Integrate the fluorescence along the axis of time.
    In the fluorescence data structure, the rows are time
    and the columns are energy. Integration is done using the trapezoidal
    rule in numpy.
    '''
    egrid = np.array([1240./x for x in wgrid])
    ngrid = len(wgrid)

    fl_tint = np.trapz(fl, axis=0) # integrate out time
    wl_max_ind = np.argmax(fl_tint)
    fl_max = np.max(fl_tint)
    '''
    Find local maxima by comparing each element to neighbors. Only 
    indices with fluorescence intensity greater than neighbors on right
    and left will return true. 
    '''
    local_max = (fl_tint >= np.roll(fl_tint, -1, 0)) & \
            (fl_tint >= np.roll(fl_tint, 1, 0))
    wl_max_inds = []
    for i in range(len(local_max)):
        if local_max[i]:
            wl_max_inds.append(i)

    max_wls = wgrid[wl_max_inds]
    fl_1D = fl_tint / np.max(fl_tint)

    return fl_1D, egrid

def plot_spectra(ab_grid, ab_spectrum, fl_2D, wgrid, tgrid, expt_ab_x, expt_ab_y, expt_fl_x, expt_fl_y):

    fl_spectrum, fl_grid = integrate_fluorescence_time(fl_2D, wgrid, tgrid)
    
    expt_ab_max = find_max(expt_ab_y, expt_ab_x)
    print('Experimental Absorption Maximum')
    print('Lamda max = %f at %f eV, %f nm' %(expt_ab_max[0], expt_ab_max[1], 1240./expt_ab_max[1]))
    print()
    
    expt_fl_max = find_max(expt_fl_y, expt_fl_x)
    print('Experimental Emission Maximum')
    print('Lamda max = %f at %f eV, %f nm' %(expt_fl_max[0], expt_fl_max[1], 1240./expt_fl_max[1]))
    print()
    
    sim_ab_max = find_max(ab_spectrum, ab_grid)
    print('Simulation Absorption Maximum')
    print('Lamda max = %f at %f eV, %f nm' %(sim_ab_max[0], sim_ab_max[1], 1240./sim_ab_max[1]))
    print()
    
    sim_fl_max = find_max(fl_spectrum, fl_grid)
    print('Simulation Emission Maximum')
    print('Lamda max = %f at %f eV, %f nm' %(sim_fl_max[0], sim_fl_max[1], 1240./sim_fl_max[1]))
    print()
    
    fl_shift = sim_fl_max[1] - expt_fl_max[1]
    ab_shift = sim_ab_max[1] - expt_ab_max[1]
    print('Absorption shift: %f' %ab_shift)
    print('Emission shift: %f' %fl_shift)

    stokes_expt = expt_ab_max[1] - expt_fl_max[1]
    stokes_sim = sim_ab_max[1] - sim_fl_max[1]
    print('Experimental Stokes Shift: %f' %stokes_expt)
    print('Simulation Stokes Shift: %f' %stokes_sim)

    ab_grid_shift = ab_grid - ab_shift
    fl_grid_shift = fl_grid - fl_shift

    '''
    Plot 1D figure showing the Stokes shift for experiment and theory.
    '''
    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    legendsize = 12
    ax = plt.subplot(111)

    ax.plot(ab_grid, ab_spectrum, label='Ab Sim', linewidth=2.5, linestyle='-', color='mediumseagreen')
    ax.plot(fl_grid, fl_spectrum, label='Fl Sim', linewidth=2.5, linestyle='--', color='mediumseagreen')
    ax.plot(expt_ab_x, expt_ab_y, label='Ab Expt', color='black', linestyle='-', linewidth=2.5)
    ax.plot(expt_fl_x, expt_fl_y, label='Fl Expt', color='black', linestyle='--', linewidth=2.5)

    ax.plot( [sim_ab_max[1]]*2, [0, 1.3], linestyle='-', linewidth=1.5, color='mediumseagreen')
    ax.plot( [sim_fl_max[1]]*2, [0, 1.3], linestyle='--', linewidth=1.5, color='mediumseagreen')
    ax.plot([expt_ab_max[1]]*2, [0, 1.3], linestyle='-', linewidth=1.5, color='black')
    ax.plot([expt_fl_max[1]]*2, [0, 1.3], linestyle='--', linewidth=1.5, color='black')

    head_length = 0.03
    head_width=0.03
    ax.arrow( expt_ab_max[1] + 0.01, expt_ab_max[0] + 0.2, (ab_shift-head_length-0.01), 0, width=0.003, head_length=head_length, head_width=head_width, color = 'slateblue' )
    ax.text( expt_ab_max[1] + 0.33, expt_ab_max[0] + 0.21, '%.2f eV' %(ab_shift), color='slateblue', fontsize=ticksize)
    ax.arrow( expt_fl_max[1] + 0.01, expt_fl_max[0] + 0.2, (fl_shift-head_length-0.01), 0, width=0.003, head_length=head_length, head_width=head_width, color = 'slateblue' )
    ax.text( expt_fl_max[1] + 0.03, expt_fl_max[0] + 0.23, '%.2f eV' %(fl_shift), color='slateblue', fontsize=ticksize)

    ax.arrow( expt_ab_max[1], sim_ab_max[0] + 0.10, -1 * (stokes_expt-head_length), 0, width=0.003, head_length=head_length, head_width=head_width, color = 'black' )
    ax.text( expt_ab_max[1] - 0.4, sim_ab_max[0] + 0.04, '%.2f eV' %(stokes_expt), color='black', fontsize=ticksize)
    ax.arrow( sim_ab_max[1], sim_fl_max[0] - 0.21, -1 * (stokes_sim-head_length), 0, width=0.003, head_length=head_length, head_width=head_width, color = 'mediumseagreen' )
    ax.text( sim_ab_max[1] - 0.43, sim_fl_max[0] - 0.18, '%.2f eV' %(stokes_sim), color='mediumseagreen', fontsize=ticksize)

    ax.legend(loc='best', fontsize=legendsize, frameon=False, ncol=1)
    plt.ylabel('Normalized Intensity', fontsize=labelsize)                                                 
    plt.xlabel('$\Delta$E [eV]', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.ylim([0, 1.3])
    plt.xlim([1, 6])
    plt.tight_layout()
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    fig.savefig('./figures/combined_ab_fl.pdf', dpi=300)
    plt.close()

if __name__=='__main__':

    ab_expt_file = 'path to experimental steady state absorption spectrum written to a txt file by WebPlotDigitizer'
    expt_ab = np.loadtxt(ab_expt_file, delimiter=',')
    expt_ab_x, expt_ab_y = process_expt(expt_ab)

    fl_expt_file = 'path to experimental steady state fluorescence spectrum written to a txt file by WebPlotDigitizer'
    expt_fl = np.loadtxt(fl_expt_file, delimiter=',')
    expt_fl_x, expt_fl_y = process_expt(expt_fl)
    
    ab_file = 'path to absorption pickle file or whatever format you used'
    ab_data = np.load(ab_file)
    ab_grid = ab_data['grid']
    ab_spectrum = ab_data['absorption']
    
    fl_file = 'path to the fluorescence data file computed somewhere else'
    fl_data = pickle.load(open(fl_file, 'rb'))
    wgrid = fl_data['wgrid']
    tgrid = fl_data['tgrid']
    fl_2D = fl_data['fluorescence']
    
    plot_spectra(ab_grid, ab_spectrum, fl_2D, wgrid, tgrid, expt_ab_x, expt_ab_y, expt_fl_x, expt_fl_y)
