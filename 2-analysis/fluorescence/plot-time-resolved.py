import numpy as np
import sys
from scipy.optimize import curve_fit
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
'''
Computes the fluorescence shift and plots 1D fluorescence lineouts that
can be compared to experimental time-resolved fluorescence results that
in the past have been often measured at specific wavelengths.
The fluorescence shift is printed here but also saved to ./data/fl_shift.txt
'''

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
    xnew = np.linspace(-0.09, 0.6, 1001)
    xnew_fs = xnew*1000.
    ynew = interpolate.splev(xnew, tck, der=0)
    ynew = ynew / np.max(ynew)

    return xnew_fs, ynew 

def find_max(data, x, fl=False):

    max_ind = np.argmax(data)
    lmax = data[max_ind]
    xmax = x[max_ind]
    if fl:
        max_ind = np.argmax(data[:(len(data)//3)])
        lmax = data[max_ind]
        xmax = x[max_ind]

    return lmax, xmax

def integrate_fluorescence_energy(fl_data, bounds=None):
    '''
    Integrate the fluorescence along the axis of energy.
    In the fluorescence data structure, the rows are time
    and the columns are energy. Integration is done using the trapezoidal
    rule in numpy.
    '''
    wgrid = fl_data['wgrid']
    egrid = fl_data['egrid']
    tgrid = fl_data['tgrid']
    ngrid = len(wgrid)
    fl    = fl_data['fluorescence']

    if bounds:
        truncated_egrid = [True if x in bounds else False for x in range(len(egrid))]
    else:
        truncated_egrid = egrid > 1.6
    eint = np.trapz(fl[:,truncated_egrid], axis=1) # integrate out energy
    fl_1D = eint / np.max(eint)

    return fl_1D, tgrid

def fl_slices(fl_data, shift, width=0., error=None):

    if error:
        er = error['fluorescence_error']
    fl = fl_data['fluorescence']
    egrid = fl_data['egrid']
    tgrid = fl_data['tgrid']

    slices_wl = [650, 800]
    slices_ev = [ 1240./x for x in slices_wl ]
    slices_shifted = [ x+shift for x in slices_ev ]

    slice_idx = []
    if width==0:
        for x in slices_shifted:
            idx_low = np.argmin(np.abs((egrid-x)))
            idx_high = idx_low
            slice_idx.append([int(idx_high), int(idx_low)])
    else:
        for x in slices_shifted:
            idx_low  = np.argmin(np.abs((egrid - (x-width)))) # long wavelength
            idx_high = np.argmin(np.abs((egrid - (x+width)))) # short wavelength
            slice_idx.append([int(x) for x in range(idx_high, idx_low)])
            # idx = np.argmin(np.abs((egrid - x)))
            # slice_idx.append([idx])

    slices = []
    er_slices = []
    for idx in slice_idx:
        print(egrid[idx[0]])
        fl_slice = np.array(fl[:, idx[0]])
        fl_slice = fl_slice / np.max(fl_slice)
        er_slice = np.array(er[:, idx[0]])
        er_slice = er_slice / np.max(fl_slice)
        slices.append(fl_slice)
        er_slices.append(er_slice)

    return slices, er_slices

def integrate_fluorescence_time(fl_data):
    '''
    Integrate the fluorescence along the axis of time.
    In the fluorescence data structure, the rows are time
    and the columns are energy. Integration is done using the trapezoidal
    rule in numpy.
    '''
    wgrid = fl_data['wgrid']
    egrid = fl_data['egrid']
    tgrid = fl_data['tgrid']
    ngrid = len(tgrid)
    fl    = fl_data['fluorescence']

    ngrid = len(tgrid)
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

def run():

    ''' Gather data for comparison to experimental fluorescence lineouts '''
    expt_fl  = np.loadtxt('./expt-data/dobler-fl.txt', delimiter=',')
    expt_fl_x, expt_fl_y = process_expt(expt_fl)
    expt_fl_max = find_max(expt_fl_y, expt_fl_x)

    ''' Gather data from our computed fluorescence spectrum and compute
    the energy shift of our spectrum relative to experiment. Also have
    the option to shift in time, although this is zero by default. '''
    tshift = 0
    fl_data = pickle.load(open('./data/fluorescence.pickle', 'rb'))
    fl_1D, x_fl = integrate_fluorescence_time(fl_data)
    fl_1D_max = find_max(fl_1D, x_fl)
    fl_shift = fl_1D_max[1] - expt_fl_max[1]
    print('Energy Shift = %f eV' %fl_shift)
    np.savetxt('fl_shift.txt', fl_shift)

    '''
    Plot 1D for time resolved fluorescence. Take slices of the 2D fluorescence
    where the slices correspond to 650 nm and 800 nm, but shifted to match
    the experimental fluorescence spectrum.
    '''
    expt_650 = np.loadtxt('./expt-data/schmidt-f3-650.txt', delimiter=',')
    expt_800 = np.loadtxt('./expt-data/schmidt-f3-800.txt', delimiter=',')
    expt_t, expt_fl_650 = process_expt_trf(expt_650)
    _, expt_fl_800= process_expt_trf(expt_800)

    fl_error = pickle.load(open('./data/fl-error.pickle', 'rb'))
    slices, error_slices = fl_slices(fl_data, fl_shift, error=fl_error)

    tmax = np.argmax(expt_fl_650)
    taus = []
    popt, pcov = exp_fit(expt_t[tmax:] - expt_t[tmax], expt_fl_650[tmax:])
    taus.append(1./popt[0]+expt_t[tmax])
    print('650 nm experimental decay constant: %f' %(1./popt[0]+expt_t[tmax]))
    tmax = np.argmax(expt_fl_800)
    popt, pcov = exp_fit(expt_t[tmax:] - expt_t[tmax], expt_fl_800[tmax:])
    taus.append(1./popt[0]+expt_t[tmax])
    print('800 nm experimental decay constant: %f' %(1./popt[0]+expt_t[tmax]))
    tmax = np.argmax(slices[0])
    popt, pcov = exp_fit(fl_data['tgrid'][tmax:] - fl_data['tgrid'][tmax], slices[0][tmax:])
    taus.append(1./popt[0]+fl_data['tgrid'][tmax])
    print('650 nm AIMS decay constant: %f' %(1./popt[0]+fl_data['tgrid'][tmax]))
    tmax = np.argmax(slices[1])
    popt, pcov = exp_fit(fl_data['tgrid'][tmax:] - fl_data['tgrid'][tmax], slices[1][tmax:])
    taus.append(1./popt[0]+fl_data['tgrid'][tmax])
    print('800 nm AIMS decay constant: %f' %(1./popt[0]+fl_data['tgrid'][tmax]))

    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    legendsize = 12
    colors = ['black', 'darkviolet']
    error_colors = ['slateblue', 'violet']
    styles = ['-', '--']

    for i in range(2): 
        plt.plot(fl_data['tgrid'], slices[i], linewidth=2.0, color=colors[1], linestyle=styles[i])
        plt.errorbar(fl_data['tgrid'][(i*5)::10], slices[i][(i*5)::10], yerr=error_slices[i][(i*5)::10], color=colors[1], linewidth=2.0, capsize=2.0, elinewidth=0.8, ecolor=error_colors[i], linestyle=styles[i], label='AIMS, %d nm, $\\tau=$%d fs' %(np.array([650, 800])[i], taus[i+2]))
    plt.plot(expt_t+tshift, expt_fl_650, label='Expt, 650 nm, $\\tau=$%d fs' %taus[0], linewidth=2.0, color=colors[0], linestyle=styles[0])
    plt.plot(expt_t+tshift, expt_fl_800, label='Expt, 800 nm, $\\tau=$%d fs' %taus[1], linewidth=2.0, color=colors[0], linestyle=styles[1])

    plt.axis([-100, 1000, 0, 1.3])
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.ylabel('Fluorescence Intensity [au]', fontsize=labelsize)
    plt.legend(loc='best', fontsize=legendsize, frameon=False)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.tight_layout()
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/time-resolved-fluorescence.pdf')

run()
