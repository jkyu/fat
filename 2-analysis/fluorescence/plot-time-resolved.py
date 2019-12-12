import numpy as np
import os
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

def integrate_fluorescence_energy(fl, wgrid, tgrid, bounds=None):
    '''
    Integrate the fluorescence along the axis of energy.
    In the fluorescence data structure, the rows are time
    and the columns are energy. Integration is done using the trapezoidal
    rule in numpy.
    '''
    egrid = np.array([1240./x for x in wgrid])
    ngrid = len(wgrid)

    if bounds:
        truncated_egrid = [True if x in bounds else False for x in range(len(egrid))]
    else:
        truncated_egrid = egrid > 1.6
    eint = np.trapz(fl[:,truncated_egrid], axis=1) # integrate out energy
    fl_1D = eint / np.max(eint)

    return fl_1D, tgrid

def fl_slices(fl, wgrid, tgrid, slices_wl, shift, width=0., error=None):

    if error:
        er = error['fluorescence_error']
    egrid = np.array([1240./x for x in wgrid])
    # slices_wl = [650, 800]
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
        # print(egrid[idx[0]]) # print the energy corresponding to the wavelength
        fl_slice = np.array(fl[:, idx[0]])
        fl_slice = fl_slice / np.max(fl_slice)
        er_slice = np.array(er[:, idx[0]])
        er_slice = er_slice / np.max(fl_slice)
        slices.append(fl_slice)
        er_slices.append(er_slice)

    return slices, er_slices

def integrate_fluorescence_time(fl, wgrid, tgrid):
    '''
    Integrate the fluorescence along the axis of time.
    In the fluorescence data structure, the rows are time
    and the columns are energy. Integration is done using the trapezoidal
    rule in numpy.
    '''
    egrid = np.array([1240./x for x in wgrid])
    ngrid = len(tgrid)

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

def run(fl, fl_error, wgrid, tgrid, slice_wls, compute_shift=False, tshift=0):

    ''' Optional flag for computing the energy shift to match the fluorescence maximum of
    the experimental data by comparing steady state fluorescence spectra. '''
    if compute_shift:
        ''' Gather data for comparison to experimental steady state fluorescence '''
        expt_fl  = np.loadtxt('./expt-data/dobler-fl.txt', delimiter=',') # has a hard coded path
        expt_fl_x, expt_fl_y = process_expt(expt_fl)
        expt_fl_max = find_max(expt_fl_y, expt_fl_x)

        ''' Compute the energy shift of our time-integrated fluorescence relative to experiment. '''
        fl_1D, x_fl = integrate_fluorescence_time(fl, wgrid, tgrid)
        fl_1D_max = find_max(fl_1D, x_fl)
        fl_shift = fl_1D_max[1] - expt_fl_max[1]
        print('Energy Shift = %f eV' %fl_shift)
        np.savetxt('./data/fl_shift.txt', np.array([fl_shift]))
    else: 
        fl_shift = 0.

    ''' Take slices from the 2D fluorescence spectrum. '''
    slices, error_slices = fl_slices(fl, wgrid, tgrid, slice_wls, fl_shift, error=fl_error)

    ''' Collect experimental fluorescence lineout data. You probably have this saved
    by tracing some lines from a previously published fluorescence trace. '''
    expt_slices = []
    # do whatever to load the expt data. just put them into a list, though.
    for wl in slice_wls:
        expt_raw = np.loadtxt('./expt-data/%d.txt' %wl, delimiter=',')  # hard coded paths
        expt_t, expt_fl = process_expt_trf(expt_raw)
        expt_slices.append(expt_fl)

    ''' Compute exponential fit for fluorescence lineouts. The range of the
    exponential fit is taken to start at the fluorescence maximum. '''
    taus_aims = []
    taus_expt = []
    # expt_fits = [] # if you want to plot the exp fit for debugging reasons
    # aims_fits = []
    for i, wl in enumerate(slice_wls):
        tmax = np.argmax(expt_slices[i])
        popt, pcov = exp_fit(expt_t[tmax:] - expt_t[tmax], expt_slices[i][tmax:])
        taus_expt.append(1./popt[0]+expt_t[tmax])
        # fit = exp_func(expt_t, *popt) # exp fits for debugging
        # expt_fits.append(fit) # exp fits for debugging 
        print('%d nm experimental decay constant: %f' %(wl, (1./popt[0]+expt_t[tmax])))
        tmax = np.argmax(slices[i])
        popt, pcov = exp_fit(tgrid[tmax:] - tgrid[tmax], slices[i][tmax:])
        taus_aims.append(1./popt[0]+tgrid[tmax])
        # fit = exp_func(tgrid, *popt) # exp fits for debugging
        # aims_fits.append(fit) # exp fits for debugging 
        print('%d nm AIMS decay constant: %f' %(wl, (1./popt[0]+tgrid[tmax])))

    '''
    Plot 1D for time resolved fluorescence. Take slices of the 2D fluorescence
    where the slices correspond to wavelengths listed in slice_wls, but shifted to match
    the experimental fluorescence spectrum if fl_shift is not zero (default). 
    '''
    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    legendsize = 12
    colors = ['black', 'darkviolet']
    error_colors = ['slateblue', 'violet']
    styles = ['-', '--']

    for i in range(len(slice_wls)): 
        plt.plot(tgrid, slices[i], linewidth=2.0, color=colors[1], linestyle=styles[i])
        plt.errorbar(tgrid[(i*5)::10], slices[i][(i*5)::10], yerr=error_slices[i][(i*5)::10], color=colors[1], linewidth=0, capsize=2.0, elinewidth=0.8, ecolor=error_colors[i], linestyle=styles[i], label='AIMS, %d nm, $\\tau=$%d fs' %(slice_wls[i], taus_aims[i]))
        ''' The following three lines plot the exp fit for debugging. '''
        # tmax = np.argmax(slices[i]) 
        # plt.plot(tgrid-tgrid[tmax]+tshift, slices[i], label='AIMS, %d nm, $\\tau=$%d fs' %(slice_wls[i], taus_aims[i]), linewidth=2.0, color=colors[0], linestyle=styles[i])
        # plt.plot(tgrid+tshift, aims_fits[i], label='Exp Fit, %d nm' %(slice_wls[i]), linewidth=1.0, color=colors[0], linestyle=styles[i]) # exp fits for debugging

    for i in range(len(slice_wls)):
        plt.plot(expt_t+tshift, expt_slices[i], label='Expt, %d nm, $\\tau=$%d fs' %(slice_wls[i], taus_expt[i]), linewidth=2.0, color=colors[0], linestyle=styles[i])
        ''' The following three lines plot the exp fit for debugging. '''
        # tmax = np.argmax(expt_slices[i]) 
        # plt.plot(expt_t-expt_t[tmax]+tshift, expt_slices[i], label='Expt, %d nm, $\\tau=$%d fs' %(slice_wls[i], taus_expt[i]), linewidth=2.0, color=colors[0], linestyle=styles[i])
        # plt.plot(expt_t+tshift, expt_fits[i], label='Exp Fit, %d nm' %(slice_wls[i]), linewidth=1.0, color=colors[0], linestyle=styles[i]) # exp fits for debugging

    # plt.axis([-100, 1000, 0, 1.3])
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.ylabel('Fluorescence Intensity [au]', fontsize=labelsize)
    plt.legend(loc='best', fontsize=legendsize, frameon=False)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.tight_layout()
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/time-resolved-fluorescence.pdf')


''' Gather data from our computed 2D fluorescence spectrum. '''
fl_data = pickle.load(open('./data/fluorescence.pickle', 'rb'))
wgrid = fl_data['wgrid']
tgrid = fl_data['tgrid']
fl    = fl_data['fluorescence']
fl_error = pickle.load(open('./data/fl-error.pickle', 'rb'))
slice_wls = [650, 800]
run(fl, fl_error, wgrid, tgrid, slice_wls)
