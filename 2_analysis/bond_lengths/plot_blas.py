import sys
import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import interpolate
import pickle
import numpy as np

''' Nothing special. Just plots the average BLA starting from time zero.
Averaging over a long time will look weird because the wavepacket decoheres,
but good to see if there's a trend at short time delay (e.g. the rapid BLA
inversion in bR) '''

def smooth_data(x, y, npoints):
    
    ''' The smoothing function doesn't work if you have NaNs in your data (e.g. data 
    TBFs that do not start at time zero) and should not be used if averaging over 
    all TBFs '''

    # Uses spline interpolation between points to generate a smoother plot
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x[0], x[-1], npoints)
    ynew = interpolate.splev(xnew, tck, der=0)

    return xnew, ynew 

def plot(bla_data, figname='avg-bla', do_smoothing=False):

    '''
    Plot excited state trajectories
    '''
    time = bla_data['tgrid']
    keys = [x for x in bla_data['blas'].keys()]

    ''' Load data for excited state TBFs '''
    all_blas  = np.zeros((len(keys), len(time)))
    if do_smoothing:
        for i, key in enumerate(keys):
            tgrid, all_blas[i,:] = smooth_data(time[:npoints], bla_data['blas'][key][:npoints], (npoints-1)*5+1)
    else:
        tgrid = time
        for i, key in enumerate(keys):
            all_blas[i,:] = bla_data['blas'][key]

    ''' Average over trajectories accounting for nan values '''
    ma_blas   = np.ma.MaskedArray(all_blas, mask=np.isnan(all_blas))
    avg_blas  = np.ma.average(ma_blas, axis=0)

    ''' Plot '''
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    for i in range(len(keys)):
        plt.plot(tgrid, all_blas[i,:], linewidth=0.5, color='silver')
    plt.plot(tgrid, avg_blas, linestyle='-', color='black', label='Average BLA')

    plt.axis([0, 150, -0.15, 0.15])
    plt.ylabel('Bond Length Alternation [$\mathrm{\AA}$]', fontsize=labelsize)
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.legend(loc='best', frameon=False, fontsize=ticksize)
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/%s.pdf' %figname, dpi=300)

if __name__=='__main__':
    rcParams.update({'figure.autolayout': True})
    bla_data = pickle.load(open('./data/bla.pickle', 'rb'))
    figname = 'avg_bla'
    plot(bla_data, figname=figname)
