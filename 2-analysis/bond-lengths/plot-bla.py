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

    # # Uses spline interpolation between points to generate a smoother plot
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x[0], x[-1], npoints)
    ynew = interpolate.splev(xnew, tck, der=0)

    return xnew, ynew 

def phase1(bla_data, offset=78):

    '''
    Plot excited state trajectories
    '''
    ex_keys = bla_data['ex_keys']
    gs_keys = bla_data['gs_keys']
    time = bla_data['tgrid']

    ''' Load data for excited state TBFs '''
    npoints = 51
    all_blas  = np.zeros((len(ex_keys), (npoints-1)*5+1))
    for i, key in enumerate(ex_keys):
        tgrid, all_blas[i,:] = smooth_data(time[:npoints], bla_data['blas'][key][:npoints], (npoints-1)*5+1)

    ''' Average over trajectories accounting for nan values '''
    ma_blas   = np.ma.MaskedArray(all_blas, mask=np.isnan(all_blas))
    avg_blas  = np.ma.average(ma_blas, axis=0)

    ''' Plot '''
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    for i in range(len(ex_keys)):
        plt.plot(tgrid, all_blas[i,:], linewidth=0.5, color='silver')
    plt.plot(tgrid, avg_blas, linestyle='-', color='black', label='Average BLA')

    plt.axis([0, 150, -0.15, 0.15])
    plt.ylabel('Bond Length Alternation [$\mathrm{\AA}$]', fontsize=labelsize)
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.legend(loc='best', frameon=False, fontsize=ticksize)
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/avg-bla.pdf')
    plt.savefig('./figures/avg-bla.png')

rcParams.update({'figure.autolayout': True})
bla_data = pickle.load(open('./data/bla.pickle', 'rb'))

phase1(bla_data)
