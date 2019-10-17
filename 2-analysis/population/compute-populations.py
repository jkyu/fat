import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
import sys

'''
The S1 population decay and S0 population rise is computed here.
The populations are placed on a uniform time grid and averaged.
This information is dumped to a pickle file and plotted in plot-populations.py
'''

# The exponential function to fit.
def exp_func(x, A, b, c):
    return A * np.exp(-b * x) + c

def interpolate(grid, tsteps, data):

    interp_data = np.zeros((len(grid)))
    spacing = np.max(grid) / float(len(grid))

    for i in range(len(grid)):
        if i==0:
            tlow = 0
        else:
            tlow = grid[i] - spacing/2
        if i==len(grid) - 1:
            thigh = grid[-1]
        else:
            thigh = grid[i] + spacing/2
        inds = [x for x, y in enumerate(tsteps) if y >= tlow and y <= thigh]
        dat = [data[ind] for ind in inds]
        tdiffs = [np.abs(grid[i] - y) for y in tsteps if y >= tlow and y <= thigh] # computes the distance of the raw data time point from the grid point
        if len(dat) > 0:
            tdiffs_frac = tdiffs / np.sum(tdiffs) # normalizes the distance from the grid point
            interp_data[i] = np.average(dat, weights=tdiffs_frac) # weighted average of the data points by their distance from the grid point
        else: 
            interp_data[i] = interp_data[i-1]
    
    return interp_data

def get_populations(ics, tgrid, datadir):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    ex_keys = []
    gs_keys = []

    ''' Grab population information out of all ICs and bin that onto a uniform 1 fs time step time grid '''
    for ic in ics:
        data = pickle.load(open(datadir+('/%02d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():
            print(tbf_key)
            tbf = data[tbf_key]

            if tbf['tbf_id']==1:
                ex_keys.append(tbf_key)
            else:
                gs_keys.append(tbf_key)

            time_steps = tbf['time_steps']
            populations = tbf['populations']

            interp_pop = interpolate(tgrid, time_steps, populations)
            interp_populations['%s' %tbf_key] = interp_pop
    
    ''' Compute average of the population over all excited state ICs '''
    print('Averaging populations at each time point across ICs.')
    all_ex_pops = np.zeros((len(ex_keys), len(tgrid)))
    for i, tbf_key in enumerate(ex_keys):
        all_ex_pops[i,:] = interp_populations[tbf_key]
    avg_ex_pops = np.mean(all_ex_pops, axis=0)

    ''' Compute error for averaged ground state population using bootstrapping '''
    ex_error = np.zeros(len(tgrid))
    for k in range(len(tgrid)):
        sample_inds = np.arange(len(ex_keys))
        resample = [ [all_ex_pops[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
        resampled_means = np.mean(resample, axis=1)
        std = np.std(resampled_means)
        ex_error[k] = std

    ''' Compute the average of the population over all ground state ICs '''
    all_gs_pops = np.zeros((len(ics), len(tgrid)))
    for i, ic in enumerate(ics):
        gs_pop = np.zeros(len(tgrid))
        for tbf_key in gs_keys:
            if ic == int(tbf_key.split('-')[0]):
                gs_pop += interp_populations[tbf_key]
        all_gs_pops[i,:] = gs_pop
    avg_gs_pops = np.mean(all_gs_pops, axis=0)

    ''' Compute error for averaged ground state population using bootstrapping '''
    gs_error = np.zeros(len(tgrid))
    for k in range(len(tgrid)):
        sample_inds = np.arange(len(ics))
        resample = [ [all_gs_pops[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
        resampled_means = np.mean(resample, axis=1)
        std = np.std(resampled_means)
        gs_error[k] = std

    data2 = {}
    data2['ics'] = ics
    data2['tgrid'] = tgrid
    data2['ex_populations'] = avg_ex_pops
    data2['ex_error'] = ex_error
    data2['gs_populations'] = avg_gs_pops
    data2['gs_error'] = gs_error
    data2['all_populations'] = interp_populations # populations for individual ICs
    print('Dumping interpolated amplitudes to populations.pickle')

    if not os.path.isdir('./data/'):
        os.mkdir('./data/')

    with open('./data/populations.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

''' Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
ICs are 1-32, excluding 6 and 17 because those AIMS trajectories died early due to REKS convergence errors '''
tgrid = np.arange(0, 1500, 5) # edit the last number to change the grid spacing
ics = [x for x in range(1, 33) if x not in [6,17] ]
datadir = '../../1-collect-data/data/'
get_populations(ics, tgrid, datadir)
