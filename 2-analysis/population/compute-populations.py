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

def get_populations(ics, tgrid, datadir, nstates):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    for i in range(nstates):
        states['s%d' %i] = []

    ''' Grab population information out of all ICs and bin that onto a uniform 1 fs time step time grid '''
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            state_id = tbf['state_id']
            print('%s, state s%d' %(tbf_key, state_id))

            states['s%d' %state_id].append(tbf_key)

            time_steps = tbf['time_steps']
            populations = tbf['populations']

            interp_pop = interpolate(tgrid, time_steps, populations)
            interp_populations['%s' %tbf_key] = interp_pop
    
    all_populations = {}
    all_errors = {}
    for state in states.keys():
        ''' Compute the average of the population over all ICs '''
        print('Averaging populations for state %s' %state)
        state_pops = np.zeros((len(ics), len(tgrid)))
        for i, ic in enumerate(ics):
            ic_pop = np.zeros(len(tgrid))
            for tbf_key in states[state]:
                # Group TBFs of from same state and same IC
                if ic == int(tbf_key.split('-')[0]):
                    ic_pop += interp_populations[tbf_key]
            state_pops[i,:] = ic_pop
        avg_pops = np.mean(state_pops, axis=0)

        ''' Compute error for averaged ground state population using bootstrapping '''
        print('Computing sampling error for state %s' %state)
        state_error = np.zeros(len(tgrid))
        for k in range(len(tgrid)):
            sample_inds = np.arange(len(ics))
            resample = [ [state_pops[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
            resampled_means = np.mean(resample, axis=1)
            std = np.std(resampled_means)
            state_error[k] = std

        all_populations[state] = avg_pops
        all_errors[state] = state_error

    data2 = {}
    data2['ics'] = ics
    data2['tgrid'] = tgrid
    data2['nstates'] = nstates
    data2['populations'] = all_populations
    data2['errors'] = all_errors
    data2['tbf_populations'] = interp_populations # populations for individual ICs
    print('Dumping interpolated amplitudes to populations.pickle')

    if not os.path.isdir('./data/'):
        os.mkdir('./data/')

    with open('./data/populations.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/data/'
tgrid = np.arange(0, 300, 5) # edit the last number to change the grid spacing
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
get_populations(ics, tgrid, datadir, nstates)
