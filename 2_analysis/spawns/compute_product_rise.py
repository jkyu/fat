import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
import sys

'''
The population rise and decay for all electronic states are computed here.
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

def get_populations(ics, tgrid, datadir, nstates, cis_keys, trans_keys):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    for i in range(nstates):
        states['s%d' %i] = []

    ''' Grab population information out of all ICs and bin that onto a uniform time step time grid (passed in as tgrid) '''
    for ic in ics:
        ic_tfinal = 0
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']
            # print('%s, state s%d' %(tbf_key, tbf_state))

            states['s%d' %tbf_state].append(tbf_key)

            time_steps = tbf['time_steps']
            if time_steps[-1] > ic_tfinal:
                ic_tfinal = time_steps[-1]
            populations = tbf['populations']

            interp_pop = interpolate(tgrid, time_steps, populations)
            interp_populations['%s' %tbf_key] = interp_pop
        print('IC %04d final time step: %f' %(ic, ic_tfinal))

    print('Total number of TBFs:')
    print('  Number of ICs: ', len(ics))
    print('  Total number of TBFs: ', np.sum([ len(states[key]) for key in states.keys() ]))
    for state_key in states.keys():
        print('  Number of %s TBFs: %d' %(state_key.title(), len(states[state_key])))
    
    avg_populations = {}
    state_populations = {}
    pop_errors = {}
    state = 's0'
    ''' Compute the average of the population over all ICs '''
    print('Averaging populations for state %s' %state)
    state_pops_cis = np.zeros((len(ics), len(tgrid)))
    state_pops_trans = np.zeros((len(ics), len(tgrid)))
    for i, ic in enumerate(ics):
        ic_pop_cis = np.zeros(len(tgrid))
        ic_pop_trans = np.zeros(len(tgrid))
        for tbf_key in states[state]:
            # Group TBFs of from same state and same IC
            if ic == int(tbf_key.split('-')[0]) and tbf_key in cis_keys:
                ic_pop_cis += interp_populations[tbf_key]
            elif ic == int(tbf_key.split('-')[0]) and tbf_key in trans_keys:
                ic_pop_trans += interp_populations[tbf_key]
        state_pops_cis[i,:] = ic_pop_cis
        state_pops_trans[i,:] = ic_pop_trans
    avg_pops_cis = np.mean(state_pops_cis, axis=0)
    avg_pops_trans = np.mean(state_pops_trans, axis=0)

    ''' Compute error for averaged ground state population using bootstrapping '''
    print('Computing sampling error for state %s' %state)
    state_error_cis = np.zeros(len(tgrid))
    state_error_trans = np.zeros(len(tgrid))
    for k in range(len(tgrid)):
        sample_inds = np.arange(len(ics))
        resample = [ [state_pops_cis[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
        resampled_means = np.mean(resample, axis=1)
        std = np.std(resampled_means)
        state_error_cis[k] = std
        resample = [ [state_pops_trans[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
        resampled_means = np.mean(resample, axis=1)
        std = np.std(resampled_means)
        state_error_trans[k] = std

    avg_populations['cis'] = avg_pops_cis
    state_populations['cis'] = state_pops_cis
    pop_errors['cis'] = state_error_cis
    avg_populations['trans'] = avg_pops_trans
    state_populations['trans'] = state_pops_trans
    pop_errors['trans'] = state_error_trans

    data2 = {}
    data2['ics'] = ics
    data2['tgrid'] = tgrid
    data2['nstates'] = nstates
    data2['populations'] = avg_populations # averaged populations by state
    data2['state_populations'] = state_populations # all populations by state label
    data2['errors'] = pop_errors
    data2['tbf_populations'] = interp_populations # populations for individual TBFs
    print('Dumping interpolated amplitudes to populations.pickle')

    if not os.path.isdir('./data/'):
        os.mkdir('./data/')

    with open('./data/product_rise.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/good-data/'
tgrid = np.arange(0, 750, 5) # edit the last number to change the grid spacing
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
qy_data = pickle.load(open('../dihedrals/data/qy.pickle', 'rb'))
cis_keys = qy_data['cis_keys']
trans_keys = qy_data['trans_keys']
get_populations(ics, tgrid, datadir, nstates, cis_keys, trans_keys)
