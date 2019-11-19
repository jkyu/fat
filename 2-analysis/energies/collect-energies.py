import os
import sys
import numpy as np
import mdtraj as md
import pickle
import matplotlib.pyplot as plt

''' Get all energies from raw data and put it on a grid '''

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
            interp_data[i] = np.nan

    return interp_data

def get_energy(ics, tgrid, datadir, nstates):
    '''
    Examine the energies for S0 and S1 over time for the TBFs. 
    Load the fat data file and collect the spawn information.
    Gather the energy angles from the trajectories.
    '''
    print('Loading excited state trajectories. Extracting energy and population information.')

    interp_populations = {}
    interp_energies = {}
    states = {}
    for i in range(nstates):
        states['s%d' %i] = []

    ''' Grab energy information out of all ICs and bin them onto a uniform time script. '''
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            state_id = tbf['state_id']
            print('%s, state s%d' %(tbf_key, state_id))

            states['s%d' %state_id].append(tbf_key)

            time_steps = tbf['time_steps']
            populations = tbf['populations']
            ic_energies = {}
            for i in range(nstates):
                state_energies = tbf['energies']['s%d' %i] 
                ic_energies['s%d' %i] = interpolate(tgrid, time_steps, state_energies)

            interp_pop = interpolate(tgrid, time_steps, populations)
            interp_populations['%s' %tbf_key] = interp_pop
            interp_energies['%s' %tbf_key] = ic_energies

    data2 = {}
    data2['energies'] = interp_energies
    data2['populations'] = interp_populations
    data2['tgrid'] = tgrid
    print('Dumping data to energies.pickle')
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    with open('./data/energies.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/data/'
tgrid = np.arange(0, 250, 5) # edit the last number to change the grid spacing
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
# ics = fmsinfo['ics']
ics = [3]
nstates = fmsinfo['nstates']
get_energy(ics, tgrid, datadir, nstates)
