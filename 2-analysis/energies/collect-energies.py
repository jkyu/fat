import os
import sys
import numpy as np
import mdtraj as md
import pickle
import matplotlib.pyplot as plt

''' Get S0 and S1 energies from raw data and put it on a grid '''

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

def get_energy(ics, tgrid, datadir):
    '''
    Examine the energies for S0 and S1 over time for the TBFs. 
    Load the fat data file and collect the spawn information.
    Gather the energy angles from the trajectories.
    '''
    print('Loading excited state trajectories. Extracting energy and population information.')

    interp_s0_energies = {}
    interp_s1_energies = {}
    interp_populations = {}

    ex_keys = []
    gs_keys = []

    for ic in ics:
        data = pickle.load(open(datadir+('/%02d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():
            print(tbf_key)
            tbf = data[tbf_key]

            if tbf['tbf_id']==1:
                ex_keys.append(tbf_key)
            else:
                gs_keys.append(tbf_key)

            time_steps  = tbf['time_steps']
            populations = tbf['populations']
            s0_energies = tbf['energies']['s0'] * 27.21138602
            s1_energies = tbf['energies']['s1'] * 27.21138602

            interp_s0 = interpolate(tgrid, time_steps, s0_energies)
            interp_s1 = interpolate(tgrid, time_steps, s1_energies)
            interp_pop = interpolate(tgrid, time_steps, populations)

            interp_s0_energies['%s' %tbf_key] = interp_s0
            interp_s1_energies['%s' %tbf_key] = interp_s1
            interp_populations['%s' %tbf_key] = interp_pop

    data2 = {}
    data2['s0_energies'] = interp_s0_energies
    data2['s1_energies'] = interp_s1_energies
    data2['gs_keys'] = gs_keys
    data2['ex_keys'] = ex_keys
    data2['populations'] = interp_populations
    data2['tgrid'] = tgrid
    print('Dumping data to energies.pickle')
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    with open('./data/energies.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

ics = [ x for x in range(1,33) if x not in [6,17] ]
tgrid = np.arange(0, 1500, 5)
datadir = '../../1-collect-data/data/'
get_energy(ics, tgrid, datadir)
