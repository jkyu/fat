import os
import sys
import numpy as np
import mdtraj as md
import pickle
import matplotlib.pyplot as plt

'''
Takes the raw FMS data from ../1-collect-data/data/ and computes 
bond lengths. The bond lengths are interpolated on a grid to 
address the issue of adapted time steps. 
Note that the bond indexing is separated into singles and doubles
because I am interested in looking specifically at bond length alternation.
You will need to edit this if you want something else. 
'''

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

def compute_bla(ics, tgrid, datadir, bond_index):
    '''
    Load the fat data file and collect the spawn information.
    Gather the value of the dihedral angles from the trajectories.
    '''
    print('Loading trajectories for IC TBFs and computing bond distances along RPSB chain.')

    raw_tsteps = {}
    raw_singles = {}
    raw_doubles = {}
    raw_blas = {}

    gs_keys = []
    ex_keys = []

    single_names = ['C8-C9', 'C10-C11', 'C12-C13', 'C14-C15']
    single_list  = [ single_index[x] for x in single_names ]
    double_names = ['C9=C10', 'C11=C12', 'C13=C14', 'C15=NZ']
    double_list  = [ double_index[x] for x in double_names ]

    for ic in ics:
        data = pickle.load(open('../../../3-collect-data/data/%02d.pickle' %ic, 'rb'))
        for tbf_key in data.keys():

            print(tbf_key)
            tbf = data[tbf_key]

            if tbf['tbf_id']==1:
                ex_keys.append(tbf_key)
            else:
                gs_keys.append(tbf_key)

            # ic = tbf['initcond']
            time_steps = tbf['time_steps']
            trajectory = tbf['trajectory']

            singles = md.compute_distances(trajectory, single_list) * 10.
            doubles = md.compute_distances(trajectory, double_list) * 10.

            bla = np.zeros_like(time_steps)
            for t in range(len(time_steps)):
                bla[t] = np.mean(singles[t,:] - doubles[t,:])

            raw_tsteps['%s' %tbf_key] = time_steps
            raw_singles['%s' %tbf_key] = singles
            raw_doubles['%s' %tbf_key] = doubles
            raw_blas['%s' %tbf_key] = bla

    '''
    This places the bond length alternation, single bond lengths and double bond
    lengths on a set predefined grid so that we don't have issues with averaging
    due to the adaptive time steps in the dynamics.
    The data is stored in a dictionary indexed first by the TBF name (e.g. 02-03)
    and then by the speficic bond looked at (e.g. C13=C14s).
    '''
    interp_singles = {}
    interp_doubles = {}
    interp_blas = {}

    print('Aggregating single bonds in time by interpolating.')
    for tbf_key in raw_singles.keys():
        blens_dict = {}
        for bond_idx in range(len(single_names)):
            tsteps = raw_tsteps[tbf_key]
            blens = raw_singles[tbf_key][:,bond_idx]    # get lengths for named bond for each IC
            interp_blens = interpolate(tgrid, tsteps, blens)
            blens_dict[single_names[bond_idx]] = interp_blens
        interp_singles[tbf_key] = blens_dict

    print('Aggregating double bonds in time by interpolating.')
    for tbf_key in raw_doubles.keys():
        blens_dict = {}
        for bond_idx in range(len(single_names)):
            tsteps = raw_tsteps[tbf_key]
            blens = raw_doubles[tbf_key][:,bond_idx]    # get lengths for named bond for each IC
            interp_blens = interpolate(tgrid, tsteps, blens)
            blens_dict[double_names[bond_idx]] = interp_blens
        interp_doubles[tbf_key] = blens_dict

    print('Aggregating bond length alternation in time by interpolating.')
    for tbf_key in raw_blas.keys():
        interp_blas[tbf_key] = interpolate(tgrid, raw_tsteps[tbf_key], raw_blas[tbf_key])

    print('Averaging single bond lengths per IC')
    avg_singles = {}
    for tbf_key in interp_singles.keys():
        temp = np.zeros((len(single_names), len(tgrid)))
        for i, bond in enumerate(single_names):
            temp[i,:] = interp_singles[tbf_key][bond]
        avg_bond = np.mean(temp, axis=0)
        avg_singles[tbf_key] = avg_bond

    print('Averaging double bond lengths per IC')
    avg_doubles = {}
    for tbf_key in interp_doubles.keys():
        temp = np.zeros((len(double_names), len(tgrid)))
        for i, bond in enumerate(double_names):
            temp[i,:] = interp_doubles[tbf_key][bond]
        avg_bond = np.mean(temp, axis=0)
        avg_doubles[tbf_key] = avg_bond

    # Cache data
    data2 = {}
    data2['all_singles'] = interp_singles # indexed by ic, interpolated bond lengths vs t
    data2['all_doubles'] = interp_doubles
    data2['blas'] = interp_blas
    data2['avg_singles'] = avg_singles # averaged over bond lengths at each time step
    data2['avg_doubles'] = avg_doubles # indexed by IC
    data2['single_names'] = single_names # list of all bond names
    data2['double_names'] = double_names
    data2['tgrid'] = tgrid # array for time grid
    data2['ex_keys'] = ex_keys
    data2['gs_keys'] = gs_keys

    print('Dumping interpolated amplitudes to bla.pickle')
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    with open('./data/bla.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
The following single and double bonds are enumerated and indexed according
to the geometry file so that we can use mdtraj to compute the bond lengths.
'''
print('Indexing single and double bonds.')
single_index = {}
single_index['C8-C9']    = [3355, 3350]
single_index['C10-C11']  = [3348, 3346]
single_index['C12-C13']  = [3344, 3339]
single_index['C14-C15']  = [3337, 3335]

double_index = {}
double_index['C9=C10']  = [3350, 3348]
double_index['C11=C12'] = [3346, 3344]
double_index['C13=C14'] = [3339, 3337]
double_index['C15=NZ']  = [3335, 3333]

ics = [x for x in range(1,33) if x not in [6,17]]
tgrid = np.arange(0, 1500, 5)
datadir = '../1-collect-data/data/'
compute_bla(ics, tgrid, datadir, single_index, double_index)
