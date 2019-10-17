import os
import sys
import numpy as np
import mdtraj as md
import pickle
import matplotlib.pyplot as plt

'''
Reads in raw data from the FMS pickle files, computes relevant dihedral angles,
and then interpolates them on a grid to resolve the issue of adaptive time steps.
Also computes the average dihedrals if the flag is turned on and computes
the sampling error of the dihedral angles by bootstrapping.
'''

def interpolate(grid, tsteps, data, do_avg=False):

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
            if do_avg:
                interp_data[i] = interp_data[i-1]
            else:
                interp_data[i] = np.nan
    
    for i in range(len(interp_data)):
        if interp_data[i]==0:
            interp_data[i] = np.nan

    return interp_data

def compute_dihedrals(ics, tgrid, datadir, dihedral_index, do_avg=False):
    '''
    Load the fat data file and collect the spawn information.
    Gather the value of the dihedral angles from the trajectories.
    The dihedral angles are specified as a dictionary and taken into this
    function as dihedral_index.
    '''
    print('Loading trajectories for IC TBFs and computing dihedral angles.')

    dihedral_names = [x for x in dihedral_list.keys()]
    dihedral_list = [ dihedral_index[x] for x in dihedral_names ] 

    gs_keys = []
    ex_keys = []
    raw_angles = {}
    raw_tsteps = {}
    raw_pops = {}

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
            trajectory = tbf['trajectory']
            populations = tbf['populations']

            dihedrals = md.compute_dihedrals(trajectory, dihedral_list, periodic=False)

            raw_angles['%s' %tbf_key] = dihedrals
            raw_tsteps['%s' %tbf_key] = time_steps
            raw_pops['%s' %tbf_key] = populations

    '''
    Place the dihedral angles on a grid so that we don't have issues with averaging
    due to adaptive time steps. The data is stored in a dictionary indexed by the
    TBF name (e.g. 02-03) and then by the specific dihedral angle computed
    (e.g. C12-C13=C14-C15).
    '''
    interp_dihedrals = {}
    interp_zeroed = {}
    interp_populations = {}

    print('Aggregating dihedral angles and populations in time by interpolating.')
    for tbf_key in raw_angles.keys():
        dihes_dict = {}
        zeroed_dict = {}
        for dihe_idx, dihe_name in enumerate(dihedral_names):
            tsteps = raw_tsteps[tbf_key]
            dihes = raw_angles[tbf_key][:,dihe_idx]      # angle values of named dihedrals for each IC
            dihes = dihes * 180.0 / np.pi
            dihes = [ x + 360. if x<45 else x for x in dihes ]
            interp_dihes = interpolate(tgrid, tsteps, dihes, do_avg=do_avg)

            if tbf_key not in ex_keys:
                dihes_zeroed = interpolate(tgrid, np.array(tsteps) - tsteps[0], dihes, do_avg=do_avg)
                zeroed_dict[dihe_name] = dihes_zeroed

            dihes_dict[dihe_name] = interp_dihes
        interp_populations[tbf_key] = interpolate(tgrid, tsteps, raw_pops[tbf_key])
        interp_zeroed[tbf_key] = zeroed_dict
        interp_dihedrals[tbf_key] = dihes_dict

    if do_avg:
        print('Averaging dihedral angles across excited state TBFs.')
        avg_dihedrals_ex = {}
        error_ex = {}
        for dihe_name in dihedral_names:
            aggregated_dihedrals = np.zeros((len(ex_keys), len(tgrid)))
            for i, tbf_key in enumerate(ex_keys):
                aggregated_dihedrals[i,:] = interp_dihedrals[tbf_key][dihe_name]
            avg_dihedrals_ex[dihe_name] = np.mean(aggregated_dihedrals, axis=0)
            
            # Bootstrap sampling error
            # At each time step, sample the dihedrals for each excited state TBF
            # with one of the TBFs excluded. Do this 1000 times and take the 
            # average of each of these n-1 angle lists. The standard deviation
            # measures the sampling error via bootstrapping.
            error = []
            for k in range(len(tgrid)):
                sample_inds = np.arange(len(ex_keys))
                resample = [ [aggregated_dihedrals[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
                resampled_means = np.mean(resample, axis=1)
                std = np.std(resampled_means)
                error.append(std)

            error_ex[dihe_name] = np.array(error)

    print('Separating cis and trans TBFs.')
    cis_keys = []
    trans_keys = []
    for tbf_key in gs_keys:
        idx = np.max(np.argwhere(np.isfinite(interp_dihedrals[tbf_key]['C12-C13=C14-C15'])))
        final_angle = interp_dihedrals[tbf_key]['C12-C13=C14-C15'][idx]
        if final_angle > 270. or final_angle < 90.:
            cis_keys.append(tbf_key)
        else:
            trans_keys.append(tbf_key)

    if do_avg: 
        print('Averaging dihedral angles across cis ground state TBFs.')
        avg_dihedrals_gs = {}
        error_gs = {}
        for dihe_name in dihedral_names:
            aggregated_dihedrals = np.zeros((len(cis_keys), len(tgrid)))
            for i, tbf_key in enumerate(cis_keys):
                aggregated_dihedrals[i,:] = interp_zeroed[tbf_key][dihe_name]
            avg_dihedrals_gs[dihe_name] = np.mean(aggregated_dihedrals, axis=0)
            
            # Bootstrap sampling error
            error = []
            for k in range(len(tgrid)):
                sample_inds = np.arange(len(cis_keys))
                resample = [ [aggregated_dihedrals[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
                resampled_means = np.mean(resample, axis=1)
                std = np.std(resampled_means)
                error.append(std)

            error_gs[dihe_name] = np.array(error)

    # Cache data
    data2 = {}
    data2['dihedral_names'] = dihedral_names
    data2['all_dihedrals'] = interp_dihedrals
    data2['all_populations'] = interp_populations
    if do_avg: 
        data2['avg_dihedrals_ex'] = avg_dihedrals_ex
        data2['avg_dihedrals_gs'] = avg_dihedrals_gs
        data2['error_ex'] = error_ex
        data2['error_gs'] = error_gs
    data2['tgrid'] = tgrid
    data2['ex_keys'] = ex_keys
    data2['gs_keys'] = gs_keys
    data2['cis_keys'] = cis_keys
    data2['trans_keys'] = trans_keys

    print('Dumping interpolated amplitudes to dihedrals.pickle')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    if do_avg:
        with open('./data/dihedrals-avgs.pickle', 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('./data/dihedrals.pickle', 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
The following dihedral angles are enumerated and indexed according
to the geometry file so that we can use mdtraj to compute the
dihedral angles. Pass this dictionary into compute_dihedrals()
'''
print('Indexing dihedral angles.')
dihedral_index = {}
# dihedral_index['C6-C7=C8-C8']     = [3359, 3357, 3355, 3350]
# dihedral_index['C7=C8-C9=C10']    = [3357, 3355, 3350, 3348]
# dihedral_index['C8-C9=C10-C11']   = [3355, 3350, 3348, 3346]
# dihedral_index['C9=C10-C11=C12']  = [3350, 3348, 3346, 3344]
dihedral_index['C10-C11=C12-C13'] = [3348, 3346, 3344, 3339]
# dihedral_index['C11=C12-C13=C14'] = [3346, 3344, 3339, 3337]
dihedral_index['C12-C13=C14-C15'] = [3344, 3339, 3337, 3335]
# dihedral_index['C13=C14-C15=NZ']  = [3339, 3337, 3335, 3333]
dihedral_index['C14-C15=NZ-CE']   = [3337, 3335, 3333, 3330]
# dihedral_index['HOOP']            = [3340, 3339, 3337, 3338] # HOOP is CC13-C13=C14-HC14
# dihedral_index['CC13-C13-NZ-HNZ'] = [3340, 3339, 3334, 3333]

ics = [x for x in range(1, 33) if x not in [6,17] ]
datadir = '../../1-collect-data/data/'
compute_dihedrals(ics, tgrid, datadir, dihedral_index, do_avg=False)
