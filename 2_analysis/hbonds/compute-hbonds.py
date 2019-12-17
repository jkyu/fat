import os
import sys
import numpy as np
import mdtraj as md
import pickle
import matplotlib.pyplot as plt

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

def compute_hbonds(ics, tgrid, datadir, do_avg=False):

    '''
    Load the fat data file and collect the spawn information.
    Gather the value of the dihedral angles from the trajectories.
    This first load of the dihedral data is actually unnecessary. 
    I just wanted a quick and dirty way to separate the cis and
    trans TBFs. Ideally, you'd just do the sorting elsewhere or
    do it here so that this could be standalone. 
    '''
    dihe_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
    cis_keys = dihe_data['cis_keys']
    trans_keys = dihe_data['trans_keys']
    print(len(cis_keys))
    print(len(trans_keys))

    gs_keys = []
    ex_keys = []
    interp_hbonds = {}
    interp_zeroed = {}
    interp_populations = {}

    for ic in ics:
        data = pickle.load(open(datadir+('/%02d.pickle' %ic), 'rb'))
        # Should obviously not point to a direct path like this, but whatever. 
        # I don't see anyone using this script anyway. 
        qm_inds = np.loadtxt('/home/jkyu/data/br/5-aims/2-FMS/FMS-%d/qm.txt' %(ic))
        wat402_index = qm_inds[65]
        hbond_index = [[3333, int(wat402_index)]] # 3333 is the SB index
        print(ic, hbond_index)

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

            hbonds = md.compute_distances(trajectory, hbond_index)[:,0] * 10.
            interp_hbonds[tbf_key] = interpolate(tgrid, time_steps, hbonds, do_avg=do_avg)
            interp_populations[tbf_key] = interpolate(tgrid, time_steps, populations)
            if tbf_key not in ex_keys:
                interp_zeroed[tbf_key] = interpolate(tgrid, np.array(time_steps) - time_steps[0], hbonds, do_avg=do_avg)

    if do_avg:
        print('Averaging hbonds across excited state TBFs.')
        aggregated_hbonds = np.zeros((len(ex_keys), len(tgrid)))
        for i, tbf_key in enumerate(ex_keys):
            aggregated_hbonds[i,:] = interp_hbonds[tbf_key]
        avg_hbonds_ex = np.mean(aggregated_hbonds, axis=0)
        
        error = []
        for k in range(len(tgrid)):
            sample_inds = np.arange(len(ex_keys))
            resample = [ [aggregated_hbonds[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
            resampled_means = np.mean(resample, axis=1)
            std = np.std(resampled_means)
            error.append(std)
        error_ex = np.array(error)

        print('Averaging dihedral angles across cis ground state TBFs.')
        aggregated_hbonds = np.zeros((len(cis_keys), len(tgrid)))
        for i, tbf_key in enumerate(cis_keys):
            aggregated_hbonds[i,:] = interp_zeroed[tbf_key]
        avg_hbonds_cis = np.mean(aggregated_hbonds, axis=0)
        
        error = []
        for k in range(len(tgrid)):
            sample_inds = np.arange(len(cis_keys))
            resample = [ [aggregated_hbonds[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
            resampled_means = np.mean(resample, axis=1)
            std = np.std(resampled_means)
            error.append(std)
        error_cis = np.array(error)

        print('Averaging dihedral angles across trans ground state TBFs.')
        aggregated_hbonds = np.zeros((len(trans_keys), len(tgrid)))
        for i, tbf_key in enumerate(trans_keys):
            aggregated_hbonds[i,:] = interp_zeroed[tbf_key]
        avg_hbonds_trans = np.mean(aggregated_hbonds, axis=0)
        
        error = []
        for k in range(len(tgrid)):
            sample_inds = np.arange(len(trans_keys))
            resample = [ [aggregated_hbonds[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
            resampled_means = np.mean(resample, axis=1)
            std = np.std(resampled_means)
            error.append(std)
        error_trans = np.array(error)

    # Cache data
    data2 = {}
    data2['all_hbonds'] = interp_hbonds
    data2['all_populations'] = interp_populations
    if do_avg:
        data2['avg_hbonds_ex'] = avg_hbonds_ex
        data2['avg_hbonds_cis'] = avg_hbonds_cis
        data2['avg_hbonds_trans'] = avg_hbonds_trans
        data2['error_ex'] = error_ex
        data2['error_cis'] = error_cis
        data2['error_trans'] = error_trans
    data2['tgrid'] = tgrid
    data2['ex_keys'] = ex_keys
    data2['gs_keys'] = gs_keys
    data2['cis_keys'] = cis_keys
    data2['trans_keys'] = trans_keys

    print('Dumping interpolated amplitudes to hbond.pickle')
    if not os.path.isdir('./data/'):
        os.mkdir('./data')
    if do_avg:
        with open('./data/hbonds-avg.pickle', 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('./data/hbonds.pickle', 'wb') as handle:
            pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

tgrid = np.arange(0, 1500, 5)
ics = [x for x in range(1, 33) if x not in [6,17] ]
datadir = '../../1-collect-data/data/'
compute_hbonds(ics, tgrid, datadir, do_avg=True)
