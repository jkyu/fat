import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def sort_spawns(spawn_times, angle_data, angle_key, dihedral_data, dihe_key, qy_data):

    ics = dihedral_data['ics']
    tbf_keys  = angle_data['tbf_keys']
    tbf_states = angle_data['tbf_states']
    angles    = angle_data['angles_state_specific']
    dihedrals = dihedral_data['dihedrals_state_specific']
    populations = angle_data['populations']
    tgrid = angle_data['tgrid']

    trans_keys = qy_data['trans_keys']
    cis_keys = qy_data['cis_keys']

    ''' Check 1st point of spawned TBFs '''
    spawn_ds = {}
    spawn_as = {}
    spawn_ps = {}
    gs_keys = [ x for x in tbf_keys if tbf_states[x]==0 ]
    for key in gs_keys:
        angs = angles[key][angle_key]
        dihs = dihedrals[key][dihe_key]
        pop  = populations[key][-1] 
        ''' Find first non-NaN index '''
        ind = np.where(np.isnan(angs))[0][-1] + 1
        spawn_as[key] = angs[ind]
        spawn_ds[key] = dihs[ind]
        spawn_ps[key] = pop

    trans_time = population_weighted_average(trans_keys, spawn_times, spawn_ps)
    trans_mean, trans_err = bootstrap(ics, trans_keys, spawn_times, spawn_ps)
    cis_time = population_weighted_average(cis_keys, spawn_times, spawn_ps)
    cis_mean, cis_err = bootstrap(ics, cis_keys, spawn_times, spawn_ps)
    print('Trans: ')
    print('  %f +/- %f' %(trans_time, trans_err))
    print('Cis: ')
    print('  %f +/- %f' %(cis_time, cis_err))

def bootstrap(ics, tbf_keys, spawn_times, spawn_populations, nsamples=1000):

    resampled_avgs = []
    resample = [ [ x for x in np.random.choice(ics, size=(len(ics)), replace=True) ] for _ in range(100) ]
    for sample_ics in resample:
        resample_keys = []
        for sample_ic in sample_ics:
            match_keys = [ x for x in tbf_keys if int(x.split('-')[0])==sample_ic ] 
            resample_keys = resample_keys + match_keys
        avg = population_weighted_average(resample_keys, spawn_times, spawn_populations)
        resampled_avgs.append(avg)
    mean = np.mean(resampled_avgs)
    std = np.std(resampled_avgs)

    return mean, std

def population_weighted_average(tbf_keys, spawn_times, spawn_populations):

    ts = []
    pops = []
    for key in tbf_keys:
        ts.append(spawn_times[key])
        pops.append(spawn_populations[key])
    avg_spawn_time = np.average(ts, weights=pops)

    return avg_spawn_time

if __name__=='__main__':
    angle_data = pickle.load(open('../angles/data/angles.pickle', 'rb'))
    angle_keys = angle_data['angle_names']
    dihedral_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
    dihe_keys = dihedral_data['dihedral_names']
    qy_data = pickle.load(open('../dihedrals/data/qy.pickle', 'rb'))
    spawn_time_data = pickle.load(open('./data/spawn_times.pickle', 'rb'))
    sort_spawns(spawn_time_data['spawn_times'], angle_data, angle_keys[0], dihedral_data, dihe_keys[0], qy_data)
