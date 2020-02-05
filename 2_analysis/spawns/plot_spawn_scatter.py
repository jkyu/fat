import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def sort_spawns(angle_data, angle_key, dihedral_data, dihe_key, qy_data):

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

    matplotlib.rcParams.update({'figure.autolayout': True})
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    fig = plt.figure(figsize=(6,5))
    for i, key in enumerate(trans_keys):
        size = spawn_ps[key] * 500
        plt.scatter(spawn_as[key], spawn_ds[key], alpha=0.6, color='lightsteelblue', s=size, linewidths=0)
    for i, key in enumerate(cis_keys):
        size = spawn_ps[key] * 500
        plt.scatter(spawn_as[key], spawn_ds[key], alpha=0.6, color='plum', s=size, linewidths=0)
    ''' Scatter points outside of plot range for labeling purposes. '''
    plt.scatter(600, 600, alpha=0.6, color='plum', s=50, linewidths=0, label='Cis')
    plt.scatter(600, 600, alpha=0.6, color='lightsteelblue', s=50, linewidths=0, label='Trans')
    s1 = plt.scatter(600, 600, alpha=0.6, color='lightgray', linewidths=0, s=100, label='0.2 Population')
    s2 = plt.scatter(600, 600, alpha=0.6, color='lightgray', linewidths=0, s=250, label='0.5 Population')
    s3 = plt.scatter(600, 600, alpha=0.6, color='lightgray', linewidths=0, s=400, label='0.8 Population')
    plt.legend(loc='best', fontsize=ticksize)
    plt.axis([95, 165, 175, 285])
    plt.xlabel('%s Angle' %angle_key, fontsize=labelsize)
    plt.ylabel('%s Dihedral Angle' %dihe_key, fontsize=labelsize)

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/spawn_scatter.pdf', dpi=300)

if __name__=='__main__':
    angle_data = pickle.load(open('../angles/data/angles.pickle', 'rb'))
    angle_keys = angle_data['angle_names']
    dihedral_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
    dihe_keys = dihedral_data['dihedral_names']
    qy_data = pickle.load(open('../dihedrals/data/qy.pickle', 'rb'))
    sort_spawns(angle_data, angle_keys[0], dihedral_data, dihe_keys[0], qy_data)
