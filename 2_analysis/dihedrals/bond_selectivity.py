import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams

def weighted_bin(grid, angles, weights, ninitcond):
    '''
    Uses the change in population (diff_amp) along a trajectory
    to identify the torsional twists associated with population transfer.
    A torsion angle associated with torsion transfer (change in population)
    will be added to the bin. 
    '''
    
    grid_spacing = 360. / len(grid)
    for i in range(len(weights)):
        if np.isnan(weights[i]):
            weights[i] = 0

    diff_amp = np.zeros(len(weights))
    diff_amp[-1] = weights[-1]
    diff_amp[:-1] = weights[1:] - weights[:-1] 

    weighted_amplitude = np.zeros(len(grid))
    for j in range(len(diff_amp)):
        angle = angles[j]
        damp = diff_amp[j]
        
        if damp > 0:
            idx = np.argmin(np.abs(grid - angle))
            # print(grid[idx], damp)
            if angle/grid_spacing < grid[idx]:
                idx = idx - 1
            
            weighted_amplitude[idx] += damp

    # Normalization
    # weighted_amplitude = weighted_amplitude / ninitcond
    weighted_amplitude = weighted_amplitude

    return weighted_amplitude

def isomerization_selectivity(dihedral_data):
    dihedral_names = dihedral_data['dihedral_names']
    dihedrals = dihedral_data['dihedrals_state_specific']
    populations = dihedral_data['populations']
    
    # plot only S1 TBFs
    tbf_keys = dihedral_data['tbf_keys']
    ic_keys = [ x for x in tbf_keys if int(x.split('-')[1])==1 ]
    tbf_keys = ic_keys
    
    grid = np.arange(0, 360, 10)
    amps = {}
    for dname in dihedral_names:
        amp = np.zeros((len(grid)))
        for key in tbf_keys:
            pop = populations[key]
            dihe = dihedrals[key][dname]
            amp += weighted_bin(grid, dihe, pop, ninitcond=len(dihedral_data['ics']))
        amps[dname] = amp
    print(amps)
    
    xticks = np.arange(0, 9)*45
    # yticks = np.arange(0, 5)*0.01
    
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)
    
    colors = ['orchid', 'firebrick', 'darkturquoise', 'orange', 'darkseagreen', 'olive']
    for i, dname in enumerate(dihedral_names):
        plt.plot(grid, amps[dname], label='%s' %(dname), color='orchid')
    
    plt.axis([0, 360, 0, 0.30])
    plt.xlabel('Dihedral Angle', fontsize=labelsize)
    plt.ylabel('Population Transferred (S$_1 \\rightarrow$ S$_0$)', fontsize=labelsize)
    plt.legend(loc='best', frameon=False, fontsize=ticksize)
    plt.tight_layout()
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/isomerization-specificity.pdf', dpi=300)

if __name__=='__main__':
    dihedral_data = pickle.load(open('./data/dihedrals.pickle', 'rb'))
    isomerization_selectivity(dihedral_data)
