import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams

def weighted_bin(grid, angles, weights, ninitcond=30):
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

data = pickle.load(open('./data/dihedrals-avgs.pickle', 'rb'))
dihedral_names = data['dihedral_names']
ex_keys = data['ex_keys']

grid = np.arange(0, 360, 10)
amps = {}
for dname in dihedral_names:
    amp = np.zeros((len(grid)))
    for key in ex_keys:
        pop = data['all_populations'][key]
        dihe = data['all_dihedrals'][key][dname]
        amp += weighted_bin(grid, dihe, pop, ninitcond=len(ex_keys))
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

plt.plot(grid, amps[dihedral_names[0]], label='%s' %('$\\alpha$'), color='orchid')
plt.plot(grid, amps[dihedral_names[1]], label='%s' %('$\\beta$' ), color='darkorange')
plt.plot(grid, amps[dihedral_names[2]], label='%s' %('$\\gamma$'), color='slateblue')

plt.xticks(xticks, fontsize=ticksize)
# plt.yticks(yticks, fontsize=ticksize)
plt.axis([0, 360, 0, 0.07])
plt.xlabel('Dihedral Angle', fontsize=labelsize)
plt.ylabel('Population Transferred', fontsize=labelsize)
plt.legend(loc='best', frameon=False, fontsize=ticksize)
plt.tight_layout()
if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('./figures/isomerization-specificity.pdf')
plt.savefig('./figures/isomerization-specificity.png')
