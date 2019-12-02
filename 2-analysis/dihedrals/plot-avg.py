import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import numpy as np

''' Plot the TBF-averaged dihedral angle over time. 
This is kind of dumb right now because the dihedral angles are periodic,
so unless the allowed rotations are constrained by the system or I do something
to deal with the periodicity (really, how to automate describing the twists always
in one direction), the dihedral angle will average to around 0 with enough TBFs. '''

def plot(dihedral_data):

    tbf_keys = dihedral_data['tbf_keys']

    dihe_keys = dihedral_data['dihedral_names']
    state_ids = dihedral_data['state_ids']
    dihedrals = dihedral_data['dihedrals']
    tgrid = dihedral_data['tgrid']

    ''' For each dihedral angle of interest (dihe_keys), the dihedral angles in time are stored.
    Average over the TBFs, so we have the TBF-averaged change in each dihedral angle over time. '''
    avg_dihedrals = {} 
    for dihe_key in dihe_keys: 
        all_tbfs = np.zeros((len(tbf_keys), len(tgrid)))
        for i, tbf_key in enumerate(tbf_keys):
            all_tbfs[i, :] = dihedrals[tbf_key][dihe_key] # big array of dihedrals indexed by tbf number
        ma_tbfs   = np.ma.MaskedArray(all_tbfs, mask=np.isnan(all_tbfs)) 
        avg_tbfs  = np.ma.average(ma_tbfs, axis=0) # averaged dihedral angle over all TBFs 
        avg_dihedrals[dihe_key] = avg_tbfs # store tbf-averaged dihedral angles to dictionary

    fig = plt.figure(figsize=(6,5))
    labelsize = 14
    ticksize = 12
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    p1 = plt.subplot(1,1,1)

    for i, dihe_key in enumerate(dihe_keys):
        avg_dihe = avg_dihedrals[dihe_key]
        p1.plot(tgrid, avg_dihe, linestyle='-', color=colors[i], label=dihe_key)
    p1.legend(loc='best', frameon=False, fontsize=ticksize)
    # p1.axis([tgrid[0], tgrid[-1], -0.15, 0.15])
    p1.set_ylabel('Dihedral Angle', fontsize=labelsize)
    p1.set_xlabel('Time [fs]', fontsize=labelsize)

    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/avg-dihedrals.png', dpi=300)
    plt.close()

rcParams.update({'figure.autolayout': True})
dihedral_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))

plot(dihedral_data)
