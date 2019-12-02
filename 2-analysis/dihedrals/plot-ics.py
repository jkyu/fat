import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import numpy as np

def plot(ic, dihedral_data, nstates):

    tbf_keys = dihedral_data['tbf_keys']
    tbf_keys = [x for x in tbf_keys if ic==int(x.split('-')[0])]

    dihe_keys = dihedral_data['dihedral_names']
    state_ids = dihedral_data['state_ids']
    dihedrals = dihedral_data['dihedrals']
    tgrid = dihedral_data['tgrid']

    fig = plt.figure(figsize=(6,5))
    labelsize = 14
    ticksize = 12
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    labels = ['S%d' %x for x in range(nstates)]
    labeled = [False]*nstates

    p1 = plt.subplot(1,1,1)

    for tbf_key in tbf_keys:
        for dihe_key in dihe_keys:
            dihes = dihedrals[tbf_key][dihe_key]
            state_id = state_ids[tbf_key]
            if not labeled[state_id]:
                label = labels[state_id]
                labeled[state_id] = True
            else: label = None
            p1.plot(tgrid, dihes, linestyle='-', color=colors[state_id], label=label)
    p1.legend(loc='best', frameon=False, fontsize=ticksize)
    # p1.axis([tgrid[0], tgrid[-1], -0.15, 0.15])
    p1.set_ylabel('Dihedral Angle', fontsize=labelsize)
    p1.set_xlabel('Time [fs]', fontsize=labelsize)

    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/%04d.png' %ic)
    plt.close()

rcParams.update({'figure.autolayout': True})
dihedral_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
fmsinfo = pickle.load(open('../../1-collect-data/data/fmsinfo.pickle', 'rb'))
nstates = fmsinfo['nstates']
ics = [x for x in range(11,50)]

for ic in ics:
    plot(ic, dihedral_data, nstates)
