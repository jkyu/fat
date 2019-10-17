import sys
import os
import numpy as np
import pickle
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')

def plot():

    data = pickle.load(open('./data/dihedrals-avgs.pickle', 'rb'))
    tgrid = data['tgrid']
    
    all_dihedrals = data['all_dihedrals']
    dihedral_names = data['dihedral_names']
    d1112 = dihedral_names[0]
    d1314 = dihedral_names[1]
    d15nz = dihedral_names[2]

    avg_dihedrals_ex = data['avg_dihedrals_ex']
    error_ex = data['error_ex']
    ex_keys = data['ex_keys']

    avg_dihedrals_gs = data['avg_dihedrals_gs']
    error_gs = data['error_gs']
    gs_keys = data['gs_keys']

    cis_keys = data['cis_keys']
    trans_keys = data['trans_keys']

    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    labelsize = 13
    ticksize = 11
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    p1 = plt.subplot(1,2,1)
    p2 = plt.subplot(1,2,2)
    
    p1.errorbar(tgrid, avg_dihedrals_ex[d1112], yerr=error_ex[d1112], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='orchid', ecolor='violet', label='$\\alpha$')
    p1.errorbar(tgrid, avg_dihedrals_ex[d1314], yerr=error_ex[d1314], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='darkorange', ecolor='lightsalmon', label='$\\beta$')
    p1.errorbar(tgrid, avg_dihedrals_ex[d15nz], yerr=error_ex[d15nz], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='slateblue', ecolor='lightsteelblue', label='$\\gamma$')
    p1.axis([0, 1400, 150, 280])
    p1.set_ylabel('Dihedral Angle', fontsize=labelsize)
    # plt.ylabel('Dihedral Angle', fontsize=labelsize)
    # plt.xlabel('Time (fs)', fontsize=labelsize)
    p1.legend(loc='best', frameon=False, fontsize=ticksize, ncol=3)
    
    p2.errorbar(tgrid, avg_dihedrals_gs[d1112], yerr=error_gs[d1112], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='orchid', ecolor='violet', label='$\\alpha$')
    p2.errorbar(tgrid, avg_dihedrals_gs[d1314], yerr=error_gs[d1314], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='darkorange', ecolor='lightsalmon', label='$\\beta$')
    p2.errorbar(tgrid, avg_dihedrals_gs[d15nz], yerr=error_gs[d15nz], linewidth=0.8, capsize=0.6, elinewidth=0.3, color='slateblue', ecolor='lightsteelblue', label='$\\gamma$')
    p2.axis([0, 250, 110, 360])
    p2.set_ylabel('Dihedral Angle', fontsize=labelsize)
    p2.set_xlabel('.', color=(0, 0, 0, 0))
    p2.legend(loc='best', frameon=False, fontsize=ticksize)

    fig.text(0.5, 0.03, 'Time [fs]', va='center', ha='center', fontsize=labelsize)

    plt.savefig('./figures/avg-dihedrals.pdf')
    plt.savefig('./figures/avg-dihedrals.png')

plot()
