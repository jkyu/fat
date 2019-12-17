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

    data = pickle.load(open('./data/hbonds-avg.pickle', 'rb'))
    tgrid = data['tgrid']
    avg_hbonds_ex = data['avg_hbonds_ex']
    avg_hbonds_cis = data['avg_hbonds_cis']
    avg_hbonds_trans = data['avg_hbonds_trans']
    error_ex = data['error_ex']
    error_cis = data['error_cis']
    error_trans = data['error_trans']

    rcParams.update({'figure.autolayout': True})
    # fig = plt.figure(figsize=(6,4))
    # labelsize = 16
    # ticksize = 14
    # plt.rc('xtick',labelsize=ticksize)
    # plt.rc('ytick',labelsize=ticksize)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 3))
    labelsize = 13
    ticksize = 11
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    p1 = plt.subplot(1,2,1)
    p2 = plt.subplot(1,2,2)
    
    p1.errorbar(tgrid, avg_hbonds_ex, yerr=error_ex, linewidth=1.5, capsize=0.6, elinewidth=0.3, color='firebrick', ecolor='lightcoral') #, label='Excited State')
    p1.axis([0, 1400, 2.50, 2.65])
    p1.set_yticks([2.50, 2.55, 2.60, 2.65])
    p1.set_ylabel('Wat402-$\mathrm{N_Z}$ Distance [$\mathrm{\AA}$]', fontsize=labelsize)
    # plt.ylabel('Dihedral Angle', fontsize=labelsize)
    p1.set_xlabel('Time After Photoexcitation [fs]', fontsize=labelsize)
    p1.legend(loc='best', frameon=False, fontsize=ticksize, ncol=3)
    
    p2.errorbar(tgrid, avg_hbonds_cis, yerr=error_cis, linewidth=1.5, capsize=0.6, elinewidth=0.3, color='steelblue', ecolor='lightblue', linestyle='--') #, label='cis')
    p2.errorbar(tgrid, avg_hbonds_trans, yerr=error_trans, linewidth=1.5, capsize=0.6, elinewidth=0.3, color='steelblue', ecolor='lightblue', linestyle=':') #, label='trans')
    p2.axis([0, 700, 2.45, 2.85])
    # p2.set_ylabel('Wat402-Schiff Base Distance [$\mathrm{\AA}$]', fontsize=labelsize)
    # p2.set_ylabel('Wat402-Schiff Base Distance [$\mathrm{\AA}$]', fontsize=labelsize)
    p2.set_xlabel('Time After Internal Conversion [fs]', fontsize=labelsize)
    # p2.set_xlabel('.', color=(0, 0, 0, 0))
    p2.legend(loc='best', frameon=False, fontsize=ticksize)

    # fig.text(0.5, 0.03, 'Time [fs]', va='center', ha='center', fontsize=labelsize)
    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    plt.savefig('./figures/avg-hbonds.pdf')
    plt.savefig('./figures/avg-hbonds.png')

plot()
