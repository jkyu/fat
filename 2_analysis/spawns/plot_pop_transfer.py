from shutil import copyfile
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

def get_populations(ics, datadir, fmsdir, topfile):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    for i in range(nstates):
        states['s%d' %i] = []
    meci_keys = np.load('./meci-classification.npz')
    tp_keys = meci_keys['tp_keys']
    eth_keys = meci_keys['eth_keys']

    tp_pop = 0
    eth_pop = 0
    tp_pop_list = []
    eth_pop_list = []
    tp_gap_list = []
    eth_gap_list = []
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        ic_pop = 0
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']
            if tbf_state==0:
                parent_id = tbf['spawn_info']['parent_id']
                if parent_id == 1:

                    states['s%d' %tbf_state].append(tbf_key)
                    pop = tbf['populations']
                    if len(pop) > 20:
                        pop_transferred = pop[20]
                        s0 = tbf['energies']['s0'][:20]
                        s1 = tbf['energies']['s1'][:20]
                        min_gap = np.min(np.abs(s1-s0)) * 27.21138602
                    else: 
                        pop_transferred = pop[-1]
                        s0 = tbf['energies']['s0'][:-1]
                        s1 = tbf['energies']['s1'][:-1]
                        min_gap = np.min(np.abs(s1-s0)) * 27.21138602
                    ic_pop = ic_pop + pop_transferred

                    print('%s: %f' %(tbf_key, pop_transferred))

                    geomfile = fmsdir+'/%04d/Spawn.%d' %(ic, tbf['tbf_id'])
                    if tbf_key in tp_keys:
                        tp_pop = tp_pop + pop_transferred
                        tp_gap_list.append(min_gap)
                        tp_pop_list.append(pop_transferred)
                        # if not os.path.isdir('./tp'):
                        #     os.mkdir('./tp')
                        # copyfile(geomfile, './tp/%s.xyz' %(tbf_key))
                    elif tbf_key in eth_keys:
                        eth_pop = eth_pop + pop_transferred
                        eth_gap_list.append(min_gap)
                        eth_pop_list.append(pop_transferred)
                        # if not os.path.isdir('./eth'):
                        #     os.mkdir('./eth')
                        # copyfile(geomfile, './eth/%s.xyz' %(tbf_key))

        print('Total transferred: ', ic_pop)
        print()
    print('Total twisted-pyramidal: ', tp_pop / (tp_pop + eth_pop))
    print('Total ethylidene: ', eth_pop / (tp_pop + eth_pop))

    return np.array(tp_pop_list), np.array(eth_pop_list), np.array(tp_gap_list), np.array(eth_gap_list)

def plot_pop_transfer_energy_gap(tp_pops, eth_pops, tp_gaps, eth_gaps):

    # Make bins from energy gaps
    nbins = 9
    tp_bins = np.zeros(nbins)
    eth_bins = np.zeros(nbins)

    total_pop_transfer = np.sum(tp_pops) + np.sum(eth_pops)

    # Since our bins start at an energy gap of 0 eV and have a max of <1 eV
    # can just multiply by 10 to get index of the array to add the population
    # transferred. This means the bins have a spacing of 0.1 eV
    for pop, gap in zip(tp_pops, tp_gaps):
        tp_bins[int(np.floor(gap*10))] += pop 
    for pop, gap in zip(eth_pops, eth_gaps):
        eth_bins[int(np.floor(gap*10))] += pop 

    print(tp_bins)
    print(eth_bins)
    print(tp_bins/total_pop_transfer)
    print(eth_bins/total_pop_transfer)
        
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    width = 0.035
    x_labels = np.arange(nbins)*0.1
    tp_bars = plt.bar(x_labels - width/2, tp_bins/total_pop_transfer, width, label='Twisted-Pyramidalized CI', color='darkslateblue')
    eth_bars = plt.bar(x_labels + width/2, eth_bins/total_pop_transfer, width, label='H-Migration CI', color='mediumvioletred')

    plt.ylabel('Fractional Population Transferred', fontsize=labelsize)
    plt.xlabel('Minimum Energy Gap [eV]', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

    plt.savefig('./figures/pop-transfer.pdf', dpi=300)
    plt.close()

    fig = plt.figure(figsize=(3,3))
    plt.pie([np.sum(tp_bins)/total_pop_transfer, np.sum(eth_bins)/total_pop_transfer], autopct='%1.2f%%', colors=['darkslateblue', 'mediumvioletred'], textprops=dict(color='white', size=16, weight='bold'))
    plt.axis('equal')
    plt.savefig('./figures/pie.pdf', dpi=300)


    
''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
fmsdir = '../../../'
topfile = '../../../ethylene.pdb'
tp_pops, eth_pops, tp_gaps, eth_gaps = get_populations(ics, datadir, fmsdir, topfile)
plot_pop_transfer_energy_gap(tp_pops, eth_pops, tp_gaps, eth_gaps)
