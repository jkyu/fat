from shutil import copyfile
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

def get_populations(ics, datadir, state_from, state_to):
    ''' Get population data for transfer events from state_from to state_to. '''

    print('Loading excited state trajectories and extracting population information.')
    absolute_transfer = {}
    relative_transfer = {}
    energy_gaps = {}

    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        total_ic_transfer = 0
        for tbf_key in data.keys():
            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']
            if tbf_state==state_to:
                parent_state = tbf['spawn_info']['parent_state']
                if parent_state==state_from:

                    # Get the population of the parent TBF at the time of spawning
                    parent_key = '%04d-%04d' %(ic, tbf['spawn_info']['parent_id'])
                    parent_tbf = data[parent_key]
                    spawn_time = tbf['spawn_info']['spawn_time']
                    parent_tbf_times = parent_tbf['time_steps']
                    spawn_ind_parent = np.argmin(np.abs(parent_tbf_times - spawn_time))
                    parent_pop = parent_tbf['populations'][spawn_ind_parent]

                    # Get the population of the current TBF after population exchange has stabilized
                    tbf_pop = tbf['populations']
                    if len(tbf_pop) > 20:
                        pop_transferred = tbf_pop[20]
                        upper = tbf['energies']['s%d' %state_from][:20]
                        lower = tbf['energies']['s%d' %state_to][:20]
                        min_gap = np.min(np.abs(upper-lower)) * 27.21138602
                    else: 
                        pop_transferred = tbf_pop[-1]
                        upper = tbf['energies']['s%d' %state_from][:-1]
                        lower = tbf['energies']['s%d' %state_to][:-1]
                        min_gap = np.min(np.abs(upper-lower)) * 27.21138602

                    if parent_pop > pop_transferred: 
                        transfer_efficiency = pop_transferred / parent_pop
                        total_ic_transfer += pop_transferred
                        absolute_transfer[tbf_key] = pop_transferred
                        relative_transfer[tbf_key] = transfer_efficiency
                        energy_gaps[tbf_key] = min_gap

        print('Total transferred: ', total_ic_transfer)
        print()

    return absolute_transfer, relative_transfer, energy_gaps

def plot_transfer(absolute_transfer, relative_transfer, energy_gaps, bin_spacing=0.05):

    tbf_keys = [ x for x in absolute_transfer.keys() ]
    all_populations = [ absolute_transfer[key] for key in tbf_keys ]
    all_efficiencies = [ relative_transfer[key] for key in tbf_keys ]
    all_energy_gaps = [ energy_gaps[key] for key in tbf_keys ]

    nbins = int( np.ceil(np.max(all_energy_gaps) / bin_spacing) )

    # Compute fractional population transferred from absolute transfer
    abs_bins = np.zeros(nbins)
    total_population_transfer = np.sum(all_populations)
    for pop, gap in zip(all_populations, all_energy_gaps):
        abs_bins[ int(np.floor(gap/bin_spacing)) ] += pop
    abs_bins = abs_bins / total_population_transfer

    # Compute average efficiency for each gap
    rel_bins = {}
    for b in range(nbins):
        rel_bins['%d' %b] = []
    for pop, gap in zip(all_efficiencies, all_energy_gaps):
        rel_bins['%d' %(int(np.floor(gap/bin_spacing)))].append(pop)
    rel_bins2 = np.zeros(nbins)
    for b in range(nbins):
        nevents = len(rel_bins['%d' %b])
        rel_bins2[b] = np.sum(rel_bins['%d' %b]) / nevents
    # If fractional transfer is small in a bin, its efficiency is more or less an artifact,
    # so zero it out. Threshold set at 0.01. 
    rel_bins = [ rel_bins2[b] if abs_bins[b]>0.01 else 0 for b in range(nbins) ]

    rcParams.update({'figure.autolayout': True})
    fig, ax1 = plt.subplots(figsize=(6,5))
    labelsize = 14
    ticksize = 12
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)
    
    width = 0.3 * bin_spacing
    x_labels = np.arange(nbins)*bin_spacing
    abs_bars = ax1.bar(x_labels - width/2, abs_bins, width, label='Absolute Population Transfer', color='darkslateblue')
    ax1.set_ylabel('Fractional Population Transferred', fontsize=labelsize, color='darkslateblue')
    ax1.set_xlabel('Minimum Energy Gap [eV]', fontsize=labelsize, color='black')
    ax1.set_xticks([ bin_spacing*x for x in range(0, nbins) ])
    ax1.tick_params('y', colors='darkslateblue', labelsize=ticksize)
    ax1.tick_params('x', colors='black', labelsize=ticksize)

    ax2 = ax1.twinx()
    rel_bars = ax2.bar(x_labels + width/2, rel_bins, width, label='Population Transfer Efficiency', color='mediumvioletred')
    ax2.set_ylabel('Relative Population Transferred', fontsize=labelsize, color='mediumvioletred', rotation=270, labelpad=20)
    ax2.tick_params('y', colors='mediumvioletred', labelsize=ticksize)

    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/population_transfer.pdf', dpi=300)
    plt.close()

''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1_collect_data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
absolute_transfer, relative_transfer, energy_gaps = get_populations(ics, datadir, state_from=1, state_to=0)
plot_transfer(absolute_transfer, relative_transfer, energy_gaps)
