import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import numpy as np

''' Plot the TBF-averaged angle over time. This is ripped off from the dihedral 
plotting code. '''

def plot(dihedral_data, figname='avg-angles'):

    tbf_keys = angle_data['tbf_keys']

    angle_keys = angle_data['angle_names']
    state_ids = angle_data['state_ids']
    angles = angle_data['angles_state_specific']
    tgrid = angle_data['tgrid']

    # plot only S1 TBFs
    ic_keys = [ x for x in tbf_keys if int(x.split('-')[1])==1 ]
    tbf_keys = ic_keys

    ''' For each angle of interest (angle_keys), the angles in time are stored.
    Average over the TBFs, so we have the TBF-averaged change in each angle over time. '''
    avg_angles = {} 
    for angle_key in angle_keys: 
        all_tbfs = np.zeros((len(tbf_keys), len(tgrid)))
        for i, tbf_key in enumerate(tbf_keys):
            all_tbfs[i, :] = angles[tbf_key][angle_key] # big array of angles indexed by tbf number
        ma_tbfs   = np.ma.MaskedArray(all_tbfs, mask=np.isnan(all_tbfs)) 
        avg_tbfs  = np.ma.average(ma_tbfs, axis=0) # averaged angle angle over all TBFs 
        avg_angles[angle_key] = avg_tbfs # store tbf-averaged angle angles to dictionary

    fig = plt.figure(figsize=(6,5))
    labelsize = 14
    ticksize = 12
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    p1 = plt.subplot(1,1,1)

    for i, angle_key in enumerate(angle_keys):
        avg_ang = avg_angles[angle_key]
        p1.plot(tgrid, avg_ang, linestyle='-', color=colors[i], label=angle_key)
    p1.legend(loc='best', frameon=False, fontsize=ticksize)
    # p1.axis([tgrid[0], tgrid[-1], 180, 270])
    p1.set_ylabel('Angle', fontsize=labelsize)
    p1.set_xlabel('Time [fs]', fontsize=labelsize)

    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/%s.pdf' %figname, dpi=300)
    plt.close()

rcParams.update({'figure.autolayout': True})
angle_data = pickle.load(open('./data/angles.pickle', 'rb'))
figname = 'avg-angles'
plot(angle_data, figname=figname)
