import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import numpy as np

def plot(ic, en_data, angle_data, bla_data):

    ex_keys = bla_data['ex_keys']
    gs_keys = bla_data['gs_keys']

    ic_str = '%02d' %ic
    ic_tbf_key = [x for x in ex_keys if x.split('-')[0]==ic_str]
    spawn_tbf_keys = [x for x in gs_keys if x.split('-')[0]==ic_str]
    keys = ic_tbf_key + spawn_tbf_keys

    dihe_keys = angle_data['dihedral_names']
    time = en_data['tgrid']

    fig = plt.figure(figsize=(4,8))
    labelsize = 14
    ticksize = 12
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    p1 = plt.subplot(3,1,1)
    p2 = plt.subplot(3,1,2)
    p3 = plt.subplot(3,1,3)

    ''' Plot for IC '''
    key = ic_tbf_key[0]
    s0_energy = en_data['s0_energies'][key]
    s1_energy = en_data['s1_energies'][key]
    en_gap = s1_energy - s0_energy
    d1112 = angle_data['all_dihedrals'][key][dihe_keys[0]]
    d1314 = angle_data['all_dihedrals'][key][dihe_keys[1]]
    d15nz = angle_data['all_dihedrals'][key][dihe_keys[2]]
    bla = bla_data['blas'][key]

    tmax_ind = np.max(np.argwhere(np.isfinite(s0_energy)))
    if len(spawn_tbf_keys) == 0:
        pass
    elif tmax_ind+25 < len(time):
        tmax_ind = tmax_ind+25

    p1.plot(time, en_gap, linestyle='-', color='black', label='S1 TBF')
    p2.plot(time, d1112, linestyle='-', color='orchid', label='$\\alpha$')
    p2.plot(time, d1314, linestyle='-', color='darkorange', label='$\\beta$')
    p2.plot(time, d15nz, linestyle='-', color='slateblue', label='$\\gamma$')
    p3.plot(time, bla, linestyle='-', color='black', label='S1 TBF')

    count = 0
    for key in spawn_tbf_keys:

        s0_energy = en_data['s0_energies'][key]
        s1_energy = en_data['s1_energies'][key]
        en_gap = s1_energy - s0_energy
        d1112 = angle_data['all_dihedrals'][key][dihe_keys[0]]
        d1314 = angle_data['all_dihedrals'][key][dihe_keys[1]]
        d15nz = angle_data['all_dihedrals'][key][dihe_keys[2]]
        bla = bla_data['blas'][key]

        # figure out where to cut off the time scale
        # tmax_ind_new = np.max(np.argwhere(np.isfinite(s0_energy)))
        # if tmax_ind < tmax_ind_new:
        #     tmax_ind = tmax_ind_new
        spawn_ind = np.min(np.argwhere(np.isfinite(s0_energy)))

        if count == 0: 
            label1 = 'S0 TBF (J)'
            label2 = 'Spawn Point'
        else: 
            label1 = 'S0 TBF (bR)'
            label2 = None

        if count==0:
            p1.plot(time, en_gap, linestyle='--', color='black', label=label1)
            p2.plot(time, d1112, linestyle='--', color='orchid')
            p2.plot(time, d1314, linestyle='--', color='darkorange')
            p2.plot(time, d15nz, linestyle='--', color='slateblue')
            p3.plot(time, bla, linestyle='--', color='black')
        else:
            p1.plot(time, en_gap, linestyle=':', color='black', label=label1)
            p2.plot(time, d1112, linestyle=':', color='orchid')
            p2.plot(time, d1314, linestyle=':', color='darkorange')
            p2.plot(time, d15nz, linestyle=':', color='slateblue')
            p3.plot(time, bla, linestyle=':', color='black')
        p1.scatter(time[spawn_ind], en_gap[spawn_ind], color='red')
        p2.scatter(time[spawn_ind], d1112[spawn_ind], color='red')
        p2.scatter(time[spawn_ind], d1314[spawn_ind], color='red')
        p2.scatter(time[spawn_ind], d15nz[spawn_ind], color='red')
        p3.scatter(time[spawn_ind], bla[spawn_ind], color='red', label=label2)

        count += 1

    p1.axis([time[0], time[tmax_ind], 0, 4.0])
    p1.set_ylabel('E$_{\mathrm{S1}}$ - E$_{\mathrm{S0}}$ [$\Delta$eV]', fontsize=labelsize)
    p1.legend(loc='best', frameon=False, fontsize=ticksize, ncol=3)

    p2.axis([time[0], time[tmax_ind], 120, 380])
    p2.set_ylabel('Dihedral Angle', fontsize=labelsize)
    p2.legend(loc='best', frameon=False, fontsize=ticksize, ncol=3)

    p3.axis([time[0], time[tmax_ind], -0.15, 0.15])
    p3.set_ylabel('BLA [$\mathrm{\AA}$]', fontsize=labelsize)
    # p3.legend(loc='best', frameon=False, fontsize=ticksize)
    p3.set_xlabel('Time [fs]', fontsize=labelsize)

    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/%02d.pdf' %ic)
    plt.close()

rcParams.update({'figure.autolayout': True})
en_data = pickle.load(open('../energies/data/energies.pickle', 'rb'))
angle_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
bla_data = pickle.load(open('../bond-lengths/data/bla.pickle', 'rb'))
ics = [4,5]

for ic in ics:
    print(ic)
    plot(ic, en_data, angle_data, bla_data)

