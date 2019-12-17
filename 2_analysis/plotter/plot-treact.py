import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import numpy as np

''' Plots the averaged dynamics over all TBFs centered at the first spawning point.
Helps to see what happens in the region of nonadiabatic transitions. 
Obviously can only run this after already having computed dihedrals and BLAs '''

def smooth_data(x, y, npoints):

    # # Uses spline interpolation between points to generate a smoother plot
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.linspace(x[0], x[-1], npoints)
    ynew = interpolate.splev(xnew, tck, der=0)

    return xnew, ynew 

def treact(en_data, angle_data, bla_data, offset=78):

    '''
    Plot excited state trajectories
    '''
    ex_keys = bla_data['ex_keys']
    gs_keys = bla_data['gs_keys']
    dihe_keys = angle_data['dihedral_names']
    time = angle_data['tgrid']

    ''' Get first spawn point '''
    spawn_tbf_keys = [x for x in gs_keys if x.split('-')[1]=='02']
    tspawn_inds = []
    spawn_ics = []
    for key in spawn_tbf_keys:
        spawn_ics.append(key.split('-')[0])
        s0_energy = en_data['s0_energies'][key]
        tspawn_ind = np.min(np.argwhere(np.isfinite(s0_energy)))
        tspawn_inds.append(tspawn_ind)
    ex_keys = [x for x in ex_keys if x.split('-')[0] in spawn_ics]

    ''' Load data for excited state TBFs '''
    npoints = 151
    all_gaps  = np.zeros((len(ex_keys), npoints))
    all_blas  = np.zeros((len(ex_keys), npoints))
    all_d1112 = np.zeros((len(ex_keys), npoints))
    all_d1314 = np.zeros((len(ex_keys), npoints))
    all_d15nz = np.zeros((len(ex_keys), npoints))

    for i, key in enumerate(ex_keys):
        s0_energy = en_data['s0_energies'][key]
        s1_energy = en_data['s1_energies'][key]
        en_gap = s1_energy - s0_energy
        d1112 = angle_data['all_dihedrals'][key][dihe_keys[0]]
        d1314 = angle_data['all_dihedrals'][key][dihe_keys[1]]
        d15nz = angle_data['all_dihedrals'][key][dihe_keys[2]]
        bla = bla_data['blas'][key]

        tstart = tspawn_inds[i] - 15
        tend = tstart + 30
        tgrid, all_gaps[i,:] = smooth_data(np.linspace(0, 30*5, 30), en_gap[tstart:tend], npoints)
        _, all_blas[i,:]     = smooth_data(np.linspace(0, 30*5, 30), bla[tstart:tend], npoints)
        _, all_d1112[i,:]    = smooth_data(np.linspace(0, 30*5, 30), d1112[tstart:tend], npoints)
        _, all_d1314[i,:]    = smooth_data(np.linspace(0, 30*5, 30), d1314[tstart:tend], npoints)
        _, all_d15nz[i,:]    = smooth_data(np.linspace(0, 30*5, 30), d15nz[tstart:tend], npoints)

    ''' Average over trajectories accounting for nan values '''
    ma_gaps   = np.ma.MaskedArray(all_gaps, mask=np.isnan(all_gaps))
    avg_gaps  = np.ma.average(ma_gaps, axis=0)
    ma_blas   = np.ma.MaskedArray(all_blas, mask=np.isnan(all_blas))
    avg_blas  = np.ma.average(ma_blas, axis=0)
    ma_d1112  = np.ma.MaskedArray(all_d1112, mask=np.isnan(all_d1112))
    avg_d1112 = np.ma.average(ma_d1112, axis=0)
    ma_d1314  = np.ma.MaskedArray(all_d1314, mask=np.isnan(all_d1314))
    avg_d1314 = np.ma.average(ma_d1314, axis=0)
    ma_d15nz  = np.ma.MaskedArray(all_d15nz, mask=np.isnan(all_d15nz))
    avg_d15nz = np.ma.average(ma_d15nz, axis=0)

    ''' Plot '''
    fig = plt.figure(figsize=(6,6))
    labelsize = 13
    ticksize = 11
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    p1 = plt.subplot(3,1,1)
    p2 = plt.subplot(3,1,2)
    p3 = plt.subplot(3,1,3)

    tgrid = tgrid - tgrid[offset] # center at spawning point
    spawn = tgrid[offset]

    p1.plot(tgrid, avg_gaps, linestyle='-', color='black', label='S1')
    p1.scatter(spawn, avg_gaps[offset], color='red')

    p2.plot(tgrid, avg_d1112, linestyle='-', color='orchid', label='$\\alpha$')
    p2.plot(tgrid, avg_d1314, linestyle='-', color='darkorange', label='$\\beta$')
    p2.plot(tgrid, avg_d15nz, linestyle='-', color='slateblue', label='$\\gamma$')
    p2.scatter(spawn, avg_d1112[offset], color='red') #, label='Spawn Point')
    p2.scatter(spawn, avg_d1314[offset], color='red')
    p2.scatter(spawn, avg_d15nz[offset], color='red')

    p3.plot(tgrid, avg_blas, linestyle='-', color='black')
    p3.scatter(spawn, avg_blas[offset], color='red', label='Spawn Point')


    '''
    Plot CIS photoproduct ground state trajectories.
    '''
    # cis_keys = [x for x in angle_data['cis_keys'] if x.split('-')[1]=='02']
    # trans_keys = [x for x in angle_data['trans_keys'] if x.split('-')[1]=='02']
    cis_keys = angle_data['cis_keys']
    trans_keys = angle_data['trans_keys']

    ''' Load data for S0, cis TBFs '''
    npoints = 76
    all_gaps  = np.zeros((len(cis_keys), npoints))
    all_blas  = np.zeros((len(cis_keys), npoints))
    all_d1112 = np.zeros((len(cis_keys), npoints))
    all_d1314 = np.zeros((len(cis_keys), npoints))
    all_d15nz = np.zeros((len(cis_keys), npoints))

    for i, key in enumerate(cis_keys):
        s0_energy = en_data['s0_energies'][key]
        s1_energy = en_data['s1_energies'][key]
        en_gap = s1_energy - s0_energy
        d1112 = angle_data['all_dihedrals'][key][dihe_keys[0]]
        d1314 = angle_data['all_dihedrals'][key][dihe_keys[1]]
        d15nz = angle_data['all_dihedrals'][key][dihe_keys[2]]
        bla = bla_data['blas'][key]

        tstart = np.min(np.argwhere(np.isfinite(s0_energy)))
        tend = tstart + 15
        tgrid, all_gaps[i,:] = smooth_data(np.linspace(0, 15*5, 15), en_gap[tstart:tend], npoints)
        _, all_blas[i,:]     = smooth_data(np.linspace(0, 15*5, 15), bla[tstart:tend], npoints)
        _, all_d1112[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d1112[tstart:tend], npoints)
        _, all_d1314[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d1314[tstart:tend], npoints)
        _, all_d15nz[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d15nz[tstart:tend], npoints)

    ''' Average over trajectories accounting for nan values '''
    ma_gaps   = np.ma.MaskedArray(all_gaps, mask=np.isnan(all_gaps))
    avg_gaps  = np.ma.average(ma_gaps, axis=0)
    ma_blas   = np.ma.MaskedArray(all_blas, mask=np.isnan(all_blas))
    avg_blas  = np.ma.average(ma_blas, axis=0)
    ma_d1112  = np.ma.MaskedArray(all_d1112, mask=np.isnan(all_d1112))
    avg_d1112 = np.ma.average(ma_d1112, axis=0)
    ma_d1314  = np.ma.MaskedArray(all_d1314, mask=np.isnan(all_d1314))
    avg_d1314 = np.ma.average(ma_d1314, axis=0)
    ma_d15nz  = np.ma.MaskedArray(all_d15nz, mask=np.isnan(all_d15nz))
    avg_d15nz = np.ma.average(ma_d15nz, axis=0)

    p1.plot(tgrid, avg_gaps, linestyle='--', color='black', label='S0 (cis)')
    p2.plot(tgrid, avg_d1112, linestyle='--', color='orchid')
    p2.plot(tgrid, avg_d1314, linestyle='--', color='darkorange')
    p2.plot(tgrid, avg_d15nz, linestyle='--', color='slateblue')
    p3.plot(tgrid, avg_blas, linestyle='--', color='black')


    '''
    Plot TRANS photoproduct ground state trajectories
    '''
    npoints = 76
    all_gaps  = np.zeros((len(trans_keys), npoints))
    all_blas  = np.zeros((len(trans_keys), npoints))
    all_d1112 = np.zeros((len(trans_keys), npoints))
    all_d1314 = np.zeros((len(trans_keys), npoints))
    all_d15nz = np.zeros((len(trans_keys), npoints))

    for i, key in enumerate(trans_keys):
        s0_energy = en_data['s0_energies'][key]
        s1_energy = en_data['s1_energies'][key]
        en_gap = s1_energy - s0_energy
        d1112 = angle_data['all_dihedrals'][key][dihe_keys[0]]
        d1314 = angle_data['all_dihedrals'][key][dihe_keys[1]]
        d15nz = angle_data['all_dihedrals'][key][dihe_keys[2]]
        bla = bla_data['blas'][key]

        tstart = np.min(np.argwhere(np.isfinite(s0_energy)))
        tend = tstart + 15
        tgrid, all_gaps[i,:] = smooth_data(np.linspace(0, 15*5, 15), en_gap[tstart:tend], npoints)
        _, all_blas[i,:]     = smooth_data(np.linspace(0, 15*5, 15), bla[tstart:tend], npoints)
        _, all_d1112[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d1112[tstart:tend], npoints)
        _, all_d1314[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d1314[tstart:tend], npoints)
        _, all_d15nz[i,:]    = smooth_data(np.linspace(0, 15*5, 15), d15nz[tstart:tend], npoints)

    ''' Average over trajectories accounting for nan values '''
    ma_gaps   = np.ma.MaskedArray(all_gaps, mask=np.isnan(all_gaps))
    avg_gaps  = np.ma.average(ma_gaps, axis=0)
    ma_blas   = np.ma.MaskedArray(all_blas, mask=np.isnan(all_blas))
    avg_blas  = np.ma.average(ma_blas, axis=0)
    ma_d1112  = np.ma.MaskedArray(all_d1112, mask=np.isnan(all_d1112))
    avg_d1112 = np.ma.average(ma_d1112, axis=0)
    ma_d1314  = np.ma.MaskedArray(all_d1314, mask=np.isnan(all_d1314))
    avg_d1314 = np.ma.average(ma_d1314, axis=0)
    ma_d15nz  = np.ma.MaskedArray(all_d15nz, mask=np.isnan(all_d15nz))
    avg_d15nz = np.ma.average(ma_d15nz, axis=0)

    p1.plot(tgrid, avg_gaps, linestyle=':', color='black', label='S0 (trans)')
    p1.axis([-72, 72, 0, 3.6])
    p1.set_ylabel('E$_{\mathrm{S1}}$ - E$_{\mathrm{S0}}$ [eV]', fontsize=labelsize)
    p1.legend(loc='upper left', frameon=False, fontsize=ticksize)

    p2.plot(tgrid, avg_d1112, linestyle=':', color='orchid')
    p2.plot(tgrid, avg_d1314, linestyle=':', color='darkorange')
    p2.plot(tgrid, avg_d15nz, linestyle=':', color='slateblue')
    p2.axis([-72, 72, 120, 360])
    p2.set_ylabel('Dihedral Angle', fontsize=labelsize)
    p2.legend(loc='best', frameon=False, fontsize=ticksize)

    p3.plot(tgrid, avg_blas, linestyle=':', color='black')
    p3.axis([-72, 72, -0.1, 0.12])
    p3.set_ylabel('BLA [$\mathrm{\AA}$]', fontsize=labelsize)
    p3.legend(loc='best', frameon=False, fontsize=ticksize)
    p3.set_xlabel('t$_\mathrm{react}$ [fs]', fontsize=labelsize)

    fig.align_ylabels()

    if not os.path.isdir('./figures'):
        os.mkdir('./figures')
    plt.savefig('./figures/treact.pdf')
    # plt.savefig('test.png')
    plt.close()


rcParams.update({'figure.autolayout': True})
en_data = pickle.load(open('../energies/data/energies.pickle', 'rb'))
angle_data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
bla_data = pickle.load(open('../bond-lengths/data/bla.pickle', 'rb'))

treact(en_data, angle_data, bla_data)
