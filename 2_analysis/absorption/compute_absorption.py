#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys
import os

def parse_casci(dirname, nstates, enefile='ens.txt', oscfile='osc.txt'):

    os.system('bash dump_energies.sh %s %d %s %s' %(dirname, nstates, enefile, oscfile))

    nex = nstates - 1 # exclude the ground state; only excited states
    osc = np.loadtxt(oscfile)
    ene = np.loadtxt(enefile)

    osc2 = np.reshape(osc, (len(osc)//nex, nex))
    ene2 = np.reshape(ene, (len(ene)//nex, nex))

    ''' figure out which state is the bright state '''
    osc3 = []
    ene3 = []
    for i in range(np.shape(osc2)[0]):
        ind = np.argmax(osc2[i,:])
        osc3.append(osc2[i, ind])
        ene3.append(ene2[i, ind])
    ene3 = np.array(ene3)
    osc3 = np.array(osc3)

    return ene3, osc3

def broaden(x, en, osc, sigma=0.15):
    # Absorption curve is a lorentzian broadening to each excitation energy multiplied by oscillator strength
    # Pass in a different argument for sigma if the lorentzian width is not appropriate
    return 0.5 * sigma / np.pi * 1.0 / ((x-en)**2 + (0.5 * sigma)**2)

def compute_spectra(ene, osc, grid): 
    
    spectrum = np.zeros_like(grid)
    ''' Compute contribution of each geometry in the wigner sample after braodening '''
    for i in range(len(ene)):
        abs_contribution = broaden(grid, ene[i], osc[i])
        spectrum += abs_contribution
    spectrum_max = np.max(spectrum)
    spectrum = spectrum / spectrum_max
    
    return spectrum

def plot_spectrum(grid, spectrum, figname='absorption'):

    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    legendsize = 12

    plt.plot(grid, spectrum, linewidth=2.0,color='k')
    plt.ylabel('Absorption Cross Section')
    plt.xlabel('$\Delta E$ [eV]')
    plt.ylim([0, 1.1])
    plt.xlim([np.min(grid), np.max(grid)])
    if not os.path.isdir('./figures/'):
        os.mkdir('./figures/')
    plt.savefig('./figures/%s.pdf' %figname, dpi=300)

if __name__=='__main__':

    en_min = 2
    en_max = 6.5
    npoints = 2000
    grid = np.linspace(en_min, en_max, npoints) 

    nstates = 3
    dirname = 'directory of single point calculations for the absorption spectrum'
    ene, osc = parse_casci(dirname, nstates)

    spectrum = compute_spectra(ene, osc, grid)
    data = { 'grid' : grid, 'absorption' : spectrum }
    print('Saving absorption data.')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/%s.pickle' %('absorption'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # np.savez('absorption.npz', grid=grid, absorption=spectrum)
    plot_spectrum(grid, spectrum)
