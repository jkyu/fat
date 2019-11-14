import numpy as np
import os
import math
import sys
import pickle
'''
Fluorescence intensity at a given emission frequency I(t,v) is related to the 
population on S1, the transition dipole moment between S1 and S0 and the emission frequency. 
The latter two are obtained from EST calculations at the center  of each TBF. 
I need the following for each TBF on S1:
(1) population on the Ith TBF where I is the initial condition, so we need S1 populations - N.dat
(2) squared transition dipole moment between S1 and S0 at the center of the Gaussian wavepacket 
for the Ith TBF - TDip.x
(3) Electronic energy of S1 and S0 at the center of the Gaussian wavepacket - PotEn.x
This is done by the function collect_trajectories() and the helper functions it calls in turn.
'''
## Helper functions for computing fluorescence intensity ##

def gaussian_convolution(egrid, tgrid, e0, t0, intensity, sigE=0.1, sigT=30):
    '''
    To convolve the intensity in time and energy, multiply the intensity
    (a delta function) by gaussians, one centered on the transition 
    frequency (written as an energy in units of eV) and the second
    centered on the adaptive time step. Here, e0 and t0 are the centers
    of these two gaussians respectively and egrid and tgrid are the grid points 
    that we are using for our fluorescence plot. sigE and sigT are the
    gaussian width parameters. The width = stdev = sqrt(variance).
    The energy width is set to 0.15 eV (from Punwong paper) and the time width is 
    150 fs (from Schmidt et al.) as default parameters for the convolution. 
    Note that the Gaussian width of the laser pulse is measured as the
    full-width half maximum (FWHM), where FWHM = 2 * sigma * sqrt(2 * ln 2) 
    => FWHM ~= 2.3548200450309493 * sigma. Therefore, we choose sigma = 63.7
    '''
    fluorescence = np.zeros((len(tgrid), len(egrid)))
    conv_T = intensity * np.exp( -((tgrid - t0) / sigT)**2 * (1./2.) ) * \
            (1. / (sigT * np.sqrt(2.*np.pi)))
    for i, cT in enumerate(conv_T):
        conv_E = cT * np.exp( -((egrid - e0) / sigE)**2 * (1./2.) ) * \
                (1. / (sigE * np.sqrt(2.*np.pi)))
        fluorescence[i,:] = conv_E

    return fluorescence

def compute_intensity(en_gap, populations, transition_dipoles):

    # h = 4.135667662E-15 # this is h in eV*s
    h = 2. * np.pi # hbar = 1 = h/2pi in atomic units
    en_term = (en_gap / h)**3
    pop_tdip_weight = populations * (transition_dipoles * 0.393456)**2
    intensity = pop_tdip_weight * en_term
    return intensity

## Functions that actually do stuff and get called by main

def compute_fluorescence(initconds, tgrid, wgrid, datadir):

    npoints = len(tgrid)
    interp_grid = tgrid - tgrid[0] # zeros the grid for interpolation
    fl_grid = np.zeros((npoints, npoints))

    egrid = np.array([ 1240./x for x in wgrid ])

    s1_tbf_data = {}

    print('Loading all trajectories and collecting data for TBFs on S1.')
    for ic in initconds:
        print(ic)
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():
            
            tbf = data[tbf_key]
            state_id = tbf['state_id']

            if state_id==1: # TBF must be on S1 to be considered for fluorescence.
                print('Computing fluorescence signal from TBF %s' %tbf_key)
                time_steps = tbf['time_steps']
                populations = tbf['populations']
                s0_energies = tbf['energies']['s0']
                s1_energies = tbf['energies']['s1']
                transition_dipoles = tbf['transition_dipoles']

                en_gap = s1_energies - s0_energies # atomic units used in fluorescence calculation
                intensities = compute_intensity(en_gap, populations, transition_dipoles)
                interp_intensities = interpolate(interp_grid, time_steps, en_gap)
                interp_gaps = interpolate(interp_grid, time_steps, en_gap) * 27.21138602
                # ic_fl = approx_fluorescence(egrid, tgrid, interp_gaps, interp_intensities)
                ic_fl = convolve(egrid, tgrid, interp_gaps, interp_grid, interp_intensities)
                s1_tbf_data[tbf_key] = ic_fl
                fl_grid = fl_grid + ic_fl

        # data = pickle.load(open(datadir+('/%02d.pickle' %ic), 'rb'))
        # tbf = data[ex_key]

        # time_steps  = tbf['time_steps']
        # populations = tbf['populations']
        # s0_energies = tbf['energies']['s0'] # * 27.21138602
        # s1_energies = tbf['energies']['s1'] # * 27.21138602
        # transition_dipoles = tbf['transition_dipoles']

        # en_gap = s1_energies - s0_energies # energies computed using atomic units
        # intensities = compute_intensity(en_gap, populations, transition_dipoles)

        # interp_intensities = interpolate(interp_grid, time_steps, intensities)
        # interp_gaps = interpolate(interp_grid, time_steps, en_gap) * 27.21138602
        # # ic_fl = approx_fluorescence(en_grid, t_grid, interp_gaps, interp_intensities)
        # ic_fl = convolve(en_grid, t_grid, interp_gaps, interp_grid, interp_intensities)
        # tbf_data[ex_key] = ic_fl
        # fl_grid = fl_grid + ic_fl

    print('Normalizing fluorescence spectrum.')
    fl_grid = fl_grid / np.max(fl_grid)

    data = {}
    data['ics'] = ics
    data['tgrid'] = tgrid
    data['egrid'] = egrid
    data['wgrid'] = wgrid
    data['fluorescence'] = fl_grid
    data['tbf_fluorescence'] = s1_tbf_data

    print('Saving fluorescence data.')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/fluorescence.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def approx_fluorescence(egrid, tgrid, en_gap, intens):

    ''' Approximates the fluorescence spectrum as best
    as possible without convolution or interpolation.
    If the grid is fine enough, the energy gap at each 
    time point for the trajectory is used to assign 
    the intensity to the fluorescence grid. This is just
    an eye test to see what is happening, so it requires
    that the grid and intensity have the same number of
    points. '''
    fl_grid = np.zeros((len(egrid), len(tgrid)))
    for t in range(len(tgrid)):
        gap = en_gap[t]
        flin = intens[t]
        if np.isfinite(flin):
            ind = np.argmin(np.abs(egrid - gap))
            dE = np.min(np.abs(egrid - gap))
            if dE < 0.1:
                fl_grid[ind, t] += flin

    fl_grid = fl_grid / np.max(fl_grid)

    return fl_grid

def convolve(egrid, tgrid, en_gap, tsteps, intens, sigE=0.15, sigT=65):

    fl_grid = np.zeros((len(egrid), len(tgrid)))
    for idx1 in range(len(en_gap)):
        t0 = tsteps[idx1]
        e0 = en_gap[idx1]
        flin = intens[idx1]
        conv_T = flin * np.exp( -0.5 * ((tgrid - t0)/sigT)**2 ) * (1./(sigT * np.sqrt(2.*np.pi)))
        for idx2, cT in enumerate(conv_T):
            conv_E = cT * np.exp( -0.5 * ((egrid - e0)/sigE)**2 ) * (1./(sigE*np.sqrt(2.*np.pi))) 
            fl_grid[idx2,:] += conv_E

    fl_grid = fl_grid / np.max(fl_grid)

    return fl_grid

def interpolate(grid, tsteps, data):

    interp_data = np.zeros((len(grid)))
    spacing = np.max(grid) / float(len(grid))

    for i in range(len(grid)):
        if i==0:
            tlow = 0
        else:
            tlow = grid[i] - spacing/2
        if i==len(grid) - 1:
            thigh = grid[-1]
        else:
            thigh = grid[i] + spacing/2
        inds = [x for x, y in enumerate(tsteps) if y >= tlow and y <= thigh]
        dat = [data[ind] for ind in inds]
        tdiffs = [np.abs(grid[i] - y) for y in tsteps if y >= tlow and y <= thigh] # computes the distance of the raw data time point from the grid point
        if len(dat) > 0:
            tdiffs_frac = tdiffs / np.sum(tdiffs) # normalizes the distance from the grid point
            interp_data[i] = np.average(dat, weights=tdiffs_frac) # weighted average of the data points by their distance from the grid point
        else: 
            interp_data[i] = 0.

    return interp_data

if __name__=='__main__':

    datadir = '../../1-collect-data/data/'
    fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
    ics = fmsinfo['ics']
    npoints = 201
    tgrid  = np.linspace(-90., 510., npoints)
    wgrid = np.linspace(340, 1140, npoints)
    compute_fluorescence(ics, tgrid, wgrid, datadir)
