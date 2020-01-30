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

def compute_intensity(en_gap, populations, transition_dipoles):

    # h = 4.135667662E-15 # this is h in eV*s
    h = 2. * np.pi # hbar = 1 = h/2pi in atomic units
    intensity = populations * (transition_dipoles)**2 * (en_gap / h)**3
    return intensity

## Functions that actually do stuff and get called by main

def compute_fluorescence(initconds, datafiles, tgrid, wgrid, ex_state=1, outfile_name='fluorescence', ewidth=0.15, twidth=150):

    npoints = len(tgrid)
    interp_grid = tgrid - tgrid[0] # zeros the grid for interpolation
    fl_grid = np.zeros((npoints, npoints))

    egrid = np.array([ 1240./x for x in wgrid ])

    tbf_data = {}

    print('Loading all trajectories and collecting data for TBFs on excited state S%d' %ex_state) 
    for datafile in datafiles:
        print('Computing S%d fluorescence signal from FMS simulation in %s' %(ex_state, datafile))
        data = pickle.load(open(datafile, 'rb'))
        for tbf_key in data.keys():
            
            tbf = data[tbf_key]
            state_id = tbf['state_id']

            if state_id==ex_state: # 0-indexed. e.g., ex_state=1 is fluorescence from S1
                print('Computing fluorescence signal from TBF %s' %tbf_key)
                time_steps = tbf['time_steps']
                populations = tbf['populations']
                ground_energies = tbf['energies']['s0']
                excited_energies = tbf['energies']['s%d' %ex_state]
                transition_dipoles = tbf['transition_dipoles']['s%d' %ex_state]

                en_gap = excited_energies - ground_energies # atomic units used in fluorescence calculation
                intensities = compute_intensity(en_gap, populations, transition_dipoles)
                interp_intensities = interpolate(interp_grid, time_steps, intensities)
                interp_gaps = interpolate(interp_grid, time_steps, en_gap) * 27.21138602
                # ic_fl = approx_fluorescence(egrid, tgrid, interp_gaps, interp_intensities)
                ic_fl = convolve(egrid, tgrid, interp_gaps, interp_grid, interp_intensities, ewidth, twidth)
                tbf_data[tbf_key] = ic_fl
                fl_grid = fl_grid + ic_fl

    data = {}
    data['ics'] = ics
    data['tgrid'] = tgrid
    data['egrid'] = egrid
    data['wgrid'] = wgrid
    data['fluorescence'] = fl_grid
    data['tbf_fluorescence'] = tbf_data

    print('Saving fluorescence data.')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/%s.pickle' %outfile_name, 'wb') as handle:
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

    return fl_grid

def convolve(egrid, tgrid, en_gap, tsteps, intens, ewidth=0.15, twidth=150):
    '''
    To convolve the intensity in time and energy, multiply the intensity
    (a delta function) by gaussians, one centered on the transition 
    frequency (written as an energy in units of eV) and the second
    centered on the adaptive time step. Here, e0 and t0 are the centers
    of these two gaussians respectively and egrid and tgrid are the grid points 
    that we are using for our fluorescence plot. sigE and sigT are the
    gaussian width parameters. The width = stdev = sqrt(variance).
    The energy width is set to 0.15 eV and the time width is 
    150 fs (from Schmidt et al.) as default parameters for the convolution. 
    Note that the Gaussian width of the laser pulse is measured as the
    full-width half maximum (FWHM), where FWHM = 2 * sigma * sqrt(2 * ln 2) 
    => FWHM ~= 2.3548200450309493 * sigma. 
    '''
    sigE = ewidth / 2.3548200450309493
    sigT = twidth / 2.3548200450309493
    fl_grid = np.zeros((len(egrid), len(tgrid)))
    for idx1 in range(len(en_gap)):
        t0 = tsteps[idx1]
        e0 = en_gap[idx1]
        flin = intens[idx1]
        conv_T = flin * np.exp( -0.5 * ((tgrid - t0)/sigT)**2 ) * (1./(sigT * np.sqrt(2.*np.pi)))
        for idx2, cT in enumerate(conv_T):
            conv_E = cT * np.exp( -0.5 * ((egrid - e0)/sigE)**2 ) * (1./(sigE*np.sqrt(2.*np.pi))) 
            fl_grid[idx2,:] += conv_E

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

    datadir = '../../1_collect_data/'
    fmsinfo = pickle.load(open(datadir+'/data/fmsinfo.pickle', 'rb'))
    ics = fmsinfo['ics']
    picklefiles = fmsinfo['datafiles']
    datafiles = [ datadir+x for x in picklefiles ] 
    npoints = 51
    tgrid  = np.linspace(-40., 260., npoints)
    wgrid = np.linspace(1240/4., 1240/1.5, npoints)
    ex_state = 1 # S1-S0 fluorescence. ex_state is 0 indexed, so S1 is ex_state=1
    outfile_name = 'fluorescence' # name of the pickle file saved to the data directory
    ewidth = 0.20 # energy uncertainty - FWHM for gaussian convolution
    twidth = 150 # time uncertainty - FWHM for gaussian convolution
    compute_fluorescence(ics, datafiles, tgrid, wgrid, ex_state=ex_state, outfile_name=outfile_name, ewidth=ewidth, twidth=twidth)
