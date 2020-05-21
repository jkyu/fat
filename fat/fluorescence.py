from fat import *
import numpy as np
import os
import pickle
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib import rcParams
import sys
"""
Computation of time-resolved fluorescence spectra from the AIMS simulations.
Included here are routines for computing the 2D spectra (energy and time) and the time-integrated (steady-state) fluorescence signal.
Functionality for steady-state fluorescence not ported here yet. 
Plotting functions are included as well.
Authored by Jimmy K. Yu (jkyu).
"""
def compute_intensity(en_gaps, populations, transition_dipoles):
    """
    Description: 
        Helper function for compute_fluorescence(). Computes the fluorescence intensities for an entire TBF weighted by the population at each time point.
    Arguments:
        1) en_gaps: a numpy array containing the Sn-S0 energy gap (in a.u.) at each time point, where Sn is the adiabatic state label of the bright state of interest.
        2) populations: a numpy array containing the population of the TBF at each time point.
        3) transition_dipoles: a numpy array containing the S0->Sn transition dipole.
    Returns:
        1) intensities: a numpy array containing the fluorescence intensity contribution of the TBF at each time point. 
    """
    h = 2. * np.pi # hbar = 1 = h/2pi in atomic units
    intensities = populations * (transition_dipoles)**2 * (en_gaps / h)**3
    return intensities

def convolve_2D(egrid, tgrid, en_gaps, intensity, ewidth, twidth):
    """
    Description: 
        Helper function for compute_fluorescence(). Performs 2D gaussian convolution in time and energy in order to reflect the instrument response expected of an experimental spectrum. Set the energy and time widths appropriately relative to the resolution of the experiment.
        To convolve the intensity in time and energy, multiply the intensity (a delta function) by gaussians, one centered on the transition frequency (written as an energy in units of eV) and the second centered on the adaptive time step. 
        Note that the Gaussian width of the laser pulse is measured as the full-width half maximum (FWHM), where FWHM = 2 * sigma * sqrt(2 * ln 2) => FWHM ~= 2.3548200450309493 * sigma. 
    Arguments:
        1) egrid: a numpy array containing the grid points along the energy axis
        2) tgrid: a numpy array containing the grid points along the time axis
        3) en_gaps: a numpy array containing the energy gaps at each time point over the TBF
        4) intensity: a 2D numpy grid containing the contribution to the 2D fluorescence spectrum by the TBF
        5) ewidth: a float indicating the expected uncertainty in energy
        6) twidth: a float indicating the expected uncertainty in time (resolution)
    Returns:
        1) fl_grid: a 2D numpy grid containing the convolved contribution to the 2D fluorescence spectrum by the TBF
    """
    sigE = ewidth / 2.3548200450309493
    sigT = twidth / 2.3548200450309493
    fl_grid = np.zeros((len(egrid), len(tgrid)))
    for idx1 in range(len(egrid)):
        t0 = tgrid[idx1]
        e0 = en_gaps[idx1]
        flin = intensity[idx1]
        conv_T = flin * np.exp( -0.5 * ((tgrid - t0)/sigT)**2 ) * (1./(sigT * np.sqrt(2.*np.pi)))
        for idx2, cT in enumerate(conv_T):
            conv_E = cT * np.exp( -0.5 * ((egrid - e0)/sigE)**2 ) * (1./(sigE*np.sqrt(2.*np.pi))) 
            fl_grid[idx2,:] += conv_E

    return fl_grid

def compute_fluorescence(datafiles, tgrid, wgrid, ex_state=1, ewidth=0.15, twidth=150, datadir=None, save_to_disk=True):
    """
    Description:
        Function for computing the fluorescence signal from the AIMS simulation over all provided initial conditions.
    Arguments:
        1) datafiles: list of strings that indicate the path to the pickle files containing information for each AIMS simulation (one for each initial condition)
        2) tgrid: a numpy array containing the time grid
        3) wgrid: a numpy array containing the wavelength grid. Note that a wavelength grid is specified and converted to energy in order to make visualization on a uniform wavelength scale easier.
        4) ex_state: an integer specifying the nonadiabatic state label (e.g. 1 for S1) corresponding to the bright state of interest (from which fluorescence occurs)
        5) ewidth: a float indicating the uncertainty in energy for broadening the spectrum
        6) twidth: a float indicating the uncertainty in time for broadening the spectrum (corresponds roughly to time resolution of experiments)
        7) datadir: path to directory to which to save data to disk [Default: None]
        8) save_to_disk: boolean specifying whether or not to save data to disk [Default: True]
    """
    npoints = len(tgrid)
    # interp_grid = tgrid - tgrid[0] # zeros the grid for interpolation
    egrid = np.array([ 1240./x for x in wgrid ])
    fl_grid = np.zeros((npoints, npoints))
    tbf_data = {}

    # Compute the fluorescence intensity contributed by each initial condition/datafile
    print('Loading all trajectories and collecting data for TBFs on excited state S%d' %ex_state) 
    for datafile in datafiles:
        print('Computing S%d fluorescence signal from FMS simulation in %s' %(ex_state, datafile))
        data = pickle.load(open(datafile, 'rb'))
        for tbf_key in data.keys():
            
            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']

            if tbf_state==ex_state: # 0-indexed. e.g., ex_state=1 is fluorescence from S1
                print('Computing fluorescence signal from TBF %s' %tbf_key)
                time_steps = tbf['time_steps']
                populations = tbf['populations']
                ground_energies = tbf['energies']['s0']
                excited_energies = tbf['energies']['s%d' %ex_state]
                transition_dipoles = tbf['transition_dipoles']['s%d' %ex_state]

                en_gap = excited_energies - ground_energies # atomic units used in fluorescence calculation
                intensities = compute_intensity(en_gap, populations, transition_dipoles)
                interp_intensities = interpolate_to_grid(tgrid, time_steps, intensities)
                interp_gaps = interpolate_to_grid(tgrid, time_steps, en_gap) * 27.21138602
                ic_fl = convolve_2D(egrid, tgrid, interp_gaps, interp_intensities, ewidth, twidth)
                tbf_data[tbf_key] = ic_fl
                fl_grid = fl_grid + ic_fl

    # Save data
    fluorescence_data = {}
    fluorescence_data['tgrid'] = tgrid
    fluorescence_data['egrid'] = egrid
    fluorescence_data['wgrid'] = wgrid
    fluorescence_data['fluorescence'] = fl_grid
    fluorescence_data['tbf_fluorescence'] = tbf_data

    fluorescence_error = compute_fluorescence_error(tbf_data)
    fluorescence_data['fluorescence_error'] = fluorescence_error

    if save_to_disk:
        print('Saving fluorescence data to %s/fluorescence.pickle' %datadir)
        if not os.path.isdir('%s' %datadir):
            os.mkdir('%s' %datadir)
        with open('%s/fluorescence.pickle' %datadir, 'wb') as handle:
            pickle.dump(fluorescence_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return fluorescence_data

def recompute_fluorescence_grid(tbf_data, keys):
    """
    Description: 
        Helper function for compute_fluorescence_error(). Recomputes the fluorescence grid given a resampled set of TBF keys by bootstrapping.
    Arguments: 
        1) tbf_data: a dictionary containing the fluorescence contributions by each TBF
        2) keys: a list of tbf keys giving the resampled TBFs from bootstrapping
    Returns:
        1) fl_grid: a fl_grid computed only from the resampled keys
    """
    fl_grid = np.zeros_like(tbf_data[keys[0]])
    for key in keys:
        ic_grid = tbf_data[key]
        fl_grid = fl_grid + ic_grid
    return fl_grid

def compute_fluorescence_error(tbf_data):
    """
    Description: 
        Helper function for compute_fluorescence() that computes the error at each grid point on the 2D fluorescence spectrum. Useful for plotting 1D lineouts. 
    Arguments: 
        1) tbf_data: a dictionary containing the fluorescence contributions by each TBF
    Returns:
        1) grid_error: a 2D numpy grid containing the error at each grid point
    """
    keys = [x for x in tbf_data.keys()]
    fl_grid = recompute_fluorescence_grid(tbf_data, keys)
    ics = list(set([ int(x.split('-')[0]) for x in keys ]))
    sampled_ics = [ [x for x in np.random.choice(ics, size=len(ics), replace=True)] for _ in range(1000) ]
    sampled_keys = [ [x for x in keys if int(x.split('-')[0]) in ic_subset] for ic_subset in sampled_ics ]
    sampled_grids = np.array([ recompute_fluorescence_grid(tbf_data, key_subset) for key_subset in sampled_keys ])
    grid_error = np.std(sampled_grids, axis=0)

    return grid_error

def interpolate_fluorescence_plot(tgrid, wgrid, fgrid, ngrid=1000):
    ''' 
    Description: 
        Helper function for plot_fluorescence(). Places the 2D fluorescence spectrum on a finer grid.
    Arguments: 
        1) tgrid: numpy array for the original time grid used to compute the spectrum
        2) wgrid: numpy array for the original wavelength grid used to compute the spectrum
        3) fgrid: 2D numpy grid for the originally computed fluorescence spectrum
        4) ngrid: number of new grid points on each axis desired for the finer grid [Default: 1000]
    Returns:
        1) tgrid3: numpy array for the new time grid with ngrid points
        2) wgrid3: numpy array for the new wavelength grid with ngrid points
        3) fgrid3: 2D numpy grid for the new fluorescence spectrum with ngrid x ngrid points
    '''
    # Unravel the fluorescence grid 
    ts = []
    ws = []
    fs = []
    for i, t in enumerate(tgrid):
        for j, w in enumerate(wgrid):
            ts.append(t)
            ws.append(w)
            fs.append(fgrid[i,j])
    # Expand the time and wavelength grids
    tgrid2 = np.linspace(tgrid[0], tgrid[-1], ngrid)
    wgrid2 = np.linspace(wgrid[0], wgrid[-1], ngrid)
    # Place the expanded grids on a mesh.
    tgrid3, wgrid3 = np.meshgrid(tgrid2, wgrid2)
    # Interpolate the fluorescence grid onto the mesh.
    fgrid3 = scipy.interpolate.griddata( (ts, ws), fs, (tgrid3, wgrid3), method='cubic', fill_value=0)
    # Normalize the interpolated grid.
    fgrid3 = fgrid3 / np.max(fgrid3)

    return tgrid3, wgrid3, fgrid3

def plot_2D_fluorescence(fgrid, wgrid, tgrid, fl_shift=0, lineout_wls=[], figdir=None, figname='2D_fluorescence'):
    """
    Description: 
        Plotting function for the 2D fluorescence spectrum. Handles constructing the contour map and the colormap.
        As a note, the rows of the 2D fluorescence spectrum matrix correspond to the time axis and the columns to the energy axis.
    Arguments:
        1) fgrid: 2D numpy grid originally computed 2D fluorescence spectrum
        2) wgrid: numpy array of wavelengths used in the grid to compute fgrid
        3) tgrid: numpy array of time steps used in the grid to compute fgrid
        4) fl_shift: a float giving the energy shift (if applicable) of the fluorescnece spectrum to match experiment. [Default: 0]
        5) lineout_wls: a list containing integers specifying wavelengths at which to plot lineouts
        6) figdir: string giving directory to save figure
        7) figname: string giving the figure name
    """
    # Energy shift for the fluorescence based on comparison to experiment. 
    if fl_shift > 0:
        # Keep in mind that you're going to want to manually set the plotting ranges
        # for the wavelength if you use the energy shift.
        egrid = [(1240./x - fl_shift) for x in wgrid]
        wgrid = [int(1240./x) for x in egrid]

    # Place fluorescence plot on a finer grid
    tgrid3, wgrid3, fgrid3 = interpolate_fluorescence_plot(tgrid, wgrid, fgrid)

    # Plot the fluorescence grid
    fig = plt.figure(facecolor='white', figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)

    # Construct a custom truncated jet colormap.
    ngrid_cm = 51
    colors = [ (40, 80, 250), (15, 160, 240), (25, 200, 225), (100, 250, 190), (180, 240, 150), (220, 210, 120), (255, 165, 90), (255, 100, 50), (255, 0, 0) ]
    colors = [ (x[0]/255., x[1]/255., x[2]/255.) for x in colors]
    cmap_name = 'special'
    cm = cl.LinearSegmentedColormap.from_list(cmap_name, colors, N=ngrid_cm)
    levels = np.linspace(np.min(fgrid3), np.max(fgrid3), ngrid_cm)
    fl_im = plt.contourf(tgrid3, wgrid3, fgrid3, cmap=cm, levels=levels, vmin=np.min(fgrid3), vmax=np.max(fgrid3))
    cb = fig.colorbar(fl_im, format='%.1f')
    cb.set_ticks( [ x for x in np.arange( np.min(fgrid3), np.max(fgrid3)+0.1, 0.1) ] )
    cb.set_label('Fluorescence Intensity', rotation=270, labelpad=15, fontsize=labelsize)

    # Plot lineouts if requested by giving a list of lineout wavelengths
    if len(lineout_wls) > 0:
        for wl in lineout_wls:
            plt.plot([tgrid[0], tgrid[-1]], [wl]*2, linewidth=1.5, linestyle='--', color='silver')

    plt.xlabel('Time [fs]',fontsize=labelsize)
    plt.ylabel('Wavelength [nm]',fontsize=labelsize)
    plt.tight_layout()

    print('Saving the fluorescence figure to ./figures/%s.pdf' %figname)
    if figdir:
        if not os.path.isdir(figdir):
            os.mkdir(figdir)
        plt.savefig('%s/%s.pdf' %(figdir,figname), dpi=300)
    plt.close()
