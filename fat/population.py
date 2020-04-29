from fat import *
import numpy as np
import os
from scipy.optimize import curve_fit
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
"""
Computation of the population dynamics for all electronic states.
The populations are placed on a uniform time grid and averaged over all AIMS simulations.
This information is dumped to a pickled file and saved to a data directory to be plotted.
Authored by Jimmy K. Yu (jkyu).
"""
def compute_populations(ics, datafiles, tgrid, nstates, datadir=None, save_to_disk=True):
    """
    Arguments: 
        1) ics: an array of integers indexing the initial conditions
        2) datafiles: a list of strings containing the paths to the fat pickle files for the AIMS simulations.
        3) tgrid: a numpy array that contains all time points on the grid.
        4) nstates: an int giving the number of states involved in the population dynamics 
        5) datadir: directory to store data if requested [Default: None]
        6) save_to_disk: a boolean for whether or not to dump the population dynamics to disk [Default: True]
    Description: 
        Computes the population dynamics averaged over all initial conditions.
        First places the population dynamics for each IC onto a grid and then averages over all ICs.
        Then computes the error in the population dynamics using bootstrapping. 
    Returns:
        1) population_dynamics: a dictionary containing details of the population dynamics containing keys
            - ics: a list of initial conditions
            - tgrid: an array of the uniform time grid used for the population computation
            - nstates: an int for the number of adiabatic states involved in the population dynamics
            - populations: averaged populations indexed by adiabatic state label, e.g., 's0' or 's1'
            - ic_populations: matrices containing the populations by initial condition. rows are indexed by initial condition and columns are indexed by time grid points. The matrices themselves are indexed by adiabatic state label.
            - errors: bootstrapping errors for the populations averaged over ics indexed by adiabatic state label, e.g., 's0' or 's1'. These errors correspond to the averaged populations given by the 'populations' key.
            - tbf_populations: populations for individual TBFs indexed by TBF label ('%04d-%04d' %(ic, tbf_id))
    """
    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    for i in range(nstates):
        states['s%d' %i] = []

    # Grab population information out of all ICs and bin that onto a uniform time step time grid (passed in as tgrid) 
    for ic, datafile in zip(ics, datafiles):
        ic_tfinal = 0
        data = pickle.load(open(datafile, 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']

            states['s%d' %tbf_state].append(tbf_key)

            time_steps = tbf['time_steps']
            if time_steps[-1] > ic_tfinal:
                ic_tfinal = time_steps[-1]
            populations = tbf['populations']

            interp_pop = interpolate_to_grid(tgrid, time_steps, populations, extended=True)
            interp_populations['%s' %tbf_key] = interp_pop
        print('IC %04d final time step: %f. From file %s.' %(ic, ic_tfinal, datafile))

    print('Total number of TBFs:')
    print('  Number of ICs: ', len(ics))
    print('  Total number of TBFs: ', np.sum([ len(states[key]) for key in states.keys() ]))
    for state_key in states.keys():
        print('  Number of %s TBFs: %d' %(state_key.title(), len(states[state_key])))
    
    avg_populations = {}
    state_populations = {}
    pop_errors = {}
    for state in states.keys():
        # Compute the average of the population over all ICs
        print('Averaging populations for state %s' %state)
        state_pops = np.zeros((len(ics), len(tgrid)))
        for i, ic in enumerate(ics):
            ic_pop = np.zeros(len(tgrid))
            for tbf_key in states[state]:
                # Group TBFs from same state and same IC
                if ic == int(tbf_key.split('-')[0]):
                    ic_pop += interp_populations[tbf_key]
            state_pops[i,:] = ic_pop
        avg_pops = np.mean(state_pops, axis=0)

        avg_populations[state] = avg_pops
        state_populations[state] = state_pops

        # Compute error for averaged ground state population using bootstrapping
        print('Computing sampling error for state %s' %state)
        state_error = compute_bootstrap_error(ics, tgrid, state_pops)
        pop_errors[state] = state_error

    population_dynamics = {}
    population_dynamics['ics'] = ics
    population_dynamics['tgrid'] = tgrid
    population_dynamics['nstates'] = nstates
    population_dynamics['populations'] = avg_populations
    population_dynamics['ic_populations'] = state_populations
    population_dynamics['errors'] = pop_errors
    population_dynamics['tbf_populations'] = interp_populations # populations for individual TBFs
    print('Dumping interpolated amplitudes to populations.pickle')

    if save_to_disk:
        if not os.path.isdir('%s' %datadir):
            os.mkdir('%s' %datadir)
        with open('%s/populations.pickle' %datadir, 'wb') as handle:
            pickle.dump(population_dynamics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return population_dynamics

def compute_exponential_fit(tgrid, populations, ic_populations, target_state=0):
    """
    Arguments:
        1) tgrid: array containing the uniform time grid points
        2) populations: dictionary containing the average population dynamics
        3) ic_populations: dictionary containing matrices of the population dynamics for the individual ICs
        4) target_state: an int giving the target state to fit the population decay/rise [Default: 0 for s0]
    Description: 
        Computes the time constant of the population dynamics for a single adiabatic state. 
        Also computes a bootstrapping error for this time constant estimated as the measure of ICs included/excluded from the population dynamics.
    Returns:
        1) exp_fit: an array giving a plottable exponential curve fit 
        2) time_constant: a float of the time constant of the exponential fit
        3) time_constant_error: a float giving the bootstrapping error of the exponential fit time constant
    """
    print('Computing population decay constant.')
    # Compute time constant and exponential fit for the population decay
    popt, pcov = curve_fit(exp_func, tgrid, populations['s%d' %target_state], absolute_sigma=False)
    time_constant = popt[1]
    print('Exponential fit. Tau = ', popt[1])
    exp_fit = exp_func(tgrid, *popt)

    # Compute bootstrapping error for the exponential fit
    ic_populations_state = ic_populations['s%d' %target_state]
    nics = np.shape(ic_populations_state)[0]
    resampled_errors = []
    for count in range(1000):
        resample = np.array([ ic_populations_state[x,:] for x in np.random.choice(np.arange(nics), size=(nics), replace=True) ])
        avg_resample = np.mean(resample, axis=0)
        popt, pcov = curve_fit(exp_func, tgrid, avg_resample, absolute_sigma=False)
        resampled_errors.append(popt[1])
    resampled_errors = np.array(resampled_errors)
    time_constant_error = np.std(resampled_errors)

    return exp_fit, time_constant, time_constant_error

def plot_population_dynamics(tgrid, populations, errors, exp_fit, time_constant, time_constant_error, figdir=None, figname='population_dynamics'):
    """
    Arguments:
        1) tgrid: array containing the uniform time grid points
        2) populations: dictionary containing the average population dynamics
        3) errors: dictionary containing the bootstrapping errors computed for the average population dynamics
        4) exp_fit: array containing the exponential fit of the population dynamics for one of the adiabatic states.
        5) time_constant: float containing time constant for the exponential fit.
        6) time_constant_error: float containing the error of the time constant fit. 
        7) figdir: path to directory for saving the figure (Default: None -- pass in a path to save figure)
        8) figname: string giving the name of the population dynamics figure (Default: population_dynamics)
    Description: 
        Automated plotter for population dynamics.
        For publication quality figures, please customize aethetics and labels to better suit the data. 
    Returns:
        Nothing, but saves a PDF of the population dynamics figure. 
    """
    rcParams.update({'figure.autolayout': True})
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    ecolors = ['lightsteelblue', 'moccasin', 'mediumseagreen', 'lightcoral', 'thistle',
            'peru', 'lightpink', 'lightgray', 'khaki', 'azure']
    
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)
    
    keys = [x for x in populations.keys()]
    for i, key in enumerate(keys):
        label = '$S_%d$' %(int(key[1]))
        plt.errorbar(tgrid, populations[key], yerr=errors[key], color=colors[i], linewidth=3.0, elinewidth=1, ecolor=ecolors[i], capsize=0.1, label=label)
    fit_label = 'Fit ($\\tau=%d \pm %d$ fs)' %(time_constant, time_constant_error)
    plt.plot(tgrid, exp_fit, color='black', linewidth=2.0, label=fit_label, linestyle='--')

    plt.ylim([0, 1.1])
    plt.xlim([tgrid[0], tgrid[-1]])
    plt.ylabel('Fractional Population', fontsize=labelsize)
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)
    
    if figdir:
        if not os.path.isdir(figdir):
            os.mkdir(figdir)
        plt.savefig('%s/%s.pdf' %(figdir,figname), dpi=300)

def plot_population_dynamics_single_state(tgrid, average_populations, errors, ic_populations, target_state, figdir=None, figname='population_dynamics_single_state'):
    """
    Arguments:
        1) tgrid: array containing the uniform time grid points
        2) average_populations: dictionary containing the average population dynamics
        3) errors: dictionary containing the bootstrapping errors computed for the average population dynamics
        4) ic_populations: dictionary containing the population dynamics for individual ICs
        5) target_state: int specifying the adiabatic state of interest (0-indexed)
        6) figdir: path to directory for saving the figure (Default: None -- pass in a path to save figure)
        7) figname: string giving the name of the population dynamics figure (Default: population_dynamics)
    Description: 
        Automated plotter for population dynamics for a single adiabatic state.
        This includes the individual ICs for insight as to the impact of individual ICs on the overall population decay/rise.
    Returns:
        Nothing, but saves a PDF of the population dynamics figure. 
    """
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(6,5))
    labelsize = 16
    ticksize = 14
    plt.rc('xtick',labelsize=ticksize)
    plt.rc('ytick',labelsize=ticksize)
    
    avg_pop_target = average_populations['s%d' %target_state]
    ic_pop_target = ic_populations['s%d' %target_state]
    error_target = errors['s%d' %target_state]
    nics = np.shape(ic_pop_target)[0]
    for i in range(nics):
        single = ic_pop_target[i,:]
        if i==0:
            plt.plot(tgrid, single, linestyle='-', linewidth=0.5, color='silver', label='$S_%d$ (Single IC)' %target_state)
        else:
            plt.plot(tgrid, single, linestyle='-', linewidth=0.5, color='silver')
    label = '$S_%d$ (Averaged)' %target_state
    plt.errorbar(tgrid, avg_pop_target, error_target, color='firebrick', linewidth=3.0, elinewidth=1, ecolor='lightcoral', capsize=0.1, label=label)

    plt.ylim([0, 1.1])
    plt.xlim([tgrid[0], tgrid[-1]])
    plt.ylabel('Fractional Population', fontsize=labelsize)
    plt.xlabel('Time [fs]', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)
    
    if figdir:
        if not os.path.isdir(figdir):
            os.mkdir(figdir)
        plt.savefig('%s/%s.pdf' %(figdir,figname), dpi=300)
