from fat import *
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
"""
Computation of the evolution of geometric quantities over the course of the simulations.
Included here are routines for distances, angles and dihedral angles. 
The geometric quantities are placed on a uniform time grid for analysis for analysis over all AIMS simulations due to the adaptive time stepping. 
This information is dumped to a pickled file and saved to a data directory to be plotted.
Authored by Jimmy K. Yu (jkyu).
"""
def compute_distance(frame, atom_inds):
    """
    Description: 
        Computes the Euclidean distance between two atoms given the full geometry and the indices of the two atoms.
    Arguments: 
        1) frame: a numpy array containing the full geometry (a frame of the trajectory)
        2) atom_inds: a list containing two elements, one for each of the two atoms
    Returns
        1) distance: a float of the Euclidean distance between the two atoms in the same units as provided by the coordinates
    """
    atom1 = frame[atom_inds[0]]
    atom2 = frame[atom_inds[1]]

    v12 = atom2 - atom1
    distance = np.linalg.norm(v12)

    return distance

def compute_angle(frame, atom_inds):
    """
    Description: 
        Computes the angle made by three atoms given the full geometry and the indices of the atoms.
    Arguments: 
        1) frame: a numpy array containing the full geometry (a frame of the trajectory)
        2) atom_inds: a list containing three elements, one for each of the three atoms (in order) that make angle
    Returns
        1) angle: a float of the angle (in degrees) between the three atoms
    """
    atom1 = frame[atom_inds[0]]
    atom2 = frame[atom_inds[1]]
    atom3 = frame[atom_inds[2]]

    v21 = atom2 - atom1
    v23 = atom2 - atom3
    norm21 = np.linalg.norm(v21)
    norm23 = np.linalg.norm(v23)

    angle = np.arccos( np.dot(v21, v23)/(norm21*norm23) ) * 180 / np.pi

    return angle

def compute_dihedral(frame, atom_inds):
    """
    Description: 
        Computes the dihedral angle made by four atoms given the full geometry and the indices of the atoms.
    Arguments: 
        1) frame: a numpy array containing the full geometry (a frame of the trajectory)
        2) atom_inds: a list containing four elements, one for each of the four atoms (in order) that make the dihedral angle
    Returns
        1) dihedral_angle: a float of the dihedral angle (in degrees) between the four atoms
    """
    atom1 = frame[atom_inds[0]]
    atom2 = frame[atom_inds[1]]
    atom3 = frame[atom_inds[2]]
    atom4 = frame[atom_inds[3]]

    v12 = atom2 - atom1
    v23 = atom3 - atom2
    v34 = atom4 - atom3
    v12 = v12 / np.linalg.norm(v12)
    v23 = v23 / np.linalg.norm(v23)
    v34 = v34 / np.linalg.norm(v34)

    n1 = np.cross(v12, v23)
    n2 = np.cross(v23, v34) 
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, v23), n2)
    dihedral_angle = np.arctan2(y,x) * 180./np.pi

    return dihedral_angle

def choose_geometric_quantity(trajectory, atom_inds, start_trans=False):
    """
    Description: 
        Chooses the geometric quantity to compute by the length of the liste of atom indices provided.
        Computes the geometric quantity for each frame in the trajectory. 
    Arguments: 
        1) trajectory: a numpy array containing the full trajectory (all frames simulation for the full molecule)
        2) atom_inds: a list containing two to four elements, one for each of the atoms involved in the geometric quantity of interest, e.g. distance (2), angle (3), dihedral angle (4) 
        3) start_trans: a boolean that indicates whether the initial condition begins from the trans conformation for the dihedral angle of interest (if relevant). 
    Returns
        1) gq_trajectory: a numpy array containing a float corresponding to the geometric quantity for each frame in the simulation.
    """
    if len(atom_inds)==2:
        # Compute the distance for each frame of the trajectory
        gq_trajectory = []
        for i in range(len(trajectory)):
            frame = trajectory[i]
            gq_frame = compute_distance(frame, atom_inds)
            gq_trajectory.append(gq_frame)

    elif len(atom_inds)==3:
        # Compute the angle for each frame of the trajectory
        gq_trajectory = []
        for i in range(len(trajectory)):
            frame = trajectory[i]
            gq_frame = compute_angle(frame, atom_inds)
            gq_trajectory.append(gq_frame)

    elif len(atom_inds)==4:
        # Compute the dihedral angle for each frame of the trajectory
        gq_trajectory = []
        for i in range(len(trajectory)):
            frame = trajectory[i]
            gq_frame = compute_dihedral(frame, atom_inds)
            # Handle wrapping from -359 to 1 degrees.
            if start_trans and gq_frame < 0:
                gq_frame = gq_frame + 360
            # Handle wrapping from 179 to -179 degrees.
            if i>0:
                if (gq_frame - gq_trajectory[i-1]) > 300:
                    gq_frame = gq_frame - 360
                elif (gq_frame - gq_trajectory[i-1]) < -300:
                    gq_frame = gq_frame + 360
            gq_trajectory.append(gq_frame)

    else: 
        raise ValueError('Number of atom indices given for the desired geometric quantity is not recognized for any of the currently implemented geometric quantity calculations (2=distance, 3=angle, 4=dihedral).')

    return np.array(gq_trajectory)

def compute_geometric_quantities(ics, datafiles, tgrid, gq_index, start_trans=False, compute_averages=False, compute_error=False, tbf_populations=None, datadir=None, save_to_disk=True):
    """
    Description: 
        Computes geometric quantities for the AIMS simulations over all specified ICs and trajectories.
        Places the geometric information on a uniform grid in order to facilitate time averaging.
    Arguments: 
        1) ics: a list of integers providing the indices of the initial conditions
        2) datafiles: a list of strings providing paths to the fat data files for the FMS simulations
        3) tgrid: a numpy array containing the uniformly spaces time grid
        4) gq_index: a dictionary containing the specified geometric quantities to compute. The dictionary keys name the geometric quantity and index a list of atom indices involved in the geometric quantity. Example: the key 'C10-C11=C12-C13' may index a dihedral angle involving atoms indexed [3333, 3335, 3337, 3341]. Hint: Check VMD for ordering for large molecules; look at the coordinate file for smaller molecules. 
        5) start_trans: optional boolean flag for indicating the starting configuration for the simulation. Helps with dihedral angle wrapping if needed. [Default: False]
        6) compute_averages: a boolean flag for whether or not to compute averages over all ICs for the geometric quantities [Default: False]
        7) compute_error: a boolean flag for whether or not to compute the bootstrapping error for averages o8er all ICs [Default: False -- expensive]
        8) tbf_populations: a dictionary of population dynamics indexed by TBF. Required if one wishes to compute IC averaging of the geometric quantities. [Default: None]
        9) datadir: string containing path to directory in which the final data file should be stored.
        10) save_to_disk: optional boolean flag for whether or not to dump data to disk. [Default: True]
    Returns:
        1) geometric_quantities: a dictionary containing all of the request geometric information, including the following fields:
            - ics: a list of initial conditions (same as input, redundant)
            - tgrid: an array of the uniform time grid (same as input, redundant)
            - geometric_quantity_index: a dictionary containing names and indices for all geometric quantities computed (same as input -- should be saved to file)
            - geometric_quantities: a dictionary containing all of the geometric quantities computed. Contains nested dictionaries indexed first by the name of the geometric quantity (see geometric_quantity_index) and then by key of the TBF ('%04d-%04d' %(ic, tbf_id)) of interest. This provides trajectory information for the geometric quantities placed on the uniform time grid. For example, to access the trajectory of geometric quantities, use gq_traj = geometric_analysis[gq_name][tbf_id]
            - (Optional) averaged_geometric_quantities: a dictionary containing the geometric quantities averaged over all ICs at each time step. Indexed by the name of the geometric quantity, as above.
            - (Optional) averaged_geometric_quantities_errors: bootstrapping errors for averaged_geometric_quantities. Data structured the same way. 
    """
    gq_names = [x for x in gq_index.keys()]
    gq_list = [ gq_index[x] for x in gq_names ] 

    # raw = data directly from the FMS simulations before interpolation to grid
    raw_gq = {}
    raw_tsteps = {}

    # Compute geometric quantities for all collected frames in the trajectories
    for ic, datafile in zip(ics, datafiles):
        data = pickle.load(open(datafile, 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            time_steps = tbf['time_steps']
            trajectory = tbf['trajectory']
            populations = tbf['populations']

            raw_gq_data = {}
            # Compute each of the specified geometric quantities
            for gq_name, gq_inds in zip(gq_names, gq_list):
                # Choose geometric quantity to compute by the number of indices in gq_list
                # gq_list is the list of atoms involved in the geometric quantity
                # e.g., length 2 = distance, 3 = angle, 4 = dihedral
                gq_trajectory = choose_geometric_quantity(trajectory, gq_inds, start_trans)
                raw_gq_data[gq_name] = gq_trajectory

            raw_gq['%s' %tbf_key] = raw_gq_data
            raw_tsteps['%s' %tbf_key] = time_steps

    # Interpolate the trajectories to the uniform grid
    interp_gq = {}
    for gq_name in gq_names:
        interp_gq_data = {}
        for tbf_key in raw_gq.keys():
            tsteps = raw_tsteps[tbf_key]
            gq_trajectory = raw_gq[tbf_key][gq_name]
            interp_gq_data[tbf_key] = interpolate_to_grid(tgrid, tsteps, gq_trajectory, extended=True, geometric_quantity=True)
        interp_gq[gq_name] = interp_gq_data

    # Save data
    geometric_analysis = {}
    geometric_analysis['ics'] = ics
    geometric_analysis['geometric_analysis_index'] = gq_index # dictionary containing names + atom indices
    geometric_analysis['geometric_quantities'] = interp_gq
    geometric_analysis['tgrid'] = tgrid

    if compute_averages:
        averaged_gqs = compute_averaged_geometric_quantities(tgrid, tbf_populations, interp_gq)
        geometric_analysis['averaged_geometric_quantities'] = averaged_gqs
        if compute_error:
            averaged_gqs_errors = compute_bootstrap_error_geometric(ics, tgrid, tbf_populations, interp_gq)
            geometric_analysis['averaged_geometric_quantities_errors'] = averaged_gqs_errors

    if save_to_disk:
        print('Dumping geometric analysis to %s/geometric_analysis.pickle' %datadir)
        if not os.path.isdir('%s' %datadir):
            os.mkdir('%s' %datadir)
        with open('%s/geometric_analysis.pickle' %datadir, 'wb') as handle:
            pickle.dump(geometric_analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return geometric_analysis

def compute_averaged_geometric_quantities(tgrid, tbf_populations, geometric_quantities):
    """
    Description: 
        Computes the time-averaged geometric quantities for the AIMS simulations.
    Arguments: 
        1) tgrid: a numpy array containing the uniformly spaces time grid
        2) tbf_populations: a dictionary containing the population dynamics of the individual TBFs on the uniform time grid for the entire AIMS simulation. Originally written to population_dynamics.pickle by compute_populations() 
        3) geometric_quantities: a dictionary containing the trajectory information for the computed geometric properties on the uniform time grid. Originally written to geometric_analysis.pickle by compute_geometric_quantities()
    Returns:
        1) averaged_gqs: a dictionary containing the averaged geometric quantities over all AIMS simulations. Indexed by the name of the geometric quantity. 
    """
    print('Computing averaged geometric quantities over all ICs.')
    tbf_keys = [x for x in tbf_populations.keys()]
    population_matrix = np.zeros((len(tbf_keys), len(tgrid)))
    for i, tbf_key in enumerate(tbf_keys):
        population_matrix[i,:] = tbf_populations[tbf_key]

    averaged_gqs = {} 
    gq_names = [x for x in geometric_quantities]
    for gq_name in gq_names:
        # Generate masked array to ignore NaN values during averaging (where the TBF is dead or not yet spawned)
        gq_matrix = np.zeros((len(tbf_keys), len(tgrid)))
        for i, tbf_key in enumerate(tbf_keys):
            gq_matrix[i,:] = geometric_quantities[gq_name][tbf_key] 
        gq_masked = np.ma.MaskedArray(gq_matrix, mask=np.isnan(gq_matrix)) 

        # average over all TBFs at each grid point weighted by TBF population
        gq_averaged = []
        for j in range(np.shape(gq_masked)[1]):
            amplitude = population_matrix[:,j]
            gq_averaged.append(np.ma.average(gq_masked[:,j], weights=amplitude))

        averaged_gqs[gq_name] = np.array(gq_averaged)

    return averaged_gqs

def compute_bootstrap_error_geometric(ics, tgrid, tbf_populations, geometric_quantities):
    """
    Description: 
        Compute bootstrapping error for AIMS simulations over all ICs weighted by population. 
        This is a measurement of the error by sampling with replacement over the ICs included in the analysis of the data.
    Arguments:
        1) ics: a list of integers providing the indices of the initial conditions
        2) grid: an array for the grid points
        3) tbf_populations: a dictionary containing the population dynamics of the individual TBFs on the uniform time grid for the entire AIMS simulation. Originally written to population_dynamics.pickle by compute_populations() 
        4) geometric_quantities: a dictionary containing the trajectory information for the computed geometric properties on the uniform time grid. Originally written to geometric_analysis.pickle by compute_geometric_quantities()
    Returns:
        1) error: a dictionary indexed by the geometric quantities requested. Each dictionary entry is an array that contains the error at each time step (grid point)
    """
    print('Computing bootstrapping error for geometric quantities. This might take a while.')
    tbf_keys = [x for x in tbf_populations.keys()]
    resampled = {}
    for gq_name in geometric_quantities.keys():
        resampled[gq_name] = []
    for ppp in range(1000):
        resampled_ics = np.random.choice(ics, size=len(ics), replace=True)
        resampled_tbf_keys = []
        for y in resampled_ics:
            resampled_tbf_keys = resampled_tbf_keys + [ x for x in tbf_keys if int(x.split('-')[0])==y ]
        population_matrix = np.zeros((len(resampled_tbf_keys), len(tgrid)))
        for i, tbf_key in enumerate(resampled_tbf_keys):
            population_matrix[i,:] = tbf_populations[tbf_key]
        for gq_name in geometric_quantities.keys():
            # Generate masked array to ignore NaN values during averaging (where the TBF is dead or not yet spawned)
            gq_matrix = np.zeros((len(resampled_tbf_keys), len(tgrid)))
            for i, tbf_key in enumerate(resampled_tbf_keys):
                gq_matrix[i,:] = geometric_quantities[gq_name][tbf_key] 
            gq_masked = np.ma.MaskedArray(gq_matrix, mask=np.isnan(gq_matrix)) 

            # average over all TBFs at each grid point weighted by TBF population
            gq_averaged = []
            for j in range(np.shape(gq_masked)[1]):
                amplitude = population_matrix[:,j]
                gq_averaged.append(np.ma.average(gq_masked[:,j], weights=amplitude))
            resampled[gq_name].append(np.array(gq_averaged))

    error = {}
    for gq_name in resampled.keys():
        error[gq_name] = np.mean(np.array(resampled[gq_name]), axis=1)

    return error

def plot_averaged_geometric_quantities(tgrid, averaged_gqs, errors, figdir=None, figname='geometric_quantities_averaged'):
    """
    Arguments:
        1) tgrid: array containing the uniform time grid points
        2) averaged_gq: array containing the geometric evolution of the AIMS simulation in one coordinate. These are individual entries of the averaged_geometric_quantities dictionary saved in geometric_analysis.pickle. 
        2) averaged_gqs: dictionary containing the the geometric evolution of the AIMS simulation in coordiantes specified by the dictionary keys. Load averaged_geometric_quantities saved in geometric_analysis.pickle
        3) errors: dictionary containing the bootstrapping errors computed for the geometric evolution of the AIMS simulation in coordiantes specified by the dictionary keys. Load averaged_geometric_quantities_errors saved in geometric_analysis.pickle
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
    
    gq_names = [x for x in averaged_gqs.keys()]
    for i, gq_name in enumerate(gq_names):

        fig = plt.figure(figsize=(6,5))
        labelsize = 16
        ticksize = 14
        plt.rc('xtick',labelsize=ticksize)
        plt.rc('ytick',labelsize=ticksize)

        label = gq_name
        plt.errorbar(tgrid, averaged_gqs[gq_name], yerr=errors[gq_name], color=colors[i], linewidth=3.0, elinewidth=1, ecolor=ecolors[i], capsize=0.1, label=label)

        plt.xlim([tgrid[0], tgrid[-1]])
        plt.ylabel('Coordinate', fontsize=labelsize)
        plt.xlabel('Time [fs]', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)
        
        if figdir:
            if not os.path.isdir(figdir):
                os.mkdir(figdir)
            plt.savefig('%s/%s_%s.pdf' %(figdir,figname,gq_name), dpi=300)
        plt.close()

