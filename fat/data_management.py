import numpy as np
"""
Data management and manipulation tools for data structures in fat.
Authored by Jimmy K. Yu (jkyu). 
"""
def interpolate_to_grid(grid, tsteps, data, extended=False, geometric_quantity=False):
    """
    Description: 
        Places the data of interest on a grid. 
        This is necessary for the analysis for multiple AIMS simulations due to the adaptive time stepping.
        The placement on the grid is performed by taking weighted averages (weighted by the distance from the grid point) of the raw data within grid windows centered at the grid points. 
    Arguments: 
        1) grid: a numpy array that contains all time points on the grid.
        2) tsteps: a numpy array containing the raw data time steps from the FMS simulation
        3) data: a numpy array containing the raw data for the quantity to be placed on the grid, e.g., a geometric property like bond lengths or the population
        4) extended: a boolean specifying whether to fill the empty grid points after a TBF has been killed with NaN (False) or with the final value of the TBF (True).
        5) geometric_quantity: a boolean specifying whether to fill empty grid points before a TBF has been spawned with NaN (True) or leave as zeros (False).
    Returns:
        1) grid_data: a numpy array containing the data of interest placed onto the grid. 
    """
    grid_data = np.zeros((len(grid)))
    spacing = np.max(grid) / float(len(grid))

    for i in range(len(grid)):
        # Determine the low and high values of the grid window
        if i==0:
            wlow = 0
        else:
            wlow = grid[i] - spacing/2
        if i==len(grid) - 1:
            whigh = grid[-1]
        else:
            whigh = grid[i] + spacing/2
        # Take a subset of the data points that are within the grid window
        inds = [x for x, y in enumerate(tsteps) if y >= wlow and y <= whigh]
        subset = [data[ind] for ind in inds]
        # Compute the distance of the raw data time point from the grid point
        tdiffs = [np.abs(grid[i] - y) for y in tsteps if y >= wlow and y <= whigh]
        if len(subset) > 0:
            # normalize the distance from the grid point
            tdiffs_frac = tdiffs / np.sum(tdiffs) 
            # take a weighted average of the data points by their distance from the grid point
            grid_data[i] = np.average(subset, weights=tdiffs_frac) 
        else: 
            if extended:
                # Grid windows in which no data exists will be assigned the value of the previous grid point
                # This effectively means killed TBFs are included in the averaging by using the final value for all time points after the TBF has been killed. 
                grid_data[i] = grid_data[i-1]
            elif geometric_quantity and not extended:
                # Grid windows in which no data exists will be assigned NaN values.
                # Appropriate for the case where you want to plot a TBF only where it is alive.
                grid_data[i] = np.nan
            else:
                # Grid windows in which no data exists will be assigned 0 values.
                # This is appropriate for a spectrum, where a TBF contributes 0 signal after it has died.
                grid_data[i] = 0

    # For geometric properties, TBFs should be excluded from the average where they have not yet been spawned. This handles those cases. 
    # For a population analysis, 0 population is expected for an adiabatic state before any TBFs have been spawned on it, so this should not be used. 
    if geometric_quantity: 
        first_ind = np.nonzero(grid_data)[0][0]
        grid_data[:first_ind] = np.nan

    return grid_data

def exp_func(x, A, b, c):
    """
    Description: Exponential function defined for exponential fits.
    """
    return A * np.exp(-1./b * x) + c

def compute_bootstrap_error(ics, grid, data):
    """
    Description: 
        Compute bootstrapping error for AIMS simulations over all ICs. 
        This is a measurement of the error by sampling with replacement over the ICs included in the analysis of the data.
    Arguments:
        1) ics: an array of ints for the initial conditions
        2) grid: an array for the grid points
        3) data: an array of the data on which to perform bootstrapping
    Returns:
        1) errors: a numpy array of the error at each time step (grid point)
    """
    error = np.zeros(len(grid))
    for k in range(len(grid)):
        sample_inds = np.arange(len(ics))
        resample = [ [data[x,k] for x in np.random.choice(sample_inds, size=(len(sample_inds)), replace=True)] for _ in range(1000) ]
        resampled_means = np.mean(resample, axis=1)
        std = np.std(resampled_means)
        error[k] = std

    return error
