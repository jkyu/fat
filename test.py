from fat import *
"""
Test script for the fat module.
Authored by jkyu.
"""

def example_parse_data():
    """
    Example for using the fat data management system for FMS90 simulations.
    Request example data from jkyu. At present, the example data has not been published, so will not be made publicly available.
    """
    fmsdir = './eth_data/' # Main directory containing all FMS simulations
    ic_dict = {}
    # ics = [x for x in range(0,10)] + [12, 16, 27, 35, 44, 55, 64, 68, 76, 81] + [x for x in range(90,100)]
    ics = [x for x in range(0,10)] # list of initial conditions for initiating FMS simulations
    dirlist = {}
    for ic in ics:
        dirlist['%d' %ic] = fmsdir + ('%04d/' %ic) # index of paths to all individual FMS simulations
    datadir = './example_data/' # directory to which fat data is stored 
    eth_fat = fat(ics, dirlist, datadir, parse_extensions=False, save_to_disk=True)

def example_population():
    """
    Example for computing the population dynamics using fat from the data collected by fat.
    """
    datadir = './example_data/' # directory to which fat data is stored 
    fmsinfo = pickle.load(open('%s/fmsinfo.pickle' %datadir, 'rb'))
    picklefiles = fmsinfo['datafiles']
    datafiles = [ x for x in picklefiles ] 
    ics = fmsinfo['ics']
    nstates = fmsinfo['nstates']
    tgrid = np.arange(0, 200, 5) # edit the last number to change the grid spacing
    population_dynamics = compute_populations(ics, datafiles, tgrid, nstates, datadir=datadir)

    average_populations = population_dynamics['populations']
    ic_populations = population_dynamics['ic_populations']
    target_state = 1
    exp_fit, time_constant, time_constant_error = compute_exponential_fit(tgrid, average_populations, ic_populations, target_state)
    print('Population decay constant: $\\tau=%d \pm %d fs)' %(time_constant, time_constant_error))

    figdir = './figures/'
    errors = population_dynamics['errors']
    plot_population_dynamics(tgrid, average_populations, errors, exp_fit, time_constant, time_constant_error, figdir=figdir)


if __name__=='__main__':

    example_parse_data()
    example_population()

