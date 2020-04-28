from fat import *
"""
Test script for the fat module.
Authored by jkyu.
"""

def example_ethylene():
    """
    Example for using the fat data management system for FMS90 simulations.
    Request example data from jkyu. At present, the example data has not been published, so will not be made publicly available.
    """
    fmsdir = './eth_data/' # Main directory containing all FMS simulations
    ic_dict = {}
    ics = [x for x in range(0,10)] # list of initial conditions for initiating FMS simulations
    dirlist = {}
    for ic in ics:
        dirlist['%d' %ic] = fmsdir + ('%04d/' %ic) # index of paths to all individual FMS simulations
    datadir = './example_data/' # directory to which FMS90 data is stored 
    eth_fat = fat(ics, dirlist, datadir, parse_extensions=False, save_to_disk=True)

if __name__=='__main__':

    example_ethylene()

