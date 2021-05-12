import pytest
from fat.fat import FAT

def test_FAT():
    """Test instantiation of FAT class object"""

    fmsdir = '../../ethylene/' # Main directory containing all FMS simulations
    datadir = './fat_data/' # directory to which FAT data is stored 

    ic_dict = {}
    ics = [x for x in range(0,10)] # list of initial conditions for initiating FMS simulations
    dirlist = {}
    for ic in ics:
        dirlist['%d' %ic] = fmsdir + ('%04d/' %ic) # index of paths to all individual FMS simulations

    eth_fat = FAT(ics, dirlist, datadir)

    assert eth_fat.nstates == 2


