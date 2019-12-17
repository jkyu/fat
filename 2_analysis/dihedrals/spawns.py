import sys
import numpy as np
import pickle
import mdtraj as md
from quantum_yield import compute_qy

def sort_spawns(dihedral_data, dihe_key):

    tbf_keys = dihedral_data['tbf_keys']
    state_ids = dihedral_data['state_ids']
    dihedrals = dihedral_data['dihedrals_state_specific']
    dihe_keys = dihedral_data['dihedral_names']
    populations = dihedral_data['populations']
    tgrid = dihedral_data['tgrid']

    ''' Check 1st point of spawned TBFs '''
    class1_keys = []
    class2_keys = []
    class3_keys = []
    spawn_ds = {}
    spawn_vels = {}
    gs_keys = [ x for x in tbf_keys if state_ids[x]==0 ]
    for key in gs_keys:
        ds = dihedrals[key][dihe_key]
        ''' Find first non-NaN index '''
        ind = np.where(np.isnan(ds))[0][-1] + 1
        spawn_ds[key] = ds[ind]
        ''' From bond_selectivity.py, we saw that there are at least 3 categories of CIs
        based on dihedral angle measurements. We separate them here into those 3 classes. '''
        if ds[ind] > 240:
            class1_keys.append(key)
        elif ds[ind] < 210:
            class3_keys.append(key)
        else: class2_keys.append(key)
        parent_key = '%04d-0001' %(int(key.split('-')[0]))
        parent_ds = dihedrals[parent_key][dihe_key]
        spawn_vel = (ds[ind+1] - parent_ds[ind-1]) / (tgrid[ind+1] - tgrid[ind-1])
        spawn_vels[key] = spawn_vel

    # print('Spawns > 240: ')
    # cis_pop, trans_pop, error, cis_keys, trans_keys = compute_qy(dihedrals, populations, dihe_keys[0], class1_keys)
    # print()
    print('Spawns > 210 and < 240: ')
    cis_pop, trans_pop, error, cis_keys, trans_keys = compute_qy(dihedrals, populations, dihe_keys[0], class2_keys)
    print()
    for key in trans_keys:
        print(key, spawn_vels[key])
    sys.exit()
    print('Spawns < 200')
    cis_pop, trans_pop, error, cis_keys, trans_keys = compute_qy(dihedrals, populations, dihe_keys[0], class3_keys)

dihedral_data = pickle.load(open('./data/dihedrals.pickle', 'rb'))
dihe_keys = dihedral_data['dihedral_names']
sort_spawns(dihedral_data, dihe_keys[0])
    

