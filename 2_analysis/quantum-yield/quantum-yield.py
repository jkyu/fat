import sys
import numpy as np
import pickle
import mdtraj as md

''' This script requires data from the dihedral angle computation AND the population 
computation here. It checks how much population belongs to each category. '''

def compute_qy():

    data = pickle.load(open('../dihedrals/data/dihedrals.pickle', 'rb'))
    cis_keys = data['cis_keys']
    trans_keys = data['trans_keys']
    ex_keys = data['ex_keys']

    pop_data = pickle.load(open('../population/data/populations.pickle', 'rb'))
    all_pops = pop_data['all_populations']

    trans_pop = 0
    for key in trans_keys:
        trans_pop += all_pops[key][-1]

    cis_pop = 0
    for key in cis_keys:
        cis_pop += all_pops[key][-1]

    total_pop = len(ex_keys)
    print('Total Population: %0.4f' %total_pop)
    print('Cis Photoproduct Population: %0.4f' %cis_pop)
    print('Trans Photoproduct Population: %0.4f' %trans_pop)
    print()

    print('Cis QY: %0.4f' %(cis_pop/total_pop))
    print('Trans QY: %0.4f' %(trans_pop/total_pop))
    print()

    print('Cis QY (product only): %0.4f' %(cis_pop/(cis_pop+trans_pop)))
    print('Trans QY (product only): %0.4f' %(trans_pop/(cis_pop+trans_pop)))

compute_qy()
    

