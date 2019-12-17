import sys
import numpy as np
import pickle
import mdtraj as md

''' This script requires data from the dihedral angle computation. 
It computes the quantum yield for cis and trans photoproducts. 
This is independent of whether you start from cis or trans. That is
dealt with in the compute_dihedrals script. '''

def compute_qy(dihedrals, populations, dihe_key, tbf_keys):

    ''' Compute isomerization quantum yield error by bootstrapping '''
    resampled_qy = []
    for _ in range(1000):
        resampled_ics = np.random.choice(np.arange(10, 50), size=40, replace=True)
        resampled_keys = []
        for ic in resampled_ics:
            rk = [x for x in tbf_keys if int(x.split('-')[0])==int(ic)]
            resampled_keys = resampled_keys + rk
        trans_keys = []
        cis_keys = []
        for key in resampled_keys:
            if dihedrals[key][dihe_key][-1] > 270:
                cis_keys.append(key)
            else:
                trans_keys.append(key)

        trans_pop = 0
        for key in trans_keys:
            trans_pop += populations[key][-1]

        cis_pop = 0
        for key in cis_keys:
            cis_pop += populations[key][-1]

        total_pop = trans_pop + cis_pop
        qy = cis_pop / total_pop
        resampled_qy.append(qy)
    error = np.std(resampled_qy) # the error for cis and trans are the same because the QYs are complements

    ''' Separate ground state TBFs into cis and trans by the dihedral of the TBF at its last time step '''
    trans_keys = []
    cis_keys = []
    for key in tbf_keys:
        if dihedrals[key][dihe_key][-1] > 270:
            cis_keys.append(key)
        else:
            trans_keys.append(key)

    print('Trans TBFs: ')
    print(trans_keys)
    print('Cis TBFs: ')
    print(cis_keys)
    print()

    trans_pop = 0
    for key in trans_keys:
        trans_pop += populations[key][-1]

    cis_pop = 0
    for key in cis_keys:
        cis_pop += populations[key][-1]

    ''' Compute the quantum yield as the amount of population carried by TBFs in each photoproduct category
    divided by the total population that is on the ground state. Not all of the population has decayed
    to the ground state, so this is slightly different from dividing by the total population of all of the
    initial conditions. But most simulations should have >98% of the population on the ground state anyway. '''
    total_pop = trans_pop + cis_pop
    qy_cis = cis_pop / total_pop
    qy_trans = trans_pop / total_pop

    print('Total Population: %0.4f' %total_pop)
    print('Cis Photoproduct Population: %0.4f' %cis_pop)
    print('Trans Photoproduct Population: %0.4f' %trans_pop)
    print()

    print('Cis QY: %0.4f +/- %0.4f' %(qy_cis, error))
    print('Trans QY: %0.4f +/- %0.4f' %(qy_trans, error))
    print()

    return cis_pop, trans_pop, error

if __name__=='__main__':
    dihedral_data = pickle.load(open('./data/dihedrals.pickle', 'rb'))
    dihe_keys = dihedral_data['dihedral_names']
    dihedrals = dihedral_data['dihedrals_state_specific']
    populations = dihedral_data['populations']
    ''' Take only the subset of TBFs that have decayed to the ground state '''
    state_ids = dihedral_data['state_ids']
    tbf_keys = dihedral_data['tbf_keys']
    gs_keys = [ x for x in tbf_keys if state_ids[x]==0 ]
    compute_qy(dihedrals, populations, dihe_keys[0], gs_keys)
