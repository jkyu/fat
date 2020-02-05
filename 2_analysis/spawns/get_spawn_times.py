import numpy as np
import os
import pickle
import sys

def sort_spawns(ics, datadir, nstates):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    spawn_times = {}
    for i in range(nstates):
        states['s%d' %i] = []

    ''' Grab population information out of all ICs and bin that onto a uniform 1 fs time step time grid '''
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_id = tbf['tbf_id']
            tbf_state = tbf['spawn_info']['tbf_state']
            print(tbf_key, tbf_id, tbf_state)
            if tbf_id>1 and tbf_state==0:
                
                parent_id = tbf['spawn_info']['parent_state']
                print('%s spawns on state s%d from state s%d' %(tbf_key, tbf_state, parent_id))

                states['s%d' %tbf_state].append(tbf_key)
                spawn_times[tbf_key] = tbf['time_steps'][0]

    data = {}
    data['spawn_times'] = spawn_times
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/spawn_times.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
dihedral = [  ]
sort_spawns(ics, datadir, nstates)
