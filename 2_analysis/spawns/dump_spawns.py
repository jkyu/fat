from shutil import copyfile
import numpy as np
import os
import pickle
import sys

def sort_spawns(ics, datadir, dirlist, nstates):

    print('Loading excited state trajectories and extracting population information.')
    interp_populations = {}
    states = {}
    spawn_times = {}
    for i in range(nstates):
        states['s%d' %i] = []

    spawns = []

    ''' Grab population information out of all ICs and bin that onto a uniform 1 fs time step time grid '''
    count = 0
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        fmsdir = dirlist['%d' %ic]
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_id = tbf['tbf_id']
            state_id = tbf['state_id']
            if tbf_id > 1:
                parent_id = tbf['spawn_info']['parent_state']
                spawn_string = 's%d_to_s%d' %(parent_id, state_id)
                print('%s: %s' %(tbf_key, spawn_string))

                if not os.path.isdir(spawn_string):
                    os.mkdir(spawn_string)
                geomfile = fmsdir+'/Spawn.%d' %(tbf_id)
                copyfile(geomfile, './%s/%s.xyz' %(spawn_string, tbf_key))
    
''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1_collect_data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
dirlist = fmsinfo['dirlist']
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
sort_spawns(ics, datadir, dirlist, nstates)
