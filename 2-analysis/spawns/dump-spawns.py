from shutil import copyfile
import mdtraj as md
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
import sys

def sort_spawns(ics, datadir, fmsdir, nstates, topfile, angles):

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
            state_id = tbf['state_id']
            if tbf_id > 1:
                parent_id = tbf['spawn_info']['parent_state']
                print('%s spawns on state s%d from state s%d' %(tbf_key, state_id, parent_id))

                states['s%d' %state_id].append(tbf_key)
                spawn_times[tbf_key] = tbf['time_steps'][0]
                geomfile = fmsdir+'/%04d/Spawn.%d' %(ic, tbf_id)
                spawn_geom = md.load_xyz(geomfile, topfile)
                HCH_angles = md.compute_angles(spawn_geom, angles)
                HCH_angles = [ 180*x/np.pi for x in HCH_angles[0] ]
                if HCH_angles[0] > 140 or HCH_angles[1] > 140:
                    print('2,5,3 and 0,4,1: ', HCH_angles )
                    dist = md.compute_distances(spawn_geom, [[2,3], [0,1]])*10.
                    print('2,3 and 0,1 distances: ', dist[0])

                if not os.path.isdir('./s%d' %state_id):
                    os.mkdir('./s%d' %state_id)
                copyfile(geomfile, './s%d/%s.xyz' %(state_id, tbf_key))
    
''' 
Specify the time grid and ICs to use. 
Can use a coarser time grid than is used here and it shouldn't change the result.
'''
datadir = '../../1-collect-data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
topfile = '../../../ethylene.pdb'
fmsdir = '../../../'
angles = [ [2,5,3], [0,4,1] ]
sort_spawns(ics, datadir, fmsdir, nstates, topfile, angles)
