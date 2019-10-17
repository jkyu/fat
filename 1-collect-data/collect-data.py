import pickle
import mdtraj as md
import numpy as np
import os
import sys
import math
'''
This script collects a bunch of raw data from the FMS outputs. 
One pickle file for each FMS simulation is saved to disk because
the data processing is super slow when systems are large (e.g.
protein). These pickle files are then used in the following 
analysis scripts. 
'''
def get_populations(popfile):
    '''
    Parse Amp.x file to pull out population information over time
    for the trajectory. For the population information, we record
    the amplitude norm (in the second column).
    '''
    time_steps = []
    amplitudes = []
    with open(popfile, 'rb') as f:
        _ = f.readline() # header line to get rid of
        for line in f: 
            a = line.split()
            time_step = float(a[0]) * 0.024188425
            amplitude = float(a[1])
            time_steps.append(time_step)
            amplitudes.append(amplitude)

    time_steps = np.array(time_steps)
    amplitudes = np.array(amplitudes)

    return time_steps, amplitudes 

def get_energies(enfile):
    ''' 
    Parse PotEn.x file to pull out electronic energies over time
    computed at the center of the TBF. We record the electronic 
    energy on S0 and on S1 for the location of the TBF at each time 
    step. Time is in column 1, S0 energy is in column 2, S1 energy
    is in column 3 and column 4 is the total energy (labeled Eclass).
    The total energy should stay constant throughout the trajectory.
    These energies are reported in atomic units. 
    '''
    time_steps  = []
    s0_energies = []
    s1_energies = []
    e_total = []
    energies = {}
    with open(enfile, 'rb') as f:
        _ = f.readline() # header line
        for line in f:
            a = line.split()
            time_step = float(a[0]) * 0.024188425
            s0_energy = float(a[1])
            s1_energy = float(a[2])
            etot = float(a[3])
            time_steps.append(time_step)
            s0_energies.append(s0_energy)
            s1_energies.append(s1_energy)
            e_total.append(etot)
    energies['s0'] = np.array(s0_energies)
    energies['s1'] = np.array(s1_energies)
    energies['total'] = np.array(e_total)

    time_steps = np.array(time_steps)

    return time_steps, energies

def get_transition_dipoles(tdipfile):
    '''
    Parse TDip.x file to pull out transition dipoles between
    S0 and S1 for each time step. We record the time step
    and the magnitude of the transition dipole. The time step
    is in the first column and the magnitude of the transition
    dipole is in the second column. Columns 3-5 are for the xyz
    components of the transition dipole. Here, we only need the 
    magnitude. The TDip.x file reports transition dipoles between
    the ground state and the excited states labeled Mag.x in the 
    header, with x indicating the excited state. So Mag.2 indicates
    the magnitude of the transition dipole between states 1 and 2, 
    which are S0 and S1 respectively.
    '''
    time_steps = []
    transition_dipoles = []
    with open(tdipfile, 'rb') as f:
        _ = f.readline() # header line to get rid of
        for line in f: 
            a = line.split()
            time_step = float(a[0]) * 0.024188425
            tdip = float(a[1])
            time_steps.append(time_step)
            transition_dipoles.append(tdip)

    time_steps = np.array(time_steps)
    transition_dipoles = np.array(transition_dipoles)

    return time_steps, transition_dipoles

def get_spawn_info(dirname, ic):

    spawns = []
    if os.path.isfile(dirname + 'Spawn.log'):
        with open(dirname + 'Spawn.log', 'rb') as f:
            _ = f.readline()
            for line in f:
                spawn = {}
                a = line.split()
                spawn['spawn_time']    = float(a[1]) * 0.024188425
                spawn['spawn_id']      = int(a[3])
                spawn['spawn_state']   = int(a[4])
                spawn['parent_id']     = int(a[5])
                spawn['parent_state']  = int(a[6])
                spawn['initcond']      = ic
                spawn['population_transferred'] = population_transfer(dirname, int(a[3]))
                spawns.append(spawn)

    return spawns

def population_transfer(dirname, spawn_id):

    a = None
    with open(dirname + 'Amp.%d' %(spawn_id), 'rb') as f:
        f.seek(-2, os.SEEK_END)     # Jump to second to last byte in file
        while f.read(1) != b'\n':   # Until EOL for previous line is found,
            f.seek(-2, os.SEEK_CUR) # jump one byte back from the current byte
        last = f.readline()
        a = list(map(float, last.split()))

    return a[1]

def get_positions(xyzfile, prmtop):
    '''
    Use MDtraj to load the trajectories from a combination of the appropriate prmtop
    file and position file (the latter from FMS).
    '''
    trajectory = md.load_xyz(xyzfile, prmtop)
    return trajectory

def get_tbf_data(dirname, ic, tbf_id, sys_name):

    prmtop   = dirname + '%s.prmtop' % sys_name
    enfile   = dirname + 'PotEn.%d' %tbf_id
    xyzfile  = dirname + 'positions.%d.xyz' %tbf_id
    popfile  = dirname + 'Amp.%d' %tbf_id
    tdipfile = dirname + 'TDip.%d' %tbf_id

    trajectory = get_positions(xyzfile, prmtop)
    time_steps, populations = get_populations(popfile)
    _, energies = get_energies(enfile)
    _, transition_dipoles = get_transition_dipoles(tdipfile)
    
    tbf_data = {}
    tbf_data['initcond'] = ic
    tbf_data['tbf_id']   = tbf_id
    tbf_data['energies'] = energies
    tbf_data['trajectory']  = trajectory
    tbf_data['time_steps']  = time_steps
    tbf_data['populations'] = populations
    tbf_data['transition_dipoles'] = transition_dipoles

    return tbf_data

def collect_tbfs(initconds, dirname, sysname):
    '''
    Gather TBFs in MDTraj and dump to disk to make subsequent analyses faster. 
    '''
    for ic in initconds:

        data = {}
        dirname = fmsdir + ('%d/' %ic)

        '''
        Parent TBF
        '''
        tbf_id = 1
        print('%02d-%02d' %(ic, tbf_id))

        tbf_data = get_tbf_data(dirname, ic, tbf_id, sysname)
        key = '%02d-%02d' %(ic, tbf_id)
        data[key] = tbf_data
        print('Finish')

        if os.path.isfile(dirname + 'Spawn.log'):
            spawn_info = get_spawn_info(dirname, ic)

            ''' 
            Spawned TBFs
            '''
            for i, spawn in enumerate(spawn_info):

                if len(spawn) > 0:

                    tbf_id = spawn['spawn_id']
                    print('%02d-%02d' %(ic, tbf_id))

                    tbf_data = get_tbf_data(dirname, ic, tbf_id, sysname)
                    key = '%02d-%02d' %(ic, tbf_id)
                    data[key] = tbf_data
                    print('Finish')

        if not os.path.isdir('./data/'):
            os.mkdir('./data/')

        with open('./data/%02d.pickle' %(ic), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

ics = [x for x in range(1,33)]
fmsdir = '/home/jkyu/data/br/5-aims/2-FMS/FMS-'
sysname = 'br' # this is the name of the prmtop file
collect_tbfs(ics, fmsdir, sysname)
