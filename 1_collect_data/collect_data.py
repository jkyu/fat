import glob
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
    We do this from Amp.x instead of N.dat because the coefficient
    for each TBF at each time step matters for computing observables.
    For population decay, you can just use N.dat, but this is more
    general. 
    '''
    time_steps_au = []
    amplitudes = []
    with open(popfile, 'rb') as f:
        _ = f.readline() # header line to get rid of
        for line in f: 
            a = line.split()
            time_step_au = float(a[0])
            amplitude = float(a[1])
            time_steps_au.append(time_step_au)
            amplitudes.append(amplitude)

    time_steps_au = np.array(time_steps_au)
    amplitudes = np.array(amplitudes)

    return time_steps_au, amplitudes 

def get_energies(enfile):
    ''' 
    Parse PotEn.x file to pull out electronic energies over time
    computed at the center of the TBF. We record the electronic 
    energy on S0 and on S1 for the location of the TBF at each time 
    step. Time is in column 1, S0 energy is in column 2, S.n energy
    where n = state number (1-indexed) are the next few columns. 
    The final column is the total classical energy. The classical 
    energy should stay constant throughout the FMS simulation. 
    These energies are reported in atomic units. 
    '''
    time_steps_au  = []
    e_class = []
    energies = {}
    with open(enfile, 'rb') as f:
        header = f.readline() # header line
        nstates = int(len(header.split()) - 2)
        for i in range(0, nstates):
            energies['s%d' %i] = []
        for line in f:
            a = line.split()
            time_step_au = float(a[0])
            time_steps_au.append(time_step_au)
            e_class.append(float(a[-1]))
            for i in range(0, nstates):
                energies['s%d' %i].append(float(a[i+1]))
    data = {}
    for key in energies.keys():
        data[key] = np.array(energies[key])
    data['total'] = np.array(e_class)
    time_steps_au = np.array(time_steps_au)

    return time_steps_au, data, nstates

def get_transition_dipoles(tdipfile, nstates):
    '''
    Parse TDip.x file to pull out transition dipoles between
    S0 and S1 for each time step. We record the time step
    and the magnitude of the transition dipole. The time step
    is in the first column and the magnitude of the transition
    dipole is in the second column. Columns 2.x, 2.y, 2.z are for 
    the xyz components of the transition dipole. Here, we only need 
    the magnitude. The TDip.x file reports transition dipoles between
    the ground state and the excited states labeled Mag.n in the 
    header, with n indicating the excited state (1-indexed). 
    So Mag.2 indicates the magnitude of the transition dipole between 
    states 1 and 2, which are S0 and S1 respectively. 
    The transition_dipoles dictionary is indexed by the state label of the
    excited state and only includes transitions between the ground and 
    excited states, since that is FMS provides. 
    '''
    time_steps_au = []
    transition_dipoles = {}
    with open(tdipfile, 'rb') as f:
        _ = f.readline() # header line to get rid of
        # header = f.readline() # header line
        # nstates = int((len(header.split()) - 1)//4)
        for i in range(0, nstates):
            transition_dipoles['s%d' %i] = []
        for line in f:
            a = line.split()
            time_steps_au.append(float(a[0]))
            for i in range(0, nstates):
                transition_dipoles['s%d' %i].append(float(a[i+1]))
    data = {}
    for key in transition_dipoles.keys():
        data[key] = np.array(transition_dipoles[key])
    time_steps_au = np.array(time_steps_au)

    return time_steps_au, data

def get_spawn_info(dirname, ic):

    spawns = []
    if os.path.isfile(dirname + 'Spawn.log'):
        with open(dirname + 'Spawn.log', 'rb') as f:
            _ = f.readline()
            for line in f:
                spawn = {}
                a = line.split()
                spawn['spawn_time']    = float(a[1]) * 0.024188425
                spawn['spawn_time_au'] = float(a[1])
                spawn['spawn_id']      = int(a[3])
                spawn['spawn_state']   = int(a[4]) - 1 # since these are 1-indexed
                spawn['parent_id']     = int(a[5])
                spawn['parent_state']  = int(a[6]) - 1
                spawn['initcond']      = ic
                spawn['population_transferred'] = population_transfer(dirname, int(a[3]))
                spawns.append(spawn)

    return spawns

def population_transfer(dirname, spawn_id):

    a = 0
    if not os.path.isfile(dirname+'Amp.%d' %spawn_id):
        return a
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

def get_extension(extdir, prmtop):
    '''
    Read in coordinate information from AIMD extensions for ground state AIMS TBFs.
    The time grid associated with the extension is comuted from a time step and the
    number of frames in the extension trajectory. The time step is taken as the 
    terachem default: 0.5 fs. 
    '''
    xyzfile = extdir + 'scr.coord/coors.xyz'
    trajectory_ext = md.load_xyz(xyzfile, prmtop)

    aimd_input = open(extdir + 'aimd.in', 'r')
    tstep = 0.5 # fs
    tgrid = np.array([x*tstep for x in range(len(trajectory_ext))])

    return tgrid, trajectory_ext

def get_tbf_data(dirname, ic, tbf_id, prmtop, extensions=False):

    # prmtop   = dirname + '%s.prmtop' % sys_name
    enfile   = dirname + 'PotEn.%d' %tbf_id
    xyzfile  = dirname + 'positions.%d.xyz' %tbf_id
    popfile  = dirname + 'Amp.%d' %tbf_id
    tdipfile = dirname + 'TDip.%d' %tbf_id

    # Handles the case where there is no TBF data despite a spawning point.
    if not os.path.isfile(popfile):
        raise Exception('Directory for this IC does not have FMS outputs to process.')

    trajectory = get_positions(xyzfile, prmtop)
    time_steps_au, populations = get_populations(popfile)
    _, energies, nstates = get_energies(enfile)
    _, transition_dipoles = get_transition_dipoles(tdipfile, nstates)

    time_steps = time_steps_au * 0.024188425 

    # Catch for staggered array sizes due to running simulations.
    if not len(time_steps) == len(trajectory):
        nstep = np.min([len(time_steps), len(trajectory)])
        time_steps = time_steps[:nstep]

    ''' Handling for AIMD extensions of ground state TBFs from the FMS simulations.
    The coordinates and time steps for the extension trajectory are appended to the 
    corresponding arrays from the FMS TBFs. The population at the last time point
    in the FMS TBF is taken to be constant for the entire AIMD extension trajectory
    and the populations array is extended with this constant value to be the same
    length as the extended time and position arrays. The energies and transition dipole
    arrays are not handled here because those are only used in fluorescence calculations,
    where only excited states are relevant. They don't break anything, since we won't
    encounter array length mismatches when only excited states are considered. 
    If this is a problem later for whatever reason, here is a long comment to help with
    resolving that. '''
    if os.path.exists(dirname+'ext_%d' %tbf_id) and extensions:
        extdir = dirname + 'ext_%d/' %tbf_id
        time_steps_extension, trajectory_extension = get_extension(extdir, prmtop)
        time_steps_extension = time_steps_extension + time_steps[-1]
        trajectory = md.join([trajectory, trajectory_extension[1:]])
        time_steps = np.concatenate([time_steps, time_steps_extension[1:]])
        populations_extension = np.zeros_like(time_steps)
        populations_extension[:len(populations)] = populations
        populations_extension[len(populations):] = np.array([[populations[-1]]*(len(time_steps_extension)-1)])
        populations = populations_extension
    
    tbf_data = {}
    tbf_data['initcond'] = ic
    tbf_data['tbf_id']   = tbf_id
    tbf_data['energies'] = energies
    tbf_data['nstates'] = nstates
    tbf_data['trajectory']  = trajectory
    tbf_data['time_steps']  = time_steps
    tbf_data['time_steps_au'] = time_steps
    tbf_data['populations'] = populations
    tbf_data['transition_dipoles'] = transition_dipoles

    return tbf_data

def collect_tbfs(initconds, dirlist, prmtop, initstate, write_fmsinfo=True, extensions=False):
    '''
    Gather TBFs in MDTraj and dump to disk to make subsequent analyses faster. 
    '''
    for ic in initconds:

        data = {}
        dirname = dirlist['%d' %ic]

        '''
        Parent TBF
        '''
        tbf_id = 1
        print('%04d-%04d' %(ic, tbf_id))

        tbf_data = get_tbf_data(dirname, ic, tbf_id, prmtop, extensions=False)
        tbf_data['spawn_info'] =  None
        tbf_data['state_id'] = initstate

        key = '%04d-%04d' %(ic, tbf_id)
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
                    print('%04d-%04d' %(ic, tbf_id))

                    tbf_data = get_tbf_data(dirname, ic, tbf_id, prmtop, extensions)
                    if not tbf_data==None:
                        tbf_data['spawn_info'] = spawn
                        tbf_data['state_id'] = spawn['spawn_state']
                        key = '%04d-%04d' %(ic, tbf_id)
                        data[key] = tbf_data
                    print('Finish')

        if not os.path.isdir('./data/'):
            os.mkdir('./data/')

        with open('./data/%04d.pickle' %(ic), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    simulation_data = {}
    ics = glob.glob('./data/0*.pickle')
    ics = [int(os.path.basename(x).split('.')[0]) for x in ics]
    ics.sort()
    simulation_data['ics'] = ics
    simulation_data['nstates'] = tbf_data['nstates']
    if write_fmsinfo:
        with open('./data/fmsinfo.pickle', 'wb') as handle:
            pickle.dump(simulation_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

ics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 27, 35, 44, 55, 64, 68, 76, 81]
ics = ics + [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
fmsdir = '../../' # Main directory containing all FMS simulations
dirlist = {}
for ic in ics:
    dirlist['%d' %ic] = fmsdir + ('%04d/' %ic) # index of paths to all individual FMS simulations
topfile = '../../ethylene.pdb' # this is the name of the topology file (.prmtop, .pdb, etc.)
initstate = 1 # we start on S1 for this system. All of my stored data is 0-indexed starting from the ground state. 
collect_tbfs(ics, dirlist, topfile, initstate, write_fmsinfo=True, extensions=True)
# set write_fmsinfo to False to avoid dumping out the fmsinfo file that contains
# overall dynamics information, like IC labels, nstates, etc. Helpful if you
# only want to process one simulation without clobbering a previous fmsinfo file.
