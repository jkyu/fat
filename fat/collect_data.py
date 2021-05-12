"""
FMS90 Analysis Tool (fat) for the extraction, management and analysis of data produced by FMS90 simulations.
Authored by Jimmy K. Yu (jkyu).
"""
import numpy as np
import os, sys, math, glob, pickle


class fat(object):
    """
    This class is a data management system for FMS90 simulations. 
    Instantiating the class object calls two functions:
        - collect_tbfs() parses data from the FMS outputs for each independent AIMS simulation
        - write_fmsinfo() provides parameters for the full AIMS simulation

    Arguments
    -------------------------------------
        ics (list(int))
            - list of integers that index the initial conditions (ics) [Required]
        dirlist (list(str))
            - list of paths to the simulation directories [Required]
        datadir (str)
            - path to the pickle file to dump [Default: ./data/]
        partitioned_ics (dict)
            - dictionary containing a user-specified partitioning of the ICs. 
            - not required and has a niche use case. [Default: None]
        parse_extensions (bool)
            - set to True to process single state AIMD extensions [Default: False]
        save_to_disk_ic (bool)
            - set to True to save the individual FMS data files to disk [Default: False]
        save_to_disk_full (bool)
            - set to True to save a single file containing the data for the full FMS simulation to disk [Default: False]

    Special Notes
    -------------------------------------
        - If the data is not dumped to disk by initial condition, it can be accessed directly within the fmsdata variable of the object or by saving the full FMS simulation. 
        - This is OK if the system is small and the simulation is short, e.g., ethylene.
        - By default, one pickle file is for each FMS simulation is saved to disk because the data processing is incredibly slow when systems are large, e.g, proteins.
        - Because these pickle files are stored separately, the data for the full simulation does not need to be processed in one go and updates to a single simulation do not require rerunning the data collection procedure. 
        - fmsdata is a dictionary of the individual AIMS simulations indexed by the initial condition number. 
    """
    def __init__(
            self, 
            ics, 
            dirlist, 
            datadir='./data/',
            partitioned_ics=None, 
            parse_extensions=False, 
            save_to_disk_ic=False, 
            save_to_disk_full=False
            ):

        self.ics = ics
        self.dirlist = dirlist
        self.parse_extensions = parse_extensions

        self.fmsdata, self.nstates, self.tbf_states = self.collect_tbfs(
                ics, 
                dirlist, 
                datadir, 
                parse_extensions, 
                save_to_disk_ic, 
                save_to_disk_full
                )

        if save_to_disk_ic: 
            self.fmsinfo = self.write_fmsinfo(self.dirlist, datadir, self.nstates, partitioned_ics)


    def get_populations(self, popfile):
        """
        Description 
        -------------------------------------
            - Parse Amp.x file to pull out population information over time for the trajectory. 
            - For the population information, the amplitude norm (in the second column of Amp.x) is recorded.
            - This information is collected from Amp.x instead of N.dat because the coefficient for each TBF at each time step matters for computing observables.
            - For the analysis of population decay, N.dat alone is sufficient, but this is more flexible.

        Arguments
        -------------------------------------
            popfile (str)
                - path to an Amp.x file written by FMS90 for TBF #x

        Returns 
        -------------------------------------
            time_steps_au (np.ndarray)
                - numpy array containing the FMS time steps in atomic units
            amplitudes (np.ndarray)
                - numpy array containing the amplitude of the TBF at each time step. 
        """
        time_steps_au = []
        amplitudes = []

        with open(popfile, 'rb') as f:
            _ = f.readline() # remove header line
            for line in f: 
                a = line.split()
                time_step_au = float(a[0])
                amplitude = float(a[1])
                time_steps_au.append(time_step_au)
                amplitudes.append(amplitude)
    
        time_steps_au = np.array(time_steps_au)
        amplitudes = np.array(amplitudes)
    
        return time_steps_au, amplitudes 
    

    def get_energies(self, enfile):
        """
        Description
        -------------------------------------
            - Parse PotEn.x file fir electronic energies over time computed at the center of TBF #x.
            - The electronic energy on each adiabatic state is collected for each TBF at each time step. 
            - Time is collected from column 1 and the S_n energies (where n = 1-indexed state number) are in the following columns.
            - The final column is the total energy, which should stay approximately constant throughout the FMS simulation. 
            - These energies are recorded in atomic units. 

        Arguments
        -------------------------------------
            enfile (str)
                - path to PotEn.x file written by FMS90 for TBF #x

        Returns 
        -------------------------------------
            time_steps_au (np.ndarray)
                - numpy array containing the FMS time steps in atomic units
            data (dict)
                - dictionary containing arrays of energies for an adiabatic 
                  state during the course of the simulation indexed by the 
                  adiabatic state label, e.g. 's_0' or 's_1'.
            nstates (int)
                - integer specifying the number of electronic states included
                  in the FMS simulation
        """
        time_steps_au  = []
        e_class = []
        energies = {}

        with open(enfile, 'rb') as f:
            header = f.readline() # remove header line
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
    

    def get_transition_dipoles(self, tdipfile, nstates):
        """
        Description
        -------------------------------------
            - Parse TDip.x file to pull out transition dipoles between S_0 and S_n for each time step (where n>1 indexes the adiabatic excited states). 
            - The time step is in the first column and the magnitude of the transition dipole from S_0 to S_n is in the nth column (labeled Mag.n). 
            - Columns labeled n.x, n.y, n.z the xyz components of the S_0->S_n transition dipole. 
            - For the purpose of computing a fluorescence spectrum, only the magnitude is required. 

        Arguments
        -------------------------------------
            tdipfile (str)
                - path to TDip.xfile written by FMS90 for TBF #x
            nstates (int)
                - integer specifying the number of electronic states 
                  included in the FMS simulation
        Returns 
        -------------------------------------
            time_steps_au (np.ndarray)
                - numpy array containing the FMS time steps in atomic units
            data (dict)
                - dictionary containing arrays of transition dipoles from S_0->S_n during the course of the simulation indexed by the adiabatic state label of the excited state, e.g., 's_1', 's_2', etc. 
        """
        time_steps_au = []
        transition_dipoles = {}

        with open(tdipfile, 'rb') as f:
            _ = f.readline() # remove header line
            for i in range(1, nstates):
                transition_dipoles['s%d' %i] = []
            for line in f:
                a = line.split()
                time_steps_au.append(float(a[0]))
                for i in range(1, nstates):
                    transition_dipoles['s%d' %i].append(float(a[i]))

        data = {}
        for key in transition_dipoles.keys():
            data[key] = np.array(transition_dipoles[key])
        time_steps_au = np.array(time_steps_au)
    
        return time_steps_au, data

    
    def get_spawn_info(self, dirname, ic, initstate):
        """
        Description
        -------------------------------------
            - Parse Spawn.log to obtain information about the spawning throughout the AIMS simulation. 
            - A special case is created for the initial TBF (not detailed by Spawn.log) and for simulations in which no spawning occurs.

        Arguments 
        -------------------------------------
            dirname (str)
                - path to the directory of the AIMS simulation
            ic (int)
                - index of the initial condition that launched the AIMS simulation
            initstate (int)
                - the adiabatic state on which the AIMS simulation starts

        Returns
        -------------------------------------
            spawns (list(dict))
                - a list containing a dictionaries of the spawn information for each child TBF.
            Each spawn dictionary is indexed by the following keys:
                spawn_time (float)
                    - spawn time of the TBF in units of fs
                spawn_time_au (float)
                    - a float that indicates the spawn time of the TBF in atomic units
                tbf_id (int)
                    - index of the TBF, i.e., the int x in Amp.x, positions.x.xyz, etc.
                tbf_state (int)
                    - adiabatic state of the TBF (zero-indexed), i.e. 0 = S0, 1 = S1
                parent_id (int)
                    - tbf_id of the parent TBF (from which the current TBF spawned)
                parent_state (int)
                    - adiabatic state of the parent TBF (zero-indexed)
                initcond (int)
                    - index of the initial condition that launched the AIMS simulation
                population_transferred (float)
                    - population that remains on this TBF by the end of the simulation 
                      or at the time of TBF death
        """
        spawns = []
        # Special case for the initial TBF 
        spawn = {}
        spawn['spawn_time']    = None
        spawn['spawn_time_au'] = None
        spawn['tbf_id']        = int(1)
        spawn['tbf_state']     = int(initstate)
        spawn['parent_id']     = None
        spawn['parent_state']  = None
        spawn['initcond']      = ic
        spawn['population_transferred'] = None
        spawns.append(spawn)

        # Iterate through spawned TBFs
        if os.path.isfile(dirname + 'Spawn.log'):
            with open(dirname + 'Spawn.log', 'rb') as f:
                _ = f.readline()
                for line in f:
                    spawn = {}
                    a = line.split()
                    spawn['spawn_time']    = float(a[1]) * 0.024188425
                    spawn['spawn_time_au'] = float(a[1])
                    spawn['tbf_id']        = int(a[3])
                    spawn['tbf_state']     = int(a[4]) - 1 # since these are 1-indexed
                    spawn['parent_id']     = int(a[5])
                    spawn['parent_state']  = int(a[6]) - 1
                    spawn['initcond']      = ic
                    spawn['population_transferred'] = self.get_population_transfer(dirname, int(a[3]))
                    spawns.append(spawn)
    
        return spawns

    
    def get_population_transfer(self, dirname, spawn_id):
        """
        Description
        -------------------------------------
            - Obtain the final population remaining on the TBF by the end of the AIMS simulation (or when the TBF dies)
        Arguments
        -------------------------------------
            dirname (str)
                - path to the directory of the AIMS simulation
            spawn_id (int)
                - index for the TBF of interest

        Returns
        -------------------------------------
            (float)
                - final amplitude of the TBF
        """
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

    
    def get_positions(self, xyzfile):
        """
        Description
        -------------------------------------
            - Obtain the coordinates detailing the dynamics of a single TBF 
            - This is also used to parse the coordinates for the AIMD extensions

        Arguments
        -------------------------------------
            xyzfile (str)
                - path to .xyz file that tracks the TBF history

        Returns
        -------------------------------------
            frame_positions (list(np.ndarray))
                - list of numpy arrays containing the coordinates at each time step
            atom_labels (list(str))
                - list containing the atom labels (element symbols) for the system at each frame
        """
        atom_labels = []
        frames_positions = []
        with open(xyzfile, 'r') as f:
            lines = [x for x in f]
            natoms = int(lines[0])
            nframes = int(len(lines) / (natoms+2))
            for frame_idx in range(nframes):
                coords = []
                for atom_idx in range(natoms):
                    line = lines[frame_idx*(natoms+2) + atom_idx + 2]
                    line2 = line.rstrip().rsplit()
                    coord = []
                    coord.append(float(line2[1]))
                    coord.append(float(line2[2]))
                    coord.append(float(line2[3]))
                    coords.append(coord)
                    if frame_idx==0:
                        atom_labels.append(line2[0])
                frames_positions.append(coords)
        frames_positions = np.array(frames_positions, np.float)
    
        return frames_positions, atom_labels
    

    def get_initstate(self, dirname):
        """
        Arguments: 
        -------------------------------------
            1) dirname: a string the provides the path to the directory of the individual AIMS simulation.
        Description:
        -------------------------------------
            Obtains the initial adiabatic state for the AIMS simulation. 
            Since this is collected for each AIMS simulation separately, this can be used for simulations
            in which not all ICs are initiated on the same adiabatic state. 
        Returns:
        -------------------------------------
            1) initstate: an int corresponding to the adiabatic state label (0-indexed), i.e. s0, s1, etc.
        """
        initstate = None
        trajdump1 = dirname + 'TrajDump.1'
        if os.path.isfile(trajdump1):
            with open(trajdump1) as f:
                lines = [x for x in f]
                initstate = int(float(lines[-1].split()[-1])) - 1
        else:
            raise Exception('Could not find the TBF state ID from %s.' %trajdump1)
    
        return initstate
    
    def get_extension(self, extdir, tstep=0.5):
        """
        Description:
        -------------------------------------
            - Read in coordinate information from AIMD extensions for ground state AIMS TBFs.
            - The time grid associated with the extension is comuted from a time step and the number of frames in the extension trajectory. 

        Arguments: 
        -------------------------------------
            extdir (str)
                - path to the directory of the AIMD extension for the TBF
            tstep (float)
                - the step size used for the AIMD simulation. [default = 0.5 fs, the TeraChem default]

        Returns:
        -------------------------------------
            tgrid (np.ndarray)
                - numpy array containing the time steps of the AIMD extension in units of fs.
            trajecotry_ext (list(np.ndarray))
                - list of numpy arrays containing the coordinates for each time step of the AIMD extension.
        """
        xyzfile = extdir + 'scr.coord/coors.xyz'
        trajectory_ext, _ = self.get_positions(xyzfile)
    
        aimd_input = open(extdir + 'aimd.in', 'r')
        tgrid = np.array([x*tstep for x in range(len(trajectory_ext))])
    
        return tgrid, trajectory_ext

    
    def get_tbf_data(self, dirname, ic, tbf_id, parse_extensions=False):
        """
        Description
        -------------------------------------
            - Collect relevant data for each TBF by calling helper functions within the class. 

        Arguments: 
        -------------------------------------
            dirname (str)
                - path to the FMS90 directory for the initial condition. 
            ic (int)
                - initial condition of the AIMS simulation.
            tbf_id (int)
                - index of the TBF of interest
            parse_extensions (bool)
                - set to True to process AIMD extensions. [Default: False]

        Returns:
        -------------------------------------
            tbf_data (dict)
                - a dictionary containing key quantities for each TBF.
                Each tbf_data dictionary contains the following elements:
                    initcond (int)
                        - initial condition of the AIMS simulation.
                    tbf_id (int)
                        - TBF index, i.e., the int in Amp.x, positions.x.xyz, etc.
                    energies (dict)
                        - dictionary containing arrays of energies for an adiabatic state during the course of the simulation indexed by the adiabatic state label, e.g. 's0' or 's1'.
                    nstates (int)
                        - number of states included in the FMS simulation. 
                    trajectory (list(np.ndarray))
                        - list of numpy arrays containing the coordinates for each time step
                    time_steps (np.ndarray)
                        - numpy array containing the FMS time steps in units of fs
                    time_steps_au (np.ndarray)
                        - numpy array containing the FMS time steps in atomic units
                    populations (np.ndarray)
                        - numpy array containing the amplitude of the TBF at each time step. 
                    transition_dipoles (dict)
                        - dictionary containing arrays of transition dipoles from S0->Sn during the course of the simulation indexed by the adiabatic state label of the excited state, e.g., 's1', 's2', etc. 
                    trajectory_atom_labels (list(str))
                        - list of atom labels for each coordinate frame

        Special Notes 
        -------------------------------------
            - If AIMD extensions are requested:
                - The coordinates and time steps for the extension trajectory are appended to the corresponding arrays from the FMS TBFs. 
                - The population at the last time point in the FMS TBF is taken to be constant for the entire AIMD extension trajectory and the populations array is extended with this constant value to be the same length as the extended time and position arrays. 
                - The energies and transition dipole arrays are not handled here because those are only used in fluorescence calculations, where only excited states are relevant. 
                - They don't break anything, since there will not be array length mismatches when only excited states are considered. 
                - If this is a problem later for whatever reason, this should be sufficient information to fix this. 
        """
        enfile   = dirname + 'PotEn.%d' %tbf_id
        xyzfile  = dirname + 'positions.%d.xyz' %tbf_id
        popfile  = dirname + 'Amp.%d' %tbf_id
        tdipfile = dirname + 'TDip.%d' %tbf_id
    
        # Handles the case where there is no TBF data despite a spawning point.
        if not os.path.isfile(popfile):
            raise Exception('Directory for this IC does not have FMS outputs to process.')
    
        trajectory, atom_labels = self.get_positions(xyzfile)
        time_steps_au, populations = self.get_populations(popfile)
        _, energies, nstates = self.get_energies(enfile)
        _, transition_dipoles = self.get_transition_dipoles(tdipfile, nstates)
    
        time_steps = time_steps_au * 0.024188425 
    
        # Catch for staggered array sizes due to running simulations.
        if not len(time_steps) == len(trajectory):
            nstep = np.min([len(time_steps), len(trajectory)])
            time_steps = time_steps[:nstep]
    
        # Handling of AIMD extensions
        if os.path.exists(dirname+'ext_%d' %tbf_id) and parse_extensions:
            extdir = dirname + 'ext_%d/' %tbf_id
            time_steps_extension, trajectory_extension = self.get_extension(extdir)
            time_steps_extension = time_steps_extension + time_steps[-1]
            trajectory = np.concatenate([trajectory, trajectory_extension[1:]])
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
        tbf_data['trajectory_atom_labels'] = atom_labels
    
        return tbf_data
    

    def collect_tbfs(self, initconds, dirlist, datadir, parse_extensions=False, save_to_disk_ic=False, save_to_disk_full=False):
        """
        Description
        -------------------------------------
            - For each initial condition, gather TBFs and dump to disk as pickle files to make subsequent analyses faster. 
            - Goes through all initial conditions given as a list of integer and processes the TBFs in a dict of FMS simulation directories (dirlist) corresponding to the integer associated with the initial condition. 

        Arguments 
        -------------------------------------
            initconds (list(int))
                - list of initial condition indices [Required]
            dirlist (list(str))
                - list of simulation directory paths [Required]
            datadir (str)
                - location of the pickle file to dump [Required]
            parse_extensions (bool)
                - set True to process single state AIMD extensions [Default: False]
            save_to_disk_ic (bool)
                - set True to save individual FMS data files to disk for each initial condition [Default: False]
            save_to_disk_full (bool)
                - set True to save one file containing the data from the full FMS simulation to disk [Default: False]
        Returns
        -------------------------------------
            fmsdata (dict)
                - dictionary containing the dictionaries corresponding to each initial condition indexed by a four digit key corresponding to the intial condition padded with zeros, e.g., initial condition 14 is accessed by key '0014'.
            nstates (int)
                - number of adiabatic states involved in the AIMS simulation. 
                  Used to write the fmsinfo file if requested. 
            tbf_states (dict)
                - dictionary indexed by TBF keys ('%04d-04d' %(ic, tbf_id)) reporting the adiabatic state on which the TBF lives
        """
        fmsdata = {}
        tbf_states = {}
        for ic in initconds:
    
            data = {}
            dirname = dirlist['%d' %ic]
    
            initstate = self.get_initstate(dirname)
            spawn_info = self.get_spawn_info(dirname, ic, initstate)
    
            for i, spawn in enumerate(spawn_info):
    
                tbf_id = spawn['tbf_id']
                print('%04d-%04d' %(ic, tbf_id))
    
                tbf_data = self.get_tbf_data(dirname, ic, tbf_id, parse_extensions)
                if not tbf_data==None:
                    tbf_data['spawn_info'] = spawn
                    tbf_data['tbf_state'] = spawn['tbf_state']
                    key = '%04d-%04d' %(ic, tbf_id)
                    data[key] = tbf_data
                    tbf_states[key] = spawn['tbf_state']
                print('Finish')

            if save_to_disk_ic:
                if not os.path.isdir(datadir):
                    os.mkdir(datadir)
                with open('%s/%04d.pickle' %(datadir, ic), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            fmsdata['%04d' %ic] = data
        nstates = tbf_data['nstates']

        if save_to_disk_full:
            full_data = { 'nstates' : nstates, 'fmsdata' : fmsdata, 'tbf_states' : tbf_states }
            if not os.path.isdir(datadir):
                os.mkdir(datadir)
            with open('%s/full_simulation.pickle' %(datadir), 'wb') as handle:
                pickle.dump(full_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        return fmsdata, nstates, tbf_states
    

    def write_fmsinfo(self, dirlist, datadir, nstates=None, partitioned_ics=None):
        """
        Description:
        -------------------------------------
            - Writes fmsinfo.pickle to disk. 
            - fmsinfo contains some helpful information regarding the overall simulation to help with some analyses where loading an entire simulation dataset may be too expensive. 
            - Allows for saving a partitioning of the initial conditions if this is required for some reason. 

        Arguments: 
        -------------------------------------
            dirlist (list(str)
                - list of simulation directory paths
            datadir (str)
                - location of the pickle file to dump
            nstates (int)
                - number of adiabatic states involved in the AIMS simulation.
            partitioned_ics (dict)
                - dictionary specifying the partitioning of initial conditions
                  niche use cases

        Returns:
        -------------------------------------
            fmsinfo (dict)
            - a dictionary containing the following elements:
                - ics: a full list of the initial conditions as ints ignoring user-specified partitioning of the ICs
                - partitioned_ics: a user-specified partitioning of the ICs in dictionary form
                - nstates: an int specifying the number of adiabatic states involved in the AIMS simulation
                - datafiles: a list of strings specifying the path to the saved pickle files for each initial condition
                - dirlist: a list of strings specifying the path to each FMS90 simulation directory. 
                - tbf_states: a dictionary indexed by TBF keys ('%04d-04d' %(ic, tbf_id)) reporting the adiabatic state on which the TBF lives
        """
        fmsinfo = {}
        sim_list = glob.glob('%s/*[0-9].pickle' %datadir)
        sim_list = [x for x in sim_list if 'fmsinfo' not in x]
        sim_list.sort()
    
        if partitioned_ics==None:
            ic_list = [int(os.path.basename(x).split('.')[0]) for x in sim_list]
            ic_list.sort()
            partitioned_ics = { 'ics' : ic_list }
        else:
            ic_list = []
            for key in ics.keys():
                ic_list = ic_list + ics[key]
    
        tbf_states = {}
        for ic, datafile in zip(ic_list, sim_list):
            data = pickle.load(open(datafile, 'rb'))
            for tbf_key in data.keys():
                tbf_states[tbf_key] = data[tbf_key]['spawn_info']['tbf_state']
        nstates = data[tbf_key]['nstates']
    
        fmsinfo['ics'] = ic_list
        fmsinfo['partitioned_ics'] = partitioned_ics
        fmsinfo['nstates'] = nstates
        fmsinfo['datafiles'] = sim_list
        fmsinfo['dirlist'] = dirlist
        fmsinfo['tbf_states'] = tbf_states
        print('Saving fmsinfo.pickle')
        with open('%s/fmsinfo.pickle' %datadir, 'wb') as handle:
            pickle.dump(fmsinfo, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return fmsinfo


def example_ethylene():
    """
    Example for using the fat data management system for FMS90 simulations.
    Example data will be gladly provided upon request from jkyu.
    """
    fmsdir = '../eth_data/' # Main directory containing all FMS simulations
    ic_dict = {}
    ics = [x for x in range(0,10)] # list of initial conditions for initiating FMS simulations
    dirlist = {}
    for ic in ics:
        dirlist['%d' %ic] = fmsdir + ('%04d/' %ic) # index of paths to all individual FMS simulations
    datadir = './example_data/' # directory to which FMS90 data is stored 
    eth_fat = fat(ics, dirlist, datadir)


if __name__=='__main__':

    example_ethylene()
