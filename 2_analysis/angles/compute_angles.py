import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

'''
Reads in raw data from the FMS pickle files, computes relevant dihedral angles,
and then interpolates them on a grid to resolve the issue of adaptive time steps.
Also computes the average dihedrals if the flag is turned on and computes
the sampling error of the dihedral angles by bootstrapping.
'''

def interpolate(grid, tsteps, data, do_state_specific=False):

    interp_data = np.zeros((len(grid)))
    spacing = np.max(grid) / float(len(grid))

    for i in range(len(grid)):
        if i==0:
            tlow = 0
        else:
            tlow = grid[i] - spacing/2
        if i==len(grid) - 1:
            thigh = grid[-1]
        else:
            thigh = grid[i] + spacing/2
        inds = [x for x, y in enumerate(tsteps) if y >= tlow and y <= thigh]
        dat = [data[ind] for ind in inds]
        tdiffs = [np.abs(grid[i] - y) for y in tsteps if y >= tlow and y <= thigh] # computes the distance of the raw data time point from the grid point
        if len(dat) > 0:
            tdiffs_frac = tdiffs / np.sum(tdiffs) # normalizes the distance from the grid point
            interp_data[i] = np.average(dat, weights=tdiffs_frac) # weighted average of the data points by their distance from the grid point
        else: 
            if do_state_specific:
                interp_data[i] = interp_data[i-1]
            else:
                interp_data[i] = np.nan
    
    for i in range(len(interp_data)):
        if interp_data[i]==0:
            interp_data[i] = np.nan

    return interp_data

def compute_angle(frame, angle_inds):

    atom1 = frame[angle_inds[0]]
    atom2 = frame[angle_inds[1]]
    atom3 = frame[angle_inds[2]]

    v21 = atom2 - atom1
    v23 = atom2 - atom3
    norm21 = np.linalg.norm(v21)
    norm23 = np.linalg.norm(v23)

    angle = np.arccos( np.dot(v21, v23)/(norm21*norm23) ) * 180 / np.pi

    return angle

def process_trajectories(ics, datafiles, tgrid, nstates, angle_index, outfile_name='angles'):
    '''
    Load the fat data file and collect the spawn information.
    Gather the value of the angles from the trajectories.
    The angles are specified as a dictionary and taken into this
    function as angle_index.
    '''
    print('Loading trajectories for IC TBFs and computing angles.')

    angle_names = [x for x in angle_index.keys()]
    angle_list = [ angle_index[x] for x in angle_names ] 

    raw_angles = {}
    raw_tsteps = {}
    raw_pops = {}
    tbf_states = {}

    ''' Compute the angles. The data dictionary is indexed as IC -> TBF Key -> Angle Name -> Frame Number '''
    for datafile in datafiles:
        data = pickle.load(open(datafile, 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_state = tbf['spawn_info']['tbf_state']
            tbf_states[tbf_key] = tbf_state
            print('%s, state s%d' %(tbf_key, tbf_state))

            time_steps = tbf['time_steps']
            trajectory = tbf['trajectory']
            populations = tbf['populations']

            angles_dict = {}
            for angle_name, angle_inds in zip(angle_names, angle_list):
                angle_traj = []
                ''' Compute the angle for each frame. '''
                for i in range(len(trajectory)):
                    frame = trajectory[i]
                    angle = compute_angle(frame, angle_inds)
                    # ''' If the simulation tracks a trans->cis isomerization, we want our angle range to be [0,2pi].
                    # For cis->trans, we want the angle range to be [-pi,pi]. We want to be centered at our start 
                    # configuration in order to track directionality and handle wrapping appropriately. '''
                    # if start_config=='trans' and dihe_angle < 0:
                    #     dihe_angle = dihe_angle + 360
                    # ''' Handle the wrapping over/under +/-180 degrees. '''
                    # if i>0:
                    #     if (dihe_angle - dihes_traj[i-1]) > 300:
                    #         dihe_angle = dihe_angle - 360
                    #     elif (dihe_angle - dihes_traj[i-1]) < -300:
                    #         dihe_angle = dihe_angle + 360
                    angle_traj.append(angle)
                angles_dict[angle_name] = np.array(angle_traj)

            raw_angles['%s' %tbf_key] = angles_dict
            raw_tsteps['%s' %tbf_key] = time_steps
            raw_pops['%s' %tbf_key] = populations

    '''
    Place the angles on a grid so that we don't have issues with averaging
    due to adaptive time steps. The data is stored in a dictionary indexed by the
    TBF name (e.g. 02-03) and then by the specific angle computed
    (e.g. C12-C13=C14-C15).
    '''

    print('Aggregating angles and populations in time by interpolating.')
    interp_angles = {}
    interp_populations = {}
    for tbf_key in raw_angles.keys():
        angles_dict = {}
        for angle_idx, angle_name in enumerate(angle_names):
            tsteps = raw_tsteps[tbf_key]
            angles = raw_angles[tbf_key][angle_name]      # angle values of named dihedrals for each IC
            interp_dihes = interpolate(tgrid, tsteps, angles, do_state_specific=False)
            angles_dict[angle_name] = interp_angles

        interp_populations[tbf_key] = interpolate(tgrid, tsteps, raw_pops[tbf_key], do_state_specific=True)
        interp_angles[tbf_key] = angles_dict

    print('Aggregating angles and populations in time by interpolating for state specific averaging.')
    interp_zeroed = {}
    interp_angles2 = {}
    for tbf_key in raw_angles.keys():
        angles_dict2 = {}
        zeroed_dict = {}
        for angle_idx, angle_name in enumerate(angle_names):
            tsteps = raw_tsteps[tbf_key]
            angles = raw_angles[tbf_key][angle_name]      # angle values of named angledrals for each IC
            interp_angles = interpolate(tgrid, tsteps, angles, do_state_specific=True)
            # if start_config=='trans':
            #     interp_angles = np.abs(180 - interp_angles) + 180
            # else:
            #     interp_angles = np.abs(0 - interp_angles)
            angles_dict2[angle_name] = interp_angles

            angles_zeroed = interpolate(tgrid, np.array(tsteps) - tsteps[0], angles, do_state_specific=True)
            zeroed_dict[angle_name] = angles_zeroed

        interp_zeroed[tbf_key] = zeroed_dict
        interp_angles2[tbf_key] = angles_dict2

    # Cache data
    data2 = {}
    data2['ics'] = ics
    data2['nstates'] = nstates
    data2['angle_names'] = angle_names
    data2['angles'] = interp_angles
    data2['angles_state_specific'] = interp_angles2
    data2['angles_time_zeroed'] = interp_zeroed
    data2['populations'] = interp_populations
    data2['tbf_states'] = tbf_states
    data2['tbf_keys'] = [x for x in tbf_states.keys()]
    data2['tgrid'] = tgrid

    print('Dumping interpolated amplitudes to angles.pickle')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/%s.pickle' %outfile_name, 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
The following angles are enumerated and indexed according
to the geometry file so that we can compute the angles.
Pass this dictionary into compute_angles()
'''
print('Indexing angles.')
angle_index = {}
angle_index['test'] = [3, 0, 1]

tgrid = np.arange(0, 750, 5)
datadir = '../../1_collect_data/'
fmsinfo = pickle.load(open(datadir+'/data/fmsinfo.pickle', 'rb'))
picklefiles = fmsinfo['datafiles']
datafiles = [ datadir+x for x in picklefiles ] 
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
outfile_name = 'angles'
process_trajectories(ics, datafiles, tgrid, nstates, angle_index, outfile_name=outfile_name)
