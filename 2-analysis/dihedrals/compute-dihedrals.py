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

def compute_dihedral(frame, dihe_inds):

    atom1 = frame[dihe_inds[0]]
    atom2 = frame[dihe_inds[1]]
    atom3 = frame[dihe_inds[2]]
    atom4 = frame[dihe_inds[3]]

    v12 = atom2 - atom1
    v23 = atom3 - atom2
    v34 = atom4 - atom3
    v12 = v12 / np.linalg.norm(v12)
    v23 = v23 / np.linalg.norm(v23)
    v34 = v34 / np.linalg.norm(v34)

    n1 = np.cross(v12, v23)
    n2 = np.cross(v23, v34) 
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, v23), n2)
    dihedral_angle = np.arctan2(y,x) * 180./np.pi

    return dihedral_angle

def process_trajectories(ics, tgrid, datadir, nstates, dihedral_index, start_config='cis', do_state_specific=False):
    '''
    Load the fat data file and collect the spawn information.
    Gather the value of the dihedral angles from the trajectories.
    The dihedral angles are specified as a dictionary and taken into this
    function as dihedral_index.
    '''
    print('Loading trajectories for IC TBFs and computing dihedral angles.')

    dihedral_names = [x for x in dihedral_index.keys()]
    dihedral_list = [ dihedral_index[x] for x in dihedral_names ] 

    raw_angles = {}
    raw_tsteps = {}
    raw_pops = {}
    state_ids = {}

    ''' Compute the dihedral angles. The data dictionary is indexed as IC -> TBF Key -> Dihedral Name -> Frame Number '''
    for ic in ics:
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            state_id = tbf['state_id']
            state_ids[tbf_key] = state_id
            print('%s, state s%d' %(tbf_key, state_id))

            time_steps = tbf['time_steps']
            trajectory = tbf['trajectory']
            populations = tbf['populations']

            dihes_dict = {}
            for dihe_name, dihe_inds in zip(dihedral_names, dihedral_list):
                dihes_traj = []
                ''' Compute the dihedral angle for each frame. '''
                for i in range(len(trajectory)):
                    frame = trajectory.xyz[i] * 10.
                    dihe_angle = compute_dihedral(frame, dihe_inds)
                    ''' Handle the wrapping over/under +/-180 degrees. '''
                    if i>0:
                        if (dihe_angle -  dihes_traj[i-1]) > 300:
                            dihe_angle = dihe_angle - 360
                        elif (dihe_angle - dihes_traj[i-1]) < -300:
                            dihe_angle = dihe_angle + 360
                    dihes_traj.append(dihe_angle)
                ''' If the simulation tracks a trans->cis isomerization, we want our angle range to be [0,2pi].
                For cis->trans, we want the angle range to be [-pi,pi]. We want to be centered at our start 
                configuration in order to track directionality and handle wrapping appropriately. '''
                if start_config=='trans':
                    dihes_traj = [x + 360 if x < 0 else x for x in dihes_traj]
                else: 
                    dihes_dict[dihe_name] = np.array(dihes_traj)

            raw_angles['%s' %tbf_key] = dihes_dict
            raw_tsteps['%s' %tbf_key] = time_steps
            raw_pops['%s' %tbf_key] = populations

    '''
    Place the dihedral angles on a grid so that we don't have issues with averaging
    due to adaptive time steps. The data is stored in a dictionary indexed by the
    TBF name (e.g. 02-03) and then by the specific dihedral angle computed
    (e.g. C12-C13=C14-C15).
    '''
    interp_dihedrals = {}
    interp_zeroed = {}
    interp_populations = {}

    print('Aggregating dihedral angles and populations in time by interpolating.')
    for tbf_key in raw_angles.keys():
        dihes_dict = {}
        zeroed_dict = {}
        for dihe_idx, dihe_name in enumerate(dihedral_names):
            tsteps = raw_tsteps[tbf_key]
            dihes = raw_angles[tbf_key][dihe_name]      # angle values of named dihedrals for each IC
            interp_dihes = interpolate(tgrid, tsteps, dihes, do_state_specific=do_state_specific)
            dihes_dict[dihe_name] = interp_dihes

            dihes_zeroed = interpolate(tgrid, np.array(tsteps) - tsteps[0], dihes, do_state_specific=do_state_specific)
            zeroed_dict[dihe_name] = dihes_zeroed

        interp_populations[tbf_key] = interpolate(tgrid, tsteps, raw_pops[tbf_key])
        interp_zeroed[tbf_key] = zeroed_dict
        interp_dihedrals[tbf_key] = dihes_dict

    # Cache data
    data2 = {}
    data2['ics'] = ics
    data2['nstates'] = nstates
    data2['dihedral_names'] = dihedral_names
    data2['dihedrals'] = interp_dihedrals
    data2['populations'] = interp_populations
    data2['state_ids'] = state_ids
    data2['tbf_keys'] = [x for x in state_ids.keys()]
    data2['tgrid'] = tgrid

    print('Dumping interpolated amplitudes to dihedrals.pickle')
    if not os.path.isdir('./data/'):
        os.mkdir('./data/')
    with open('./data/dihedrals.pickle', 'wb') as handle:
        pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
The following dihedral angles are enumerated and indexed according
to the geometry file so that we can compute the dihedral angles.
Pass this dictionary into compute_dihedrals()
'''
print('Indexing dihedral angles.')
dihedral_index = {}
dihedral_index['HCCH'] = [2, 5, 4, 0]

tgrid = np.arange(0, 250, 5)
datadir = '../../1-collect-data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
nstates = fmsinfo['nstates']
start_config = 'cis'
process_trajectories(ics, tgrid, datadir, nstates, dihedral_index, start_config, do_state_specific=False)
