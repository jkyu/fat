from fat.data_managment import *
import numpy as np
import os
import sys
import pickle
"""
Movie maker module.
Authored by Jimmy K. Yu (jkyu).
"""
def sample_movie_trajectory(datafile, moviedir, tgrid, ic, tbf_ids):
    """
    Documentation.
    """
    if not os.path.isdir('./%s/' %moviedir):
        os.mkdir('./%s/' %moviedir)
    tbf_keys = ['%04d-%04d' %(ic, tbf_id) for tbf_id in tbf_ids]
    data = pickle.load(open(datafile, 'rb'))

    sampled_data = {}
    for tbf_id, tbf_key in zip(tbf_ids, tbf_keys):
        tbf = data[tbf_key]
        if not os.path.isdir('./%s/%s/' %(moviedir, tbf_key)):
            os.mkdir('./%s/%s/' %(moviedir, tbf_key))
        traj = tbf['trajectory']
        atoms = tbf['trajectory_atom_labels']
        tbf_state = tbf['tbf_state']
        populations = tbf['populations']
        tsteps = tbf['time_steps']
        tsteps_sampled = np.zeros(len(tgrid))
        populations_sampled = np.zeros(len(tgrid))
        for i, t in enumerate(tgrid):
            tstart = tsteps[0]
            tend = tsteps[-1]
            # Check that only time values of the TBF within the time grid are considered
            if t>=tstart and t<=tend:
                # Find closest geometry to the grid point
                ind = np.argmin(np.abs(tsteps - t))
                geom = traj[ind]
                tsteps_sampled[i] = tsteps[ind]
                populations_sampled[i] = populations[ind]
                with open('./%s/%s/%04d.xyz' %(moviedir, tbf_key, t), 'w') as f:
                    f, sys.stdout = sys.stdout, f
                    print('%d' %(len(atoms)))
                    print('State: %d; Time: %f; Population: %f' %(tbf_state, tsteps[ind], populations[ind]))
                    for i, atom in enumerate(atoms):
                        print('%s %13.6f %12.6f %12.6f' \
                        %(atom, geom[i,0], geom[i,1], geom[i,2]))
                    f, sys.stdout = sys.stdout, f
        sampled_tbf = { 'tsteps' : tsteps_sampled,
                        'populations' : populations_sampled,
                        'ic' : ic, 'tbf_id' : tbf_id }
        sampled_data['%s' %tbf_key] = sampled_tbf

    print('Dumping sampling information to to ./%s/movie_%04d.pickle' %(moviedir, ic))
    with open('./%s/movie_%04d.pickle' %(moviedir, ic), 'wb') as handle:
        pickle.dump(sampled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)




