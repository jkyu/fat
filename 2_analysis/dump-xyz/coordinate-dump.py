import pickle
import numpy as np
import mdtraj as md
import sys
import os

# Need to find spawns. Backtrack from the spawn for parent TBF and then go forward on the spawned TBF for x fs.

def dump(ics, datadir):

    for ic in ics: 
        tdiff = 400 # +/- 400 a.u. = 10 fs from the spawning point.
        data = pickle.load(open(datadir+('/%04d.pickle' %ic), 'rb'))
        fuk = []
        aux_info = []
        for tbf_key in data.keys():

            tbf = data[tbf_key]
            tbf_id = tbf['tbf_id']
            time_steps = tbf['time_steps_au']
            print('TBF: ', tbf_key)
            if tbf_id > 1:

                spawn_info = tbf['spawn_info']
                spawn_time = spawn_info['spawn_time_au']
                tbf_start = np.argmin(np.abs(time_steps - spawn_time))
                tbf_end = np.argmin(np.abs(time_steps - (spawn_time+tdiff)))

                parent_id = spawn_info['parent_id']
                parent_key = '%04d-%04d' %(ic, parent_id)
                parent_tbf = data[parent_key]
                parent_time_steps = parent_tbf['time_steps_au']
                parent_start = np.argmin(np.abs(parent_time_steps - (spawn_time-tdiff)))
                parent_end = np.argmin(np.abs(parent_time_steps - spawn_time))
                print('Parent: ', parent_key)
                print('Start time: ', parent_time_steps[parent_start])
                print('Spawn time: ', parent_time_steps[parent_end])
                print('Spawn time: ', time_steps[tbf_start])
                print('End time: ', time_steps[tbf_end])
                
                parent_traj = parent_tbf['trajectory'][parent_start:parent_end]
                print(parent_traj)
                tbf_traj = tbf['trajectory'][tbf_start:tbf_end]
                print(tbf_traj)
                stitched = md.join([parent_traj, tbf_traj])
                print(stitched)
                fuk.append(stitched)

                aux_parent = []
                parent_state_label = 's%d' %parent_tbf['state_id']
                for i in range(parent_start, parent_end):
                    tstep = parent_time_steps[i]
                    amp = parent_tbf['populations'][i]
                    transition = '%02d-%02d -> %02d-%02d' %(ic, parent_id, ic, tbf_id)
                    aux_parent.append([tstep, amp, parent_state_label, transition])
                aux_tbf = []
                tbf_state_label = 's%d' %tbf['state_id']
                for i in range(tbf_start, tbf_end):
                    tstep = time_steps[i]
                    amp = tbf['populations'][i]
                    transition = '%02d-%02d -> %02d-%02d' %(ic, parent_id, ic, tbf_id)
                    aux_tbf.append([tstep, amp, tbf_state_label, transition])
                aux_info = aux_info + (aux_parent + aux_tbf)

        final = md.join(fuk)
        atoms = [x.name for x in final.topology.atoms]
        natom = final.n_atoms

        # Want to print out 
        # Time step
        # Nonadiabatic state label
        # Coefficient
        # Energy

        f2 = open('%d.xyz' %ic, 'w')
        for i, frame in enumerate(final):
            f2.write('%d\n' %natom)
            f2.write('Time Step: %.2f    Amplitude: %.4f    State: %s    Transition: %s\n' %(aux_info[i][0], aux_info[i][1], aux_info[i][2], aux_info[i][3]))
            for j in range(natom):
                f2.write('%5s  %12f  %12f  %12f\n' %(atoms[j], *frame.xyz[0][j]*10.))
        f2.close()


datadir = '../../1-collect-data/data/'
fmsinfo = pickle.load(open(datadir+'/fmsinfo.pickle', 'rb'))
ics = fmsinfo['ics']
dump(ics, datadir)

