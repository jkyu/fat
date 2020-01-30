import numpy as np
import os
import math
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib
matplotlib.use('Agg')

''' Computes error in the fluorescence due to sampling. This
is used to mark error bars on 1D fluorescence lineouts. '''

def compute_fluorescence(tbf_data, keys):

    fl_grid = np.zeros_like(tbf_data[keys[0]])
    for key in keys:
        ic_grid = tbf_data[key]
        fl_grid = fl_grid + ic_grid

    return fl_grid

def get_grid(tbf_data, ic):

    ic_grid = tbf_data[ic]
    ic_grid = ic_grid / np.max(ic_grid)
    ic_grid[ic_grid==0] = np.nan
    return ic_grid

data = pickle.load(open('./data/fluorescence.pickle', 'rb'))
ics = data['ics']
tbf_data = data['tbf_fluorescence']
keys = [x for x in tbf_data.keys()]

fl_grid = compute_fluorescence(tbf_data, keys)

print('Resampling ICs and recomputing the fluorescence.')
sampled_ics = [ [x for x in np.random.choice(ics, size=len(ics), replace=True)] for _ in range(1000) ]
sampled_keys = [ [x for x in keys if int(x.split('-')[0]) in ic_subset] for ic_subset in sampled_ics ]
sampled_grids = np.array([ compute_fluorescence(tbf_data, key_subset) for key_subset in sampled_keys ])

grid_error = np.std(sampled_grids, axis=0)

data2 = {}
data2['fluorescence_error'] = grid_error
print('Dumping the bootstrapping error for the fluorescence signal to ./data/fl_error.pickle.')
with open('./data/fl_error.pickle', 'wb') as handle:
    pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)
