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
normalizer = np.max(fl_grid)

print('Resampling ICs and recomputing the fluorescence.')
sampled_keys = [ [x for x in np.random.choice(keys, size=len(ics), replace=False)] for _ in range(1000) ]
sampled_grids = np.array([ (compute_fluorescence(tbf_data, ics) / normalizer) for ics in sampled_keys ])

grid_error = np.std(sampled_grids, axis=0)

data2 = {}
data2['fluorescence_error'] = grid_error
print('Dumping the bootstrapping error for the fluorescence signal to ./data/fl-error.pickle.')
with open('./data/fl-error.pickle', 'wb') as handle:
    pickle.dump(data2, handle, protocol=pickle.HIGHEST_PROTOCOL)
