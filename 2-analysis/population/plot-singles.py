import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

'''
Plot the population dynamics of each FMS simulation separately.
This allows some visualization of the population transfer
efficiency and how each FMS simulation contributes to the
overall excited state decay time.
'''

# The exponential function to fit.
def exp_func(x, A, b, c):
    return A * np.exp(-1./b * x) + c

pop_data = pickle.load(open('./data/populations.pickle', 'rb'))
tgrid = pop_data['tgrid']
ics = pop_data['ics']

target_state = 's1'
state_populations = pop_data['state_populations'][target_state]
avg_pop = pop_data['populations'][target_state]

rcParams.update({'figure.autolayout': True})
labelsize = 16
ticksize = 14
plt.rc('xtick',labelsize=ticksize)
plt.rc('ytick',labelsize=ticksize)
fig = plt.figure(figsize=(6,5))

resampled_t = []
for idx in range(len(ics)):
    single = state_populations[idx,:]
    if idx==0:
        plt.plot(tgrid, single, linestyle='--', linewidth=0.8, label='Single IC')
    else:
        plt.plot(tgrid, single, linewidth=0.8, linestyle='--')
plt.plot(tgrid, avg_pop, color='firebrick', linewidth=2.0, label='Full wavepacket')
plt.axis([tgrid[0], tgrid[-1], 0, 1.03])
plt.legend(loc='best', fontsize=ticksize)
plt.xlabel('Time [fs]', fontsize=labelsize)
plt.ylabel('Excited State Population', fontsize=labelsize)

if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('figures/singles.pdf', dpi=300)

