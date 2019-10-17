import numpy as np
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

tgrid = np.arange(0, 1500, 1)
ics = [x for x in range(1, 33) if x not in [6,17] ]
pop_data = pickle.load(open('./data/populations.pickle', 'rb'))

interp_populations = pop_data['all_populations']
ex_pop = pop_data['ex_populations']
ex_keys = [x for x in interp_populations.keys() if x.split('-')[-1]=='01']

all_ex_pops = np.zeros((len(ex_keys), len(tgrid)))
for i, tbf_key in enumerate(ex_keys):
    all_ex_pops[i,:] = interp_populations[tbf_key]

labelsize = 16
ticksize = 14
plt.rc('xtick',labelsize=ticksize)
plt.rc('ytick',labelsize=ticksize)
fig = plt.figure(figsize=(6,5))

resampled_t = []
for idx in range(len(ics)):
    single = all_ex_pops[idx,:]
    if idx==0:
        plt.plot(tgrid, single, linestyle='--', linewidth=0.8, label='Single IC')
    else:
        plt.plot(tgrid, single, linewidth=0.8, linestyle='--')
plt.plot(tgrid, ex_pop, color='firebrick', linewidth=2.0, label='Full wavepacket')
plt.axis([0, 1500, 0, 1.03])
plt.legend(loc='best', fontsize=ticksize)
plt.xlabel('Time [fs]', fontsize=labelsize)
plt.ylabel('Excited State Population', fontsize=labelsize)

if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('single/singles.png')

