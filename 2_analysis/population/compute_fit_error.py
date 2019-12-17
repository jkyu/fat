import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import pickle
import matplotlib.pyplot as plt
import sys

''' Computing error bars for the population FIT.
Bootstrapping at each time point gives the error bars at each time step 
and that is done in compute-populations.py already. 
We want the error on the time constant and that is also done by
bootstrapping -- 5000 samplings of the ICs are computed and then averaged.
5000 time constants are fit from this and we can look at the statistics. '''

# The exponential function to fit.
def exp_func(x, A, b, c):
    return A * np.exp(-1./b * x) + c

pop_data = pickle.load(open('./data/populations.pickle', 'rb'))
ics = pop_data['ics']
tgrid = pop_data['tgrid']
s1_pops = pop_data['state_populations']['s1']

resampled_t = []
for count in range(1000):
    resample = np.array([ s1_pops[x,:] for x in np.random.choice(np.arange(len(ics)), size=(len(ics)), replace=True) ])
    avg_resample = np.mean(resample, axis=0)
    popt, pcov = curve_fit(exp_func, tgrid, avg_resample, absolute_sigma=False)
    resampled_t.append(popt[1])

resampled_t = np.array(resampled_t)
# 
# taus = np.load('./data/taus.npz')['taus']
taus = resampled_t
# taus = taus[taus<1500] # Exclude bad exponential fits. Python can have trouble
# fitting some of the samples and returns infinity values.
print('Mean: ', np.mean(taus)) 
print('std: ', np.std(taus)) 
print('Q1: ', np.percentile(taus, 25))
print('Q2: ', np.percentile(taus, 50))
print('Q3: ', np.percentile(taus, 75))

np.savez('./data/fit_error.npz', taus=taus, error=np.std(taus))
