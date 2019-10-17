import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import curve_fit

def exp_func(x, A, b, c):
    return A * np.exp(-1./b * x) + c

''' The data is computed in compute-populations.py and has been dumped 
to file as ./data/populations.pickle 
Read the time grid and the population information to plot them. '''

pop_data = pickle.load(open('./data/populations.pickle', 'rb'))
ics = [ x for x in range(1,33) if x not in [6,17] ]

ex_pop = pop_data['ex_populations']
gs_pop = pop_data['gs_populations']
ex_err = pop_data['ex_error']
gs_err = pop_data['gs_error']
tgrid = pop_data['tgrid']

popt, pcov = curve_fit(exp_func, tgrid, ex_pop, absolute_sigma=False)
print('Exponential fit. Tau = ', popt[1])
fit = exp_func(tgrid, *popt)

''' Plotting functions '''

matplotlib.use('Agg')
rcParams.update({'figure.autolayout': True})

fig = plt.figure(figsize=(6,5))
labelsize = 16
ticksize = 14
plt.rc('xtick',labelsize=ticksize)
plt.rc('ytick',labelsize=ticksize)

plt.errorbar(tgrid, gs_pop, yerr=gs_err, color='steelblue', linewidth=3.0, elinewidth=1, ecolor='lightblue', capsize=0.1, label='S0 Population')
plt.errorbar(tgrid, ex_pop, yerr=ex_err, color='firebrick', linewidth=3.0, elinewidth=1, ecolor='lightcoral', capsize=0.1, label='S1 Population')
plt.plot(tgrid, fit, color='black', linewidth=2.0, label='Exponential Fit ($\\tau$=%d fs)' %popt[1], linestyle='--')

plt.ylabel('Fractional Population', fontsize=labelsize)
plt.xlabel('Time [fs]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]
leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best', fontsize=ticksize, fancybox=False, frameon=False, numpoints=1)
leg.get_frame().set_edgecolor('black')

plt.ylim([0, 1.1])
plt.xlim([0, 1450])

if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('./figures/population.pdf')
plt.savefig('./figures/population.png')
