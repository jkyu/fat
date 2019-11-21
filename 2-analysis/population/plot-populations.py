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

do_fit = 1

pop_data = pickle.load(open('./data/populations.pickle', 'rb'))
ics = pop_data['ics']
tgrid = pop_data['tgrid']
populations = pop_data['populations']
errors = pop_data['errors']
keys = [x for x in populations.keys()]

''' Plotting functions '''

matplotlib.use('Agg')
rcParams.update({'figure.autolayout': True})

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
ecolors = ['lightsteelblue', 'moccasin', 'mediumseagreen', 'lightcoral', 'thistle',
        'peru', 'lightpink', 'lightgray', 'khaki', 'azure']

fig = plt.figure(figsize=(6,5))
labelsize = 16
ticksize = 14
plt.rc('xtick',labelsize=ticksize)
plt.rc('ytick',labelsize=ticksize)

for i, key in enumerate(keys):
    label = '%s population' %key
    plt.errorbar(tgrid, populations[key], yerr=errors[key], color=colors[i], linewidth=3.0, elinewidth=1, ecolor=ecolors[i], capsize=0.1, label=label.title())
if do_fit:
    popt, pcov = curve_fit(exp_func, tgrid, populations['s1'], absolute_sigma=False)
    print('Exponential fit. Tau = ', popt[1])
    fit = exp_func(tgrid, *popt)
    if os.path.isfile('./data/fit_error.npz'):
        fit_error = np.load('./data/fit_error.npz')['error']
        fit_label = 'Exponential Fit ($\\tau=%d \pm %d$ fs)' %(popt[1], fit_error)
    else:
        fit_label = 'Exponential Fit ($\\tau=%d$ fs)' %popt[1]
    plt.plot(tgrid, fit, color='black', linewidth=2.0, label=fit_label, linestyle='--')

plt.ylabel('Fractional Population', fontsize=labelsize)
plt.xlabel('Time [fs]', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

# handles, labels = plt.gca().get_legend_handles_labels()
# order = [1, 2, 0]
# leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='best', fontsize=ticksize, fancybox=False, frameon=False, numpoints=1)
# leg.get_frame().set_edgecolor('black')

plt.ylim([0, 1.1])
plt.xlim([tgrid[0], tgrid[-1]])

if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('./figures/population.pdf')
plt.savefig('./figures/population.png')
