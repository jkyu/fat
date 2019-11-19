import numpy as np
import os
import mdtraj as md
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import pickle
import sys

rcParams.update({'figure.autolayout': True})

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
ecolors = ['lightsteelblue', 'moccasin', 'mediumseagreen', 'lightcoral', 'thistle',
        'peru', 'lightpink', 'lightgray', 'khaki', 'azure']

data = pickle.load(open('./data/energies.pickle', 'rb'))
tgrid = data['tgrid']
all_energies = data['energies']
tbf_key = '0003-0001'

fig = plt.figure(figsize=(6,4.5))
labelsize = 13
ticksize = 11

tbf_energies = all_energies[tbf_key]
for i, state_key in enumerate(tbf_energies.keys()):
    label = '%s energy' %state_key
    plt.plot(tgrid, tbf_energies[state_key], linewidth=0.8, color=colors[i], label=label.title())

plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.ylabel('Energy [eV]', fontsize=labelsize)
plt.xlabel('Time [fs]', fontsize=labelsize)
plt.legend(loc='best', frameon=False, fancybox=False, fontsize=ticksize, numpoints=1)

if not os.path.isdir('./figures/'):
    os.mkdir('./figures')
plt.savefig('./figures/duk.png')
