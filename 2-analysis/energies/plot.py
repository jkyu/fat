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

data = pickle.load(open('./data/energies.pickle', 'rb'))
tgrid = data['tgrid']
s0_energies = data['s0_energies']
s1_energies = data['s1_energies']

gs_keys = data['gs_keys'][0:2]
ex_keys = data['ex_keys'][0:1]

fig = plt.figure(figsize=(6,4.5))
labelsize = 13
ticksize = 11

plt.subplot(2,1,1)
plt.plot(tgrid, s0_energies[ex_keys[0]], linewidth=0.8, linestyle='-', color='orchid', label='S0')
plt.plot(tgrid, s1_energies[ex_keys[0]], linewidth=0.8, linestyle='-', color='slateblue', label='S1')
for i in range(1, len(ex_keys)):
    plt.plot(tgrid, s0_energies[ex_keys[i]], linewidth=0.8, linestyle='-', color='orchid')
    plt.plot(tgrid, s1_energies[ex_keys[i]], linewidth=0.8, linestyle='-', color='slateblue')
for i in range(0, len(gs_keys)):
    plt.plot(tgrid, s0_energies[gs_keys[i]], linewidth=0.8, linestyle='--', color='orchid')
    plt.plot(tgrid, s1_energies[gs_keys[i]], linewidth=0.8, linestyle='--', color='slateblue')
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.ylabel('Energy (eV)', fontsize=labelsize)
plt.xlabel('Time (fs)', fontsize=labelsize)
plt.legend(loc='best', frameon=False, fontsize=ticksize)

plt.subplot(2,1,2)
plt.plot(tgrid, s1_energies[ex_keys[0]]-s0_energies[ex_keys[0]], linewidth=0.8, linestyle='-', color='black', label='S1-S0')
for i in range(1, len(ex_keys)):
    plt.plot(tgrid, s1_energies[ex_keys[i]]-s0_energies[ex_keys[i]], linewidth=0.8, linestyle='-', color='black')
for i in range(0, len(gs_keys)):
    plt.plot(tgrid, s1_energies[gs_keys[i]]-s0_energies[gs_keys[i]], linewidth=0.8, linestyle='--', color='black')
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.ylabel('Energy (eV)', fontsize=labelsize)
plt.xlabel('Time (fs)', fontsize=labelsize)
plt.legend(loc='best', frameon=False, fontsize=ticksize)

if not os.path.isdir('./figures/'):
    os.mkdir('./figures')
plt.savefig('./figures/duk.png')
