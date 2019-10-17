# fyu90
Python tools to streamline the analysis of FMS90 dynamics and facilitate the computation of observable properties. 
This set of tools handles extracting raw data from the FMS90 outputs and storing them to disk (necessary in the case of large systems, e.g. proteins).
There are then also a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
I hard coded that because it's going to take some time to automate this for an arbitrary number of states and because I'm stupid.
Feel free to copy and modify any of my scripts, but please give me constructive feedback or let me know if you want me to help with something. 

Disclaimer: These are all written for a two-state problem, although modification to include more states is easy (I've already done this once).

Required python packages:
- NumPy
- SciPy
- MDTraj
- MatPlotLib

TODO:
- Toy examples for how this thing works. 
