# fyu90
Python tools to streamline the analysis of FMS90 dynamics and facilitate the computation of observable properties. 
This set of tools handles extracting raw data from the FMS90 outputs and storing them to disk (necessary in the case of large systems, e.g. proteins).
There are then also a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
These scripts are now suitable for problems involving an arbitrary number of electronic states (for applications that I care about). 
Feel free to copy and modify any of my scripts, but please give me constructive feedback or let me know if you want me to help with something. 

Required python packages:
- NumPy
- SciPy
- MDTraj (the dependence on MDTraj is very light and just saves me from writing a coordinate parser for now. Will handle later to reduce unnecessary dependencies.)
- MatPlotLib

TODO:
- Toy examples for how this thing works. 
- Some descriptions of the main data structures I'm using here. How to access the pickled data in general.
- Remove the requirement of a topology file (read: write my own xyz parser). 
- Update all scripts to handle arbitrary electronic states. 
    - Population scripts done.
    - Interpolation for energies now works. 
    - Bond length and dihedral angle tracking work now. [Dec 2, 2019]
    - 2D Fluorescence works for arbitrary number of states. I think 1D also works? Haven't had a reason to test yet. [Dec 2, 2019]
