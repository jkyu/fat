# fyu90
Python tools to streamline the analysis of FMS90 dynamics and facilitate the computation of observable properties. 
This set of tools handles extracting raw data from the FMS90 outputs and storing them to disk (necessary in the case of large systems, e.g. proteins).
There are then also a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
These scripts are now suitable for problems involving an arbitrary number of electronic states (for applications that I care about). 
Feel free to copy and modify any of my scripts, but please give me constructive feedback or let me know if you want me to help with something. 

Required python packages:
- NumPy
- SciPy
- MDTraj
- MatPlotLib

TODO:
- Toy examples for how this thing works. 
- Update all scripts to handle arbitrary electronic states. 
    - Population scripts mostly done (single IC plotting and error fit still need to be done, but the important stuff works)
    - All of the geometric properties
    - 2D Fluorescence works for arbitrary number of states. 1D needs to updated. 
- I want to automate away specifying IC labels in analysis scripts and nstates. Should be stored as an extra "overall dynamics" kind of item in the collect data stage. 
