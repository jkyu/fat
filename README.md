# FMS90 Analysis Tool (fat)
- For the extraction, management and analysis of data produced by FMS90 simulations.
- Authored by Jimmy K. Yu (jkyu).

Data management and analysis tool for FMS90 dynamics simulations. 
This tool handles the extraction of raw data from FMS90 simulations and facilitates the computation of observable properties. 
There are then a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
These scripts are suitable for problems involving an arbitrary number of electronic states. 

Required python packages:
- NumPy
- SciPy
- MatPlotLib
- PyMOL (optional for movie-making)

TODO:
- Continue overhaul of fat as a python package. 
- Clean up and push the movie-making functionality. 

Recent changes:
- Started conversion of the analysis scripts to a python package.
    - The data collection/management portion of fat can now be imported and used in other scripts as a library. 
- Added test.py for testing fat as a module. 

