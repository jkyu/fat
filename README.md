# FMS90 Analysis Tool (FAT)

- For the extraction, management and analysis of data produced by FMS90 simulations.
- Authored by Jimmy K. Yu (jkyu@stanford.edu).

Data management and analysis tool for FMS90 dynamics simulations. 
This tool handles the extraction of raw data from FMS90 simulations and facilitates the computation of observable properties. 
There are then a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
These scripts are suitable for problems involving an arbitrary number of electronic states. 

Required python packages:
- NumPy
- SciPy
- MatPlotLib
- PyMOL (optional for movie-making)

What can fat do right now?
- Extracts FMS90 data into a framework that is straightforward to work with for the analysis of the dynamics and the computation of observables. It's easy to write your own scripts for functionality that is not already implemented. 
- Population dynamics
- Fluorescence spectra
- Electron diffraction spectra
- Quantify reaction yields
- Geometric analysis (using manual parameters or via automated nonlinear dimensionality reduction techniques)
- Generate molecular movies

TODO:
- Continue overhaul of fat as a python package. 
- Clean up and push the movie-making functionality. 
- 1\_collect\_data and 2\_analysis will be deprecated upon completion of packaging. 

Recent changes:
- Started conversion of the analysis scripts to a python package.
    - The data collection/management portion of fat can now be imported and used in other scripts as a library. 
- Added example.py for demonstrating fat as a module. 

Known issues:
- Need a better way to handle fmsinfo than by globbing. 
