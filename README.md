# fyu90
Python tools to streamline the analysis of FMS90 dynamics and facilitate the computation of observable properties. 
This set of tools handles extracting raw data from the FMS90 outputs and storing them to disk (necessary in the case of large systems, e.g. proteins).
There are then also a variety of scripts for computing various properties, such as fluorescence and any geometric properties, and interpolating to a grid (necessary for averaging any properties due to the adaptive time stepping of FMS).
These scripts are suitable for problems involving an arbitrary number of electronic states (for applications that I care about). 
Feel free to copy and modify any of my scripts, but please give me constructive feedback or let me know if you want me to help with something. 

Required python packages:
- NumPy
- SciPy
- MatPlotLib

TODO:
- In analysis scripts, take full paths to the FMS pickle files instead of taking the data directory as an argument. A list of the full paths should be given instead of just one string (datadir). Right now, the full paths to the data are put together as part of the main functions of the analysis scripts. Want more flexibility than this. 
- Convert this into a python package instead of a set of scripts with separate main functions that require input. 
- Example with ethylene. For now, just ask me for example data. 

Recent changes:
- Removed the dependency on topology files.
- Fluorescence bug fixes. 
- Some cool decay pathway analyses. 
