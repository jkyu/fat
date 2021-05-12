# FMS Analysis Tool (FAT)

## Summary
Full multiple spawning (FMS) is a computational method for simulating ultrafast quantum dynamics processes following electronic excitations (often performed via photoexcitation). 
This tool handles the extraction of raw data from FMS simulations and automates cleaning and postprocessing of the data.
Included are a variety of routines for computing observables relevant to ultrafast quantum dynamics studies, including fluorescence and any geometric properties of interest.
Interpolation routines are also implemented to align the data to a uniformly spaced grid, a necessary step for averaging properties due to the adaptive time stepping of the FMS algorithm.
These routines suitable for problems involving an arbitrary number of electronic states and have been applied to molecular systems as large as 16,000 atoms in size with over 45,000 simulated time steps, and totalling over 800 GB in file size.

## Dependencies
Required python packages:
- NumPy
- SciPy
- MatPlotLib

Optional software:
- PyMOL (for making movies)

## Setup
To set up, clone the repository and install FAT:
```
git clone https://github.com/jkyu/fat.git
cd fat
pip install .
```

## Examples
There are a few examples for instantiating the class object and using it to compute various properties in `examples/example.py`.
The raw data used in this example will be gladly provided by the author upon request.

## Publications
FAT has contributed to several recent studies. A few highlights are listed below:
- Yu, JK; Bannwarth, C; Liang, R; Hohenstein, EG; Martinez, TJ. "Nonadiabatic Dynamics Simulation of the Wavelength-Dependent Photochemistry of Azobenzene Excited to the nπ* and ππ* Excited States." _Journal of the American Chemical Society_. **2020**. [link](https://doi.org/10.1021/jacs.0c09056)
- Yang, J; Zhu, X; Nunes, PF; Yu, JK; _et al._. "Simultaneous Observation of Nuclear and Electronic Dynamics by Ultrafast Electron Diffraction." _Science_. **2020**. [link](https://doi.org/10.1126/science.abb2235)
- Yu, JK; Liang, R; Liu, F; Martinez, TJ. "First Principles Characterization of the Elusive I Fluorescent State and the Structural Evolution of Retinal Protonated Schiff Base in Bacteriorhodopsin." _Journal of the American Chemical Society_. **2019**. [link](https://doi.org/10.1021/jacs.9b08941)
