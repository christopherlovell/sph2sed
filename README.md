
# sph2sed

Package for generating galaxy spectral energy distributions (SEDs) from smoothed particle hydrodynamics (SPH) simulations.

The package has two main components. 
**grids** contains methods for generating SED grids from SPS models. 
**sph2sed** contains the python scripts for generating SEDs from SPH particle inputs.

### SPS Grids

Right now, only FSPS is supported. FSPS and python-FSPS are required to generate new grids.

#### Cloudy 

The data pipeline provides a means of processing a given Stellar Population Synthesis (SPS) model through the Cloudy spectral synthesis code.

1. `/grids/Input_SPS/Create_cloudy_atmosphere.py` generates a cloudy atmosphere grid for each model and places it in the specified cloudy directory. It also creates input files for all parameter space in /Cloudy_input, and associated batchscripts for running them all.
2. `/grids/Cloudy_input` contains a folder for each SPS model containing a Cloudy input file for each parameter configuration. They can be run individually, or all at once using the batch script.
3. `/grids/Cloudy_output` contains the continuum and line information from each Cloudy run for each SPS model.

### sph2sed

`demo.ipynb` provides an example usage for the Eagle simulation.

## To Do

- provide example data
