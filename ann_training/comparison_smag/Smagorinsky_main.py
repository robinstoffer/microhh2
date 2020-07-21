#Main script to test and run calculate_eddy_diffusivity.py and calculate_turbulent_fluxes.py, which calculate the sub-grid scale fluxes according to the Smagorinsky-Lilly model.
#Author: Robin Stoffer (February 2019)

#Load modules
import numpy as np
import netCDF4 as nc
from calculate_eddy_diffusivity import calculate_eddy_diffusivity
from calculate_turbulent_fluxes import calculate_turbulent_fluxes

#Define input/output filepaths
input_filepath  = '../../cases/moser600/training_data/training_data.nc'
output_filepath_evisc = 'eddy_diffusivity.nc'
output_filepath_smag  = 'smagorinsky_fluxes.nc'

#Calculate eddy diffusivities
calculate_eddy_diffusivity(input_filepath = input_filepath, output_filepath = output_filepath_evisc)

#Calculate turbulent fluxes
calculate_turbulent_fluxes(flowfields_filepath = input_filepath, eddy_diffusivity_filepath = output_filepath_evisc, output_filepath = output_filepath_smag, zero_w_topbottom = True)

print('Finished')

