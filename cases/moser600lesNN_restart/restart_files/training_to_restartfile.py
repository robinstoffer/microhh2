#Script to generate binary restart files for MicroHH from training data nc-file
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

#Select time step to create the restart file from, the specific time step selected is not important
t = 28 #29th timestep, as this one is also evaluated during the offline a priori test of the NN

#Specify where training file is stored
training_filepath = "../../moser600/training_data/training_data.nc"

#Fetch training data
a = nc.Dataset(training_filepath, 'r')

#Read ghost cells and indices from the training file
igc        = int(a['igc'][:])
jgc        = int(a['jgc'][:])
kgc_center = int(a['kgc_center'][:])
kgc_edge   = int(a['kgc_edge'][:])
iend       = int(a['iend'][:])
jend       = int(a['jend'][:])
kend       = int(a['kend'][:])
khend      = int(a['khend'][:])

#Select flow fields for selected time step and remove all the ghost cells.
uc_singlefield = np.array(a['uc'][t,kgc_center:kend,jgc:jend,igc:iend])
vc_singlefield = np.array(a['vc'][t,kgc_center:kend,jgc:jend,igc:iend])
wc_singlefield = np.array(a['wc'][t,kgc_edge:khend,jgc:jend,igc:iend])

#Write flow fields to binary files
uc_singlefield.tofile("u.restart")
vc_singlefield.tofile("v.restart")
wc_singlefield.tofile("w.restart")

#Close nc-file
a.close()
