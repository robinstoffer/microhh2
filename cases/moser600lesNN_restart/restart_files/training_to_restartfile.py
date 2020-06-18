#Script to generate binary restart files for MicroHH from training data nc-file
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

#Select time step to create the restart file from, the specific time step selected is not important
t = 27 #28th timestep, as this one is also evaluated during the offline inference of the NN

#Specify where training file is stored
training_filepath = "/media/sf_Shared_folder/training_data_coarse3d.nc"

#Fetch training data
a = nc.Dataset(training_filepath, 'r')

#Get friction velocity
utau_ref = np.array(a['utau_ref'][:])

#Read ghost cells and indices from the training file
igc        = int(a['igc'][:])
jgc        = int(a['jgc'][:])
kgc_center = int(a['kgc_center'][:])
kgc_edge   = int(a['kgc_edge'][:])
iend       = int(a['iend'][:])
jend       = int(a['jend'][:])
kend       = int(a['kend'][:])
khend       = int(a['khend'][:])

#Select flow fields for selected time step, remove all the ghost cells, denormalize, and multiply with constant factor to test extrapolation capability NN
const_factor = 1.0
uc_singlefield = np.array(a['uc'][t,kgc_center:kend,jgc:jend,igc:iend]) * utau_ref * const_factor
vc_singlefield = np.array(a['vc'][t,kgc_center:kend,jgc:jend,igc:iend]) * utau_ref * const_factor
wc_singlefield = np.array(a['wc'][t,kgc_edge:khend,jgc:jend,igc:iend]) * utau_ref * const_factor

uc_avg = np.mean(uc_singlefield,axis=(1,2))
vc_avg = np.mean(vc_singlefield,axis=(1,2))
wc_avg = np.mean(wc_singlefield,axis=(1,2))
print(uc_avg)
print(vc_avg)
print(wc_avg)

#Write flow fields to binary files
uc_singlefield.tofile("u.restart")
vc_singlefield.tofile("v.restart")
wc_singlefield.tofile("w.restart")

#Close nc-file
a.close()
