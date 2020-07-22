#Visualize spectra in different stages of training data generation procedure: with or without filtering, and with or without coarse-grid interpolations.
import sys
import numpy
import struct
import netCDF4 as nc
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('PDF')
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}", r"\usepackage[utf8]{inputenc}"]
from matplotlib.pyplot import *
import subprocess

#Select time step used for visualization
t_dns = 1200  #Refers to time step of corresponding DNS binary fields
t_training = 0 #Refers to time step of a representative flow field in training data.
#NOTE: ensure that t_dns and t_training select the same time step

#Perform pre-processing needed for visualizing the spectra: for one of the training snapshots (here chosen to be the one at 3000s simulation time), convert the binary high-resolution DNS fields to netCDF-files
subprocess.run(["python3", "../python/3d_to_nc.py", "--directory", "../cases/moser600" ,"--filename", "moser600.ini", "--vars", "u", "--precision", "double", "-t0", str(t_dns), "-t1", str(t_dns), "-tstep", "1"])

#Define height index to consider the xy-crosssection for the training data
index_local_training = 3

#Define height index to consider the xy-crosssection for the DNS data. Ensure that the chosen index is as close as possible to the selected index of the training data (and the corresponding LES run in MicroHH), such that the resulting plots can be compared to each other.
index_local_dnsu = 46

#Fetch training data
training_filepath = "../cases/moser600/training_data/training_data.nc"
a = nc.Dataset(training_filepath, 'r')

#Fetch DNS data
dnsu_filepath = '../cases/moser600/u.nc'
dnsu = nc.Dataset(dnsu_filepath, 'r')

#Define settings (amongst others for normalization, currently reflect chosen Moser case
delta = 1
reynolds_tau_moser = 590
mvisc_ref_moser = a['mvisc'][:]
domainsize_x = 2 * np.pi
domainsize_y = np.pi

#Calculate friction velocity for normalisation
utau_ref_moser = (reynolds_tau_moser * mvisc_ref_moser) / delta

#Read ghost cells, indices, and coordinates from the training file
igc        = int(a['igc'][:])
jgc        = int(a['jgc'][:])
kgc_center = int(a['kgc_center'][:])
kgc_edge   = int(a['kgc_edge'][:])
iend       = int(a['iend'][:])
jend       = int(a['jend'][:])
kend       = int(a['kend'][:])
zhc        = np.array(a['zhc'][:])
zc         = np.array(a['zc'][:])
yhc        = np.array(a['yhc'][:])
yc         = np.array(a['yc'][:])
xhc        = np.array(a['xhc'][:])
xc         = np.array(a['xc'][:])

nzc = zc.shape[0]
nyc = yc.shape[0]
nxc = xc.shape[0]

#Read coordinates from the DNS files
z  = np.array(dnsu['z'][:])
y  = np.array(dnsu['y'][:])
xh = np.array(dnsu['xh'][:])

nz = z.shape[0]
ny = y.shape[0]
nx = xh.shape[0]

print('Height selected filtered fields: ', zc[index_local_training])
print('Height selected DNS fields: ', z[index_local_dnsu])

#Read filtered and DNS velocity variables
ufilt_singlefield = np.array(a['uc'][t_training,kgc_center:kend,jgc:jend,igc:iend])
udns_singlefield  = np.array(dnsu['u'][0,:,:,:]) #NOTE: assumes that high-res files only contain the desired time step, not other ones (and therefore simply is always the first one). This is valid for the current settings used in the pre-processing.

#Read filtered, interpolated u from nc-file
ufiltint_singlefield = np.array(a['res_u_boxint'][t_training,:,:,:])

#Calculate spectra
nwave_modes_x_filt    = int(nxc * 0.5)
nwave_modes_x_filtint = int(nxc * 0.5)
nwave_modes_x_dns     = int(nx * 0.5)
spectra_x_filt        = numpy.zeros((nwave_modes_x_filt))
spectra_x_filtint     = numpy.zeros((nwave_modes_x_filtint))
spectra_x_dns         = numpy.zeros((nwave_modes_x_dns))
index_spectra         = 0
#
s_filt    = ufilt_singlefield[index_local_training,:,:]
s_filtint = ufiltint_singlefield[index_local_training,:,:]
s_dns     = udns_singlefield[index_local_dnsu,:,:]

#filtered
fftx_filt = numpy.fft.rfft(s_filt,axis=1)*(1/nxc)
Px_filt = fftx_filt[:,1:] * numpy.conjugate(fftx_filt[:,1:])
if int(nxc % 2) == 0:
    Ex_filt = np.append(2*Px_filt[:,:-1],np.reshape(Px_filt[:,-1],(nyc,1)),axis=1)
else:
    Ex_filt = 2*Px_filt[:,:]

spectra_x_filt[:]    = numpy.nanmean(Ex_filt,axis=0) #Average Fourier transform over the direction where it was not calculated

#filtered+interpolated
fftx_filtint = numpy.fft.rfft(s_filtint,axis=1)*(1/nxc)
Px_filtint = fftx_filtint[:,1:] * numpy.conjugate(fftx_filtint[:,1:])
if int(nxc % 2) == 0:
    Ex_filtint = np.append(2*Px_filtint[:,:-1],np.reshape(Px_filtint[:,-1],(nyc,1)),axis=1)
else:
    Ex_filtint = 2*Px_filtint[:,:]

spectra_x_filtint[:]    = numpy.nanmean(Ex_filtint,axis=0) #Average Fourier transform over the direction where it was not calculated

#DNS
fftx_dns = numpy.fft.rfft(s_dns,axis=1)*(1/nx)
Px_dns = fftx_dns[:,1:] * numpy.conjugate(fftx_dns[:,1:])
if int(nxc % 2) == 0:
    Ex_dns = np.append(2*Px_dns[:,:-1],np.reshape(Px_dns[:,-1],(ny,1)),axis=1)
else:
    Ex_dns = 2*Px_dns[:,:]
    
spectra_x_dns[:]    = numpy.nanmean(Ex_dns,axis=0) #Average Fourier transform over the direction where it was not calculated

index_spectra +=1

#Set wave modes
n_streamwise_filt = np.arange(1,nwave_modes_x_filt+1)
n_streamwise_filtint = np.arange(1,nwave_modes_x_filtint+1)
n_streamwise_dns = np.arange(1,nwave_modes_x_dns+1)

#Calculate wave numbers
k_streamwise_filt = (n_streamwise_filt / domainsize_x) * 2 * np.pi
k_streamwise_filtint = (n_streamwise_filtint / domainsize_x) * 2 * np.pi
k_streamwise_dns = (n_streamwise_dns / domainsize_x) * 2 * np.pi

#Plot spectra DNS, filtered, filtered+interpolated
#LES
figure()
loglog(k_streamwise_dns[:]     * delta, (spectra_x_dns[:]     / (utau_ref_moser**2. * delta)), 'k-',linewidth=2.0, label='DNS')
loglog(k_streamwise_filt[:]    * delta, (spectra_x_filt[:]    / (utau_ref_moser**2. * delta)), 'r-',linewidth=2.0, label='filtered')
loglog(k_streamwise_filtint[:] * delta, (spectra_x_filtint[:] / (utau_ref_moser**2. * delta)), 'b-',linewidth=2.0, label='filtered+interpolated')

xlabel(r'$\kappa \delta \ [-]$',fontsize = 20)
ylabel(r'$E \,\ u_{\tau}^{-2} \,\ \delta^{-1} \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
fig = gcf()
fig.set_tight_layout(True)
fig.savefig("filtering_spectrax_z_" + str(zc[index_local_training]) + ".pdf")
close()

#Close nc-file
a.close()
dnsu.close()
