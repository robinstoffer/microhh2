import sys
import numpy
import struct
import netCDF4 as nc
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *

#Define height indices to consider the xy-crosssection for the training data. When comparing the plots from this script to the plots of the equivalent script for the horizontal crosssections produced by MicroHH, ensure that these height indices are consistent!!!
indexes_local_box = [0,1,2,3,16,32,47,60,61,62,63]
indexes_local_les = [0,1,2,3,16,32,47,60,61,62,63]

#Define height indices to consider the xy-crosssection for the DNS data. Ensure that the chosen indices are as close as possible to the indices for the training data (and the corresponding LES run in MicroHH), such that the resulting plots can be compared to each other. This also implies that the lengths of the arrays should be the same as the one defined above! Finally, note that different height indices are used for the u,v-components compared to the w-component to take into account the staggered grid orientation, such that for all components the DNS fields closest to the LES fields are selected.
indexes_local_dnsuv = [12,27,38,46,96,128,159,209,217,228,243]
#indexes_local_dnsw  = [0,21,34,43,95,128,159,206,213,222,235]


#Fetch training data
training_filepath = "/media/sf_Shared_folder/training_data_coarse3d.nc"
a = nc.Dataset(training_filepath, 'r')

#Fetch DNS data, focus only on uu-component
dnsu_filepath = "/home/robin/microhh2/cases/moser600lesNN_restart/restart_files/u.nc"
dnsu = nc.Dataset(dnsu_filepath, 'r')
#dnsv_filepath = "/projects/1/flowsim/simulation1/v.nc"
#dnsv = nc.Dataset(dnsv_filepath, 'r')
#dnsw_filepath = "/projects/1/flowsim/simulation1/w.nc"
#dnsw = nc.Dataset(dnsw_filepath, 'r')

#Read friction velocity
ustar   = np.array(a['utau_ref'][:], dtype = 'f4')

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
nc = nzc*nyc*nxc

#Read coordinates from the DNS files
#zh = np.array(dnsw['zh'][:])
z  = np.array(dnsu['z'][:])
#yh = np.array(dnsv['yh'][:])
y  = np.array(dnsu['y'][:])
xh = np.array(dnsu['xh'][:])
#x  = np.array(dnsv['x'][:])

nz = z.shape[0]
ny = y.shape[0]
nx = xh.shape[0] #xh from dns file has same length as x

#Read variables from netCDF-files for time step t=27 (corresponding to t=2820s and t=0 in dns field, which also used as starting point a posteriori test)
#NOTE: undo normalisation with friction velocity!
t_box=27 #
t_les=0
t_dns=0 #Calculate u,v,w.nc such that t=0 corresponds to time of training time step above!
uc_singlefield = np.array(a['uc'][t_box,kgc_center:kend,jgc:jend,igc:iend]) * ustar
#vc_singlefield = np.array(a['vc'][t_box,kgc_center:kend,jgc:jend,igc:iend]) * ustar
#wc_singlefield = np.array(a['wc'][t_box,kgc_center:kend,jgc:jend,igc:iend]) * ustar
u_singlefield  = np.array(dnsu['u'][t_dns,:,:,:])
#v_singlefield  = np.array(dnsv['v'][t_dns,:,:,:])
#w_singlefield  = np.array(dnsw['w'][t_dns,:,:,:])

#Read variables from a posteriori LES
#iter= 0 + 60*t_les
iter= 4
fin = open("/home/robin/microhh2/cases/moser600lesNN_restart/u.{:07d}".format(iter),"rb")
raw = fin.read(nc*8)
tmp = np.array(struct.unpack('<{}d'.format(nc), raw))
del(raw)
ul_singlefield   = tmp.reshape((nzc, nyc, nxc))
del(tmp)
fin.close()

#Read resolved transport uu from nc-file, denormalize values
res_tau_xu_turb = np.array(a['res_tau_xu_turb'][t_les,:,:,:-1]) * ustar * ustar #Remove additional ghost cell in staggered direction to give each unique value the same weight

#Take square of velocity variables
uc2_singlefield = uc_singlefield ** 2.
ul2_singlefield  = ul_singlefield ** 2.
u2_singlefield   = u_singlefield  ** 2.

#Loop over heights to calculate spectra
nwave_modes_x_box = int(nxc * 0.5)
nwave_modes_y_box = int(nyc * 0.5)
nwave_modes_x_div = int(nxc * 0.5)
nwave_modes_y_div = int(nyc * 0.5)
nwave_modes_x_les = int(nxc * 0.5)
nwave_modes_y_les = int(nyc * 0.5)
nwave_modes_x_dns = int(nx * 0.5)
nwave_modes_y_dns = int(ny * 0.5)
num_idx = np.size(indexes_local_les) #Assume that all arrays with the indices have the same length, which should be the case.
spectra_x_u_box  = numpy.zeros((num_idx,nwave_modes_x_box))
spectra_y_u_box  = numpy.zeros((num_idx,nwave_modes_y_box))
pdf_fields_u_box = numpy.zeros((num_idx,nyc,nxc))
spectra_x_u_div  = numpy.zeros((num_idx,nwave_modes_x_div))
spectra_y_u_div  = numpy.zeros((num_idx,nwave_modes_y_div))
pdf_fields_u_div = numpy.zeros((num_idx,nyc,nxc))
spectra_x_u_les  = numpy.zeros((num_idx,nwave_modes_x_les))
spectra_y_u_les  = numpy.zeros((num_idx,nwave_modes_y_les))
pdf_fields_u_les = numpy.zeros((num_idx,nyc,nxc))
spectra_x_u_dns  = numpy.zeros((num_idx,nwave_modes_x_dns))
spectra_y_u_dns  = numpy.zeros((num_idx,nwave_modes_y_dns))
pdf_fields_u_dns = numpy.zeros((num_idx,ny,nx))
for k in range(num_idx):
    print("Processing index " + str(k+1) + " of " + str(num_idx))
    index_box   = indexes_local_box[k]
    index_les   = indexes_local_les[k]
    index_dnsuv = indexes_local_dnsuv[k]
    #index_dnsw  = indexes_local_dnsw[k]
    s_box_u = uc2_singlefield[index_box,:,:]
    s_div_u = res_tau_xu_turb[index_les,:,:]
    s_les_u = ul2_singlefield[index_les,:,:]
    s_dns_u = u2_singlefield[index_dnsuv,:,:]
    #box
    fftx_box_u = numpy.fft.rfft(s_box_u,axis=1)*(1/nxc)
    ffty_box_u = numpy.fft.rfft(s_box_u,axis=0)*(1/nyc)
    Px_box_u = fftx_box_u[:,1:] * numpy.conjugate(fftx_box_u[:,1:])
    Py_box_u = ffty_box_u[1:,:] * numpy.conjugate(ffty_box_u[1:,:])
    if int(nxc % 2) == 0:
        Ex_box_u = np.append(2*Px_box_u[:,:-1],np.reshape(Px_box_u[:,-1],(nyc,1)),axis=1)
    else:
        Ex_box_u = 2*Px_box_u[:,:]
    
    if int(nyc % 2) == 0:
        Ey_box_u = np.append(2*Py_box_u[:-1,:],np.reshape(Py_box_u[-1,:],(1,nxc)),axis=0)
    else:
        Ey_box_u = 2*Py_box_u[:,:]
        
    spectra_x_u_box[k,:]    = numpy.nanmean(Ex_box_u,axis=0) #Average Fourier transform over the direction where it was not calculated
    spectra_y_u_box[k,:]    = numpy.nanmean(Ey_box_u,axis=1)
    pdf_fields_u_box[k,:,:] = s_box_u[:,:]
    #div
    fftx_div_u = numpy.fft.rfft(s_div_u,axis=1)*(1/nxc)
    ffty_div_u = numpy.fft.rfft(s_div_u,axis=0)*(1/nyc)
    Px_div_u = fftx_div_u[:,1:] * numpy.conjugate(fftx_div_u[:,1:])
    Py_div_u = ffty_div_u[1:,:] * numpy.conjugate(ffty_div_u[1:,:])
    if int(nxc % 2) == 0:
        Ex_div_u = np.append(2*Px_div_u[:,:-1],np.reshape(Px_div_u[:,-1],(nyc,1)),axis=1)
    else:
        Ex_div_u = 2*Px_div_u[:,:]
    
    if int(nyc % 2) == 0:
        Ey_div_u = np.append(2*Py_div_u[:-1,:],np.reshape(Py_div_u[-1,:],(1,nxc)),axis=0)
    else:
        Ey_div_u = 2*Py_div_u[:,:]
        
    spectra_x_u_div[k,:]    = numpy.nanmean(Ex_div_u,axis=0) #Average Fourier transform over the direction where it was not calculated
    spectra_y_u_div[k,:]    = numpy.nanmean(Ey_div_u,axis=1)
    pdf_fields_u_div[k,:,:] = s_div_u[:,:]
    ##LES
    fftx_les_u = numpy.fft.rfft(s_les_u,axis=1)*(1/nxc)
    ffty_les_u = numpy.fft.rfft(s_les_u,axis=0)*(1/nyc)
    Px_les_u = fftx_les_u[:,1:] * numpy.conjugate(fftx_les_u[:,1:])
    Py_les_u = ffty_les_u[1:,:] * numpy.conjugate(ffty_les_u[1:,:])
    if int(nxc % 2) == 0:
        Ex_les_u = np.append(2*Px_les_u[:,:-1],np.reshape(Px_les_u[:,-1],(nyc,1)),axis=1)
    else:
        Ex_les_u = 2*Px_les_u[:,:]
    
    if int(nyc % 2) == 0:
        Ey_les_u = np.append(2*Py_les_u[:-1,:],np.reshape(Py_les_u[-1,:],(1,nxc)),axis=0)
    else:
        Ey_les_u = 2*Py_les_u[:,:]
        
    spectra_x_u_les[k,:]    = numpy.nanmean(Ex_les_u,axis=0) #Average Fourier transform over the direction where it was not calculated
    spectra_y_u_les[k,:]    = numpy.nanmean(Ey_les_u,axis=1)
    pdf_fields_u_les[k,:,:] = s_les_u[:,:]
    #DNS
    fftx_dns_u = numpy.fft.rfft(s_dns_u,axis=1)*(1/nx)
    ffty_dns_u = numpy.fft.rfft(s_dns_u,axis=0)*(1/ny)
    Px_dns_u = fftx_dns_u[:,1:] * numpy.conjugate(fftx_dns_u[:,1:])
    Py_dns_u = ffty_dns_u[1:,:] * numpy.conjugate(ffty_dns_u[1:,:])
    if int(nxc % 2) == 0:
        Ex_dns_u = np.append(2*Px_dns_u[:,:-1],np.reshape(Px_dns_u[:,-1],(ny,1)),axis=1)
    else:
        Ex_dns_u = 2*Px_dns_u[:,:]
    
    if int(nyc % 2) == 0:
        Ey_dns_u = np.append(2*Py_dns_u[:-1,:],np.reshape(Py_dns_u[-1,:],(1,nx)),axis=0)
    else:
        Ey_dns_u = 2*Py_dns_u[:,:]
        
    spectra_x_u_dns[k,:]    = numpy.nanmean(Ex_dns_u,axis=0) #Average Fourier transform over the direction where it was not calculated
    spectra_y_u_dns[k,:]    = numpy.nanmean(Ey_dns_u,axis=1)
    pdf_fields_u_dns[k,:,:] = s_dns_u[:,:]

k_streamwise_box = np.arange(1,nwave_modes_x_box+1)
k_spanwise_box = np.arange(1,nwave_modes_y_box+1)
k_streamwise_div = np.arange(1,nwave_modes_x_div+1)
k_spanwise_div = np.arange(1,nwave_modes_y_div+1)
k_streamwise_les = np.arange(1,nwave_modes_x_les+1)
k_spanwise_les = np.arange(1,nwave_modes_y_les+1)
k_streamwise_dns = np.arange(1,nwave_modes_x_dns+1)
k_spanwise_dns = np.arange(1,nwave_modes_y_dns+1)

#Determine bins for pdfs
num_bins = 100
bin_edges_u_box = np.linspace(np.nanmin(pdf_fields_u_box[:,:]),np.nanmax(pdf_fields_u_box[:,:]), num_bins)
bin_edges_u_div = np.linspace(np.nanmin(pdf_fields_u_div[:,:]),np.nanmax(pdf_fields_u_div[:,:]), num_bins)
bin_edges_u_les = np.linspace(np.nanmin(pdf_fields_u_les[:,:]),np.nanmax(pdf_fields_u_les[:,:]), num_bins)
bin_edges_u_dns = np.linspace(np.nanmin(pdf_fields_u_dns[:,:]),np.nanmax(pdf_fields_u_dns[:,:]), num_bins)

#Plot balances
for k in range(num_idx):
    print("Plotting index " + str(k+1) + " of " + str(num_idx))
    #u
    figure()
    loglog(k_streamwise_box[:], (spectra_x_u_box[k,:] / ustar**2.), 'k-',linewidth=2.0, label='box')
    loglog(k_streamwise_div[:], (spectra_x_u_div[k,:] / ustar**2.), 'g-',linewidth=2.0, label='div')
    loglog(k_streamwise_les[:], (spectra_x_u_les[k,:] / ustar**2.), 'r-',linewidth=2.0, label='les')
    loglog(k_streamwise_dns[:], (spectra_x_u_dns[k,:] / ustar**2.), 'b-',linewidth=2.0, label='dns')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000000001, 0.3])
    tight_layout()
    savefig("/home/robin/microhh2/cases/moser600lesNN_restart/uu_spectrax_z_" + str(indexes_local_les[k]) + ".png")
    close()
    #
    figure()
    loglog(k_spanwise_box[:], (spectra_y_u_box[k,:] / ustar**2.), 'k-',linewidth=2.0, label='box')
    loglog(k_spanwise_div[:], (spectra_y_u_div[k,:] / ustar**2.), 'g-',linewidth=2.0, label='div')
    loglog(k_spanwise_les[:], (spectra_y_u_les[k,:] / ustar**2.), 'r-',linewidth=2.0, label='les')
    loglog(k_spanwise_dns[:], (spectra_y_u_dns[k,:] / ustar**2.), 'b-',linewidth=2.0, label='dns')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000000001, 0.3])
    tight_layout()
    savefig("/home/robin/microhh2/cases/moser600lesNN_restart/uu_spectray_z_" + str(indexes_local_les[k]) + ".png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_u_box[k,:,:].flatten(), bins = bin_edges_u_box, density = True, histtype = 'step', label = 'box')
    hist(pdf_fields_u_div[k,:,:].flatten(), bins = bin_edges_u_div, density = True, histtype = 'step', label = 'div')
    hist(pdf_fields_u_les[k,:,:].flatten(), bins = bin_edges_u_les, density = True, histtype = 'step', label = 'les')
    hist(pdf_fields_u_dns[k,:,:].flatten(), bins = bin_edges_u_dns, density = True, histtype = 'step', label = 'dns')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([0, 0.16, 0, 140])
    tight_layout()
    savefig("/home/robin/microhh2/cases/moser600lesNN_restart/uu_pdfu_z_" + str(indexes_local_les[k]) + ".png")
    close()
    #

#Close nc-file
a.close()
dnsu.close()
#dnsv.close()
#dnsw.close()
