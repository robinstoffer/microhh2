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
indexes_local_les = [0,1,2,3,16,32,47,60,61,62,63]

#Define height indices to consider the xy-crosssection for the DNS data. Ensure that the chosen indices are as close as possible to the indices for the training data (and the corresponding LES run in MicroHH), such that the resulting plots can be compared to each other. This also implies that the lengths of the arrays should be the same as the one defined above! Finally, note that different height indices are used for the u,v-components compared to the w-component to take into account the staggered grid orientation, such that for all components the DNS fields closest to the LES fields are selected.
indexes_local_dnsuv = [12,27,38,46,96,128,159,209,217,228,243]
indexes_local_dnsw  = [0,21,34,43,95,128,159,206,213,222,235]


#Fetch training data
training_filepath = "/projects/1/flowsim/simulation1/lesscoarse/training_data.nc"
a = nc.Dataset(training_filepath, 'r')

#Fetch DNS data
dnsu_filepath = "/projects/1/flowsim/simulation1/u.nc"
dnsu = nc.Dataset(dnsu_filepath, 'r')
dnsv_filepath = "/projects/1/flowsim/simulation1/v.nc"
dnsv = nc.Dataset(dnsv_filepath, 'r')
dnsw_filepath = "/projects/1/flowsim/simulation1/w.nc"
dnsw = nc.Dataset(dnsw_filepath, 'r')

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

#Read coordinates from the DNS files
zh = np.array(dnsw['zh'][:])
z  = np.array(dnsu['z'][:])
yh = np.array(dnsv['yh'][:])
y  = np.array(dnsu['y'][:])
xh = np.array(dnsu['xh'][:])
x  = np.array(dnsv['x'][:])

nz = z.shape[0]
ny = y.shape[0]
nx = x.shape[0]

#Read variables from netCDF-files for time step t=27 (corresponding to t=2820s and t=0 in dns field, which also used as starting point a posteriori test)
#NOTE: undo normalisation with friction velocity!
t_les=27 #
t_dns=0 #Calculate u,v,w.nc such that t=0 corresponds to time of training time step above!
uc_singlefield = np.array(a['uc'][t_les,kgc_center:kend,jgc:jend,igc:iend]) * ustar
vc_singlefield = np.array(a['vc'][t_les,kgc_center:kend,jgc:jend,igc:iend]) * ustar
wc_singlefield = np.array(a['wc'][t_les,kgc_center:kend,jgc:jend,igc:iend]) * ustar
u_singlefield  = np.array(dnsu['u'][t_dns,:,:,:])
v_singlefield  = np.array(dnsv['v'][t_dns,:,:,:])
w_singlefield  = np.array(dnsw['w'][t_dns,:,:,:])

#Loop over heights to calculate spectra
num_var = 3
nwave_modes_x_les = int(nxc * 0.5)
nwave_modes_y_les = int(nyc * 0.5)
nwave_modes_x_dns = int(nx * 0.5)
nwave_modes_y_dns = int(ny * 0.5)
num_idx = np.size(indexes_local_les) #Assume that all arrays with the indices have the same length, which should be the case.
spectra_x_les  = numpy.zeros((3,num_idx,nwave_modes_x_les))
spectra_y_les  = numpy.zeros((3,num_idx,nwave_modes_y_les))
pdf_fields_les = numpy.zeros((3,num_idx,nyc,nxc))
spectra_x_dns  = numpy.zeros((3,num_idx,nwave_modes_x_dns))
spectra_y_dns  = numpy.zeros((3,num_idx,nwave_modes_y_dns))
pdf_fields_dns = numpy.zeros((3,num_idx,ny,nx))
index_spectra = 0
for idx_var in range(num_var):
    stop = False
    for k in range(num_idx):
        print("Processing index " + str(k+1) + " of " + str(num_idx))
        index_les   = indexes_local_les[k]
        index_dnsuv = indexes_local_dnsuv[k]
        index_dnsw  = indexes_local_dnsw[k]
        if idx_var == 0:
            s_les = uc_singlefield[index_les,:,:]
            s_dns = u_singlefield[index_dnsuv,:,:]
        elif idx_var == 1:
            s_les = vc_singlefield[index_les,:,:]  
            s_dns = v_singlefield[index_dnsuv,:,:]
        else:
            s_les = wc_singlefield[index_les,:,:]
            s_dns = w_singlefield[index_dnsw,:,:]
        #LES
        fftx_les = numpy.fft.rfft(s_les,axis=1)*(1/nxc)
        ffty_les = numpy.fft.rfft(s_les,axis=0)*(1/nyc)
        Px_les = fftx_les[:,1:] * numpy.conjugate(fftx_les[:,1:])
        Py_les = ffty_les[1:,:] * numpy.conjugate(ffty_les[1:,:])
        if int(nxc % 2) == 0:
            Ex_les = np.append(2*Px_les[:,:-1],np.reshape(Px_les[:,-1],(nyc,1)),axis=1)
        else:
            Ex_les = 2*Px_les[:,:]
        
        if int(nyc % 2) == 0:
            Ey_les = np.append(2*Py_les[:-1,:],np.reshape(Py_les[-1,:],(1,nxc)),axis=0)
        else:
            Ey_les = 2*Py_les[:,:]
            
        spectra_x_les[index_spectra,k,:]    = numpy.nanmean(Ex_les,axis=0) #Average Fourier transform over the direction where it was not calculated
        spectra_y_les[index_spectra,k,:]    = numpy.nanmean(Ey_les,axis=1)
        pdf_fields_les[index_spectra,k,:,:] = s_les[:,:]
        #DNS
        fftx_dns = numpy.fft.rfft(s_dns,axis=1)*(1/nx)
        ffty_dns = numpy.fft.rfft(s_dns,axis=0)*(1/ny)
        Px_dns = fftx_dns[:,1:] * numpy.conjugate(fftx_dns[:,1:])
        Py_dns = ffty_dns[1:,:] * numpy.conjugate(ffty_dns[1:,:])
        if int(nxc % 2) == 0:
            Ex_dns = np.append(2*Px_dns[:,:-1],np.reshape(Px_dns[:,-1],(ny,1)),axis=1)
        else:
            Ex_dns = 2*Px_dns[:,:]
        
        if int(nyc % 2) == 0:
            Ey_dns = np.append(2*Py_dns[:-1,:],np.reshape(Py_dns[-1,:],(1,nx)),axis=0)
        else:
            Ey_dns = 2*Py_dns[:,:]
            
        spectra_x_dns[index_spectra,k,:]    = numpy.nanmean(Ex_dns,axis=0) #Average Fourier transform over the direction where it was not calculated
        spectra_y_dns[index_spectra,k,:]    = numpy.nanmean(Ey_dns,axis=1)
        pdf_fields_dns[index_spectra,k,:,:] = s_dns[:,:]

    index_spectra +=1

k_streamwise_les = np.arange(1,nwave_modes_x_les+1)
k_spanwise_les = np.arange(1,nwave_modes_y_les+1)
k_streamwise_dns = np.arange(1,nwave_modes_x_dns+1)
k_spanwise_dns = np.arange(1,nwave_modes_y_dns+1)

#Determine bins for pdfs
num_bins = 100
bin_edges_u_les = np.linspace(np.nanmin(pdf_fields_les[0,:,:]),np.nanmax(pdf_fields_les[0,:,:]), num_bins)
bin_edges_v_les = np.linspace(np.nanmin(pdf_fields_les[1,:,:]),np.nanmax(pdf_fields_les[1,:,:]), num_bins)
bin_edges_w_les = np.linspace(np.nanmin(pdf_fields_les[2,:,:]),np.nanmax(pdf_fields_les[2,:,:]), num_bins)
bin_edges_u_dns = np.linspace(np.nanmin(pdf_fields_dns[0,:,:]),np.nanmax(pdf_fields_dns[0,:,:]), num_bins)
bin_edges_v_dns = np.linspace(np.nanmin(pdf_fields_dns[1,:,:]),np.nanmax(pdf_fields_dns[1,:,:]), num_bins)
bin_edges_w_dns = np.linspace(np.nanmin(pdf_fields_dns[2,:,:]),np.nanmax(pdf_fields_dns[2,:,:]), num_bins)

#Plot balances
for k in range(num_idx):
    print("Plotting index " + str(k+1) + " of " + str(num_idx))
    #LES
    figure()
    loglog(k_streamwise_les[:], (spectra_x_les[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_streamwise_les[:], (spectra_x_les[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_streamwise_les[:], (spectra_x_les[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/les_spectrax_z_" + str(indexes_local_les[k]) + "_training.png")
    close()
    #
    figure()
    loglog(k_spanwise_les[:], (spectra_y_les[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_spanwise_les[:], (spectra_y_les[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_spanwise_les[:], (spectra_y_les[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/les_spectray_z_" + str(indexes_local_les[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_les[0,k,:,:].flatten(), bins = bin_edges_u_les, density = True, histtype = 'step', label = 'u')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([0, 0.16, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/les_pdfu_z_" + str(indexes_local_les[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_les[1,k,:,:].flatten(), bins = bin_edges_v_les, density = True, histtype = 'step', label = 'v')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.04, 0.04, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/les_pdfv_z_" + str(indexes_local_les[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_les[2,k,:,:].flatten(), bins = bin_edges_w_les, density = True, histtype = 'step', label = 'w')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.03, 0.03, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/les_pdfw_z_" + str(indexes_local_les[k]) + "_training.png")
    close()
    #
    #DNS
    figure()
    loglog(k_streamwise_dns[:], (spectra_x_dns[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_streamwise_dns[:], (spectra_x_dns[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_streamwise_dns[:], (spectra_x_dns[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/dns_spectrax_z_" + str(indexes_local_dnsuv[k]) + "_training.png")
    close()
    #
    figure()
    loglog(k_spanwise_dns[:], (spectra_y_dns[0,k,:] / ustar**2.), 'k-',linewidth=2.0, label='u')
    loglog(k_spanwise_dns[:], (spectra_y_dns[1,k,:] / ustar**2.), 'r-',linewidth=2.0, label='v')
    loglog(k_spanwise_dns[:], (spectra_y_dns[2,k,:] / ustar**2.), 'b-',linewidth=2.0, label='w')
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.000001, 3])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/dns_spectray_z_" + str(indexes_local_dnsuv[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_dns[0,k,:,:].flatten(), bins = bin_edges_u_dns, density = True, histtype = 'step', label = 'u')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([0, 0.16, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/dns_pdfu_z_" + str(indexes_local_dnsuv[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_dns[1,k,:,:].flatten(), bins = bin_edges_v_dns, density = True, histtype = 'step', label = 'v')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.04, 0.04, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/dns_pdfv_z_" + str(indexes_local_dnsuv[k]) + "_training.png")
    close()
    #
    figure()
    grid()
    hist(pdf_fields_dns[2,k,:,:].flatten(), bins = bin_edges_w_dns, density = True, histtype = 'step', label = 'w')
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    axis([-0.03, 0.03, 0, 140])
    tight_layout()
    savefig("/home/robinst/microhh/cases/moser600/git_repository/Training data/dns_pdfw_z_" + str(indexes_local_dnsw[k]) + "_training.png")
    close()
    #

#Close nc-file
a.close()
dnsu.close()
dnsv.close()
dnsw.close()
