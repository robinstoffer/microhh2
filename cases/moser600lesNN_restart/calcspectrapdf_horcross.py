import sys
import numpy as np
import struct as st
import netCDF4 as nc
import os
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *
sys.path.append("../../python")
from microhh_tools import *


#input_dir = './long_run_MLP53/'
#input_dir = './long_run_MLP53_time01/'
#input_dir = './long_run_MLP53_time001/'
#input_dir = './long_run_lookuptable/'
#input_dir = './long_run_MLP53_explicitfilt/'
#input_dir = './long_run_MLP53_time01_explicitfilt/'
input_dir = './long_run_MLP53_time001_explicitfilt/'

#stats_dir = './long_run_MLP53/long_run_MLP53.nc'
#stats_dir = './long_run_MLP53_time01_explicitfilt/moser600lesNN_default_0000000.nc'
stats_dir = './long_run_MLP53_time001_explicitfilt/moser600lesNN_default_0000000.nc'
#stats_dir = './long_run_MLP53_time01/moser600lesNN_default_0000000.nc'
#stats_dir = './long_run_MLP53_time001/moser600lesNN_default_0000000.nc'
#stats_dir = './long_run_lookuptable/long_run_lookuptable.nc'
#stats_dir = './long_run_lookuptable_explicitfilt.nc'
#stats_dir = './long_run_MLP53_explicitfilt/long_run_MLP53_explicitfilt.nc'
#stats_dir = './moser600lesNN_default_0000000.nc'
#stats_dir = 'test.nc'

#nx = 768
#ny = 384
#nz = 256

nx = 96
ny = 48
nz = 64

#iter = 60000
#iterstep = 500
#nt   = 7

iter_begin = 0
iterstep = 100
nt = 11
int_tstat = 10 #specify different value from 1 when stats are not stored every time step
#int_tstat = 1

# read the grid data
n = nx*ny*nz

fin = open(input_dir + "grid.{:07d}".format(0),"rb")
raw = fin.read(nx*8)
x   = np.array(st.unpack('<{}d'.format(nx), raw))
raw = fin.read(nx*8)
xh  = np.array(st.unpack('<{}d'.format(nx), raw))
raw = fin.read(ny*8)
y   = np.array(st.unpack('<{}d'.format(ny), raw))
raw = fin.read(ny*8)
yh  = np.array(st.unpack('<{}d'.format(ny), raw))
raw = fin.read(nz*8)
z   = np.array(st.unpack('<{}d'.format(nz), raw))
raw = fin.read(nz*8)
zh  = np.array(st.unpack('<{}d'.format(nz), raw))
fin.close()

#Calculate friction velocity over selected time steps
uavg = np.zeros((nt,nz))
vavg = np.zeros((nt,nz))
for t in range(nt):
    iter = iter_begin + t*iterstep
    print("Processing iter = {:07d} for u* calc".format(iter))

    fin = open(input_dir + "u.{:07d}".format(iter),"rb")

    raw = fin.read(n*8)
    tmp = np.array(st.unpack('<{}d'.format(n), raw))
    del(raw)
    u   = tmp.reshape((nz, ny, nx))
    del(tmp)
    uavg[t,:] = np.nanmean(u, axis=(1,2))
    del(u)
    fin.close()

    fin = open(input_dir + "v.{:07d}".format(iter),"rb")

    raw = fin.read(n*8)
    tmp = np.array(st.unpack('<{}d'.format(n), raw))
    del(raw)
    v   = tmp.reshape((nz, ny, nx))
    del(tmp)
    vavg[t,:] = np.nanmean(v, axis=(1,2))
    del(v)
    fin.close()

utotavg  = (uavg**2. + vavg**2.)**.5
utottavg = np.nanmean(utotavg,axis=0)
visc    = 1.0e-5
ustar   = (visc * utottavg[0] / z[0])**0.5

print('u_tau  = %.6f' % ustar)
print('Re_tau = %.2f' % (ustar / visc))

#Define height data
yplus  = z  * ustar / visc
yplush = zh * ustar / visc

starty = 0
endy   = int(z.size / 2)

# read cross-sections and calculate spectra
variables=["u","v","w"]
nwave_modes_x = int(nx * 0.5)

#Search for cross_indices, temporarily change search directory to input directory
#NOTE: assume each variable is stored at the same number of cross indices. If this is not the case, the script possibly crashes!
work_dir = os.getcwd()
os.chdir(input_dir)
indexes_local = get_cross_indices('u', 'xy')
os.chdir(work_dir)
#

num_idx = np.size(indexes_local)
spectra_x  = np.zeros((3,nt,num_idx,nwave_modes_x))
pdf_fields = np.zeros((3,nt,num_idx,ny,nx))
index_spectra = 0

for crossname in variables:
    
    if(crossname == 'u'): loc = [1,0,0]
    elif(crossname=='v'): loc = [0,1,0]
    elif(crossname=='w'): loc = [0,0,1]
    else:                 loc = [0,0,0]
    
    locx = 'x' if loc[0] == 0 else 'xh'
    locy = 'y' if loc[1] == 0 else 'yh'
    locz = 'z' if loc[2] == 0 else 'zh'
    
    #Search for cross_indices, temporarily change search directory to input directory
    work_dir = os.getcwd()
    os.chdir(input_dir)
    indexes_local = get_cross_indices(crossname, 'xy')
    os.chdir(work_dir)
    #
    stop = False
    for t in range(nt):
        iter = iter_begin + t*iterstep
        for k in range(num_idx):
            index = indexes_local[k]
            zplus = yplus if locz=='z' else yplush
            f_in  = "{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, iter)
            try:
                fin = open(input_dir + f_in, "rb")
            except:
                print('Stopping: cannot find file {}'.format(f_in))
                crossfile.sync()
                stop = True
                break
            
            print("Processing %8s, time=%7i, index=%4i"%(crossname, iter, index))
            
            #fin = open("{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, prociter), "rb")
            raw = fin.read(nx*ny*8)
            tmp = np.array(st.unpack('{0}{1}d'.format("<", nx*ny), raw))
            del(raw)
            s = tmp.reshape((ny, nx))
            del(tmp)
            fftx = np.fft.rfft(s,axis=1)*(1/nx)
            Px = fftx[:,1:] * np.conjugate(fftx[:,1:])
            if int(nx % 2) == 0:
                Ex = np.append(2*Px[:,:-1],np.reshape(Px[:,-1],(ny,1)),axis=1)
            else:
                Ex = 2*Px[:,:]
            
            spectra_x[index_spectra,t,k,:]    = np.nanmean(Ex,axis=0) #Average Fourier transform over the direction where it was not calculated
            pdf_fields[index_spectra,t,k,:,:] = s[:,:]
            fin.close()

    index_spectra +=1

k_streamwise = np.arange(1,nwave_modes_x+1)

#Plot predicted transport components and calculated TKE as function of z for specified time range, chosen to be identical to time steps plotted before
tstart = iter_begin
tend  = iter_begin + iterstep * (nt-1)
stats  = nc.Dataset(stats_dir,'r')
z      = np.array(stats['z'][:])
zh     = np.array(stats['zh'][:])
tau_wu = np.array(stats['default']['u_diff'][tstart:tend,:])
tke    = np.array(stats['budget']['tke'][tstart:tend,:])
#
figure()
colors = cm.Blues(np.linspace(0.4,1,nt)) #Start at 0.4 to leave out white range of colormap
c=0
#for t in range(tstart, tend, iterstep):
for t in range(int(tstart/int_tstat), int(tend/int_tstat), int(iterstep/int_tstat)):
    plot(zh[:], (tau_wu[t,:] / ustar**2.), color=colors[c],linewidth=2.0, label=str(t))
    #plot(zh[:8], (tau_wu[t,:8] / ustar**2.), marker='.', mew = 2, mec=colors[t], mfc=colors[t], color=colors[t],linewidth=2.0, label=str(t))
    c+=1
xlabel(r'$\frac{z}{\delta} \ [-]$',fontsize = 20)
ylabel(r'$\frac{\tau_{wu}}{u_{\tau}^2} \ [-]$',fontsize = 20)
#legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([0, 2, -2.0, 1.5])
#axis([0, 0.25, -2.0, 0])
tight_layout()
savefig(input_dir + "tau_wu.png")
close()
#
figure()
colors = cm.Blues(np.linspace(0.4,1,nt)) #Start at 0.4 to leave out white range of colormap
c=0
for t in range(int(tstart/int_tstat), int(tend/int_tstat), int(iterstep/int_tstat)):
    plot(z[:], (tke[t,:] / ustar**2.), color=colors[c],linewidth=2.0, label=str(t))
    #plot(z[:8], (tke[t,:8] / ustar**2.), marker='.', mew = 2, mec=colors[t], mfc=colors[t], color=colors[t], linewidth=2.0, label=str(t))
    c+=1
xlabel(r'$\frac{z}{\delta} \ [-]$',fontsize = 20)
ylabel(r'$\frac{TKE}{u_{\tau}^2} \ [-]$',fontsize = 20)
#legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([0, 2, 0.5, 6.0])
#axis([0, 0.25, 1.5, 4.5])
tight_layout()
savefig(input_dir + "tke.png")
close()

#Determine bins for pdfs based only on first time step
num_bins = 25
bin_edges_u = np.linspace(np.nanmin(pdf_fields[0,0,:,:]),np.nanmax(pdf_fields[0,0,:,:]), num_bins)
bin_edges_v = np.linspace(np.nanmin(pdf_fields[1,0,:,:]),np.nanmax(pdf_fields[1,0,:,:]), num_bins)
bin_edges_w = np.linspace(np.nanmin(pdf_fields[2,0,:,:]),np.nanmax(pdf_fields[2,0,:,:]), num_bins)

#Plot spectra and pdfs

#Search for cross_indices, temporarily change search directory to input directory
#NOTE: assume each variable is stored at the same number of cross indices. If this is not the case, the script possibly crashes!
work_dir = os.getcwd()
os.chdir(input_dir)
indexes_local = get_cross_indices('u', 'xy')
os.chdir(work_dir)
#

for k in range(np.size(indexes_local)):
    figure()
    colors = cm.Blues(np.linspace(0.4,1,nt)) #Start at 0.4 to leave out white range of colormap
    for t in range(nt):
        loglog(k_streamwise[:], (spectra_x[0,t,k,:] / ustar**2.), color=colors[t],linewidth=2.0, label=str(t))
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.00001, 3])
    tight_layout()
    savefig(input_dir + "spectrax_u_z_" + str(indexes_local[k]) + ".png")
    close()
    #
    figure()
    colors = cm.Blues(np.linspace(0.4,1,nt)) #Start at 0.4 to leave out white range of colormap
    for t in range(nt):
        loglog(k_streamwise[:], (spectra_x[1,t,k,:] / ustar**2.), color=colors[t],linewidth=2.0, label=str(t))
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.00001, 3])
    tight_layout()
    savefig(input_dir + "spectrax_v_z_" + str(indexes_local[k]) + ".png")
    close()
    #
    figure()
    colors = cm.Blues(np.linspace(0.4,1,nt)) #Start at 0.4 to leave out white range of colormap
    for t in range(nt):
        loglog(k_streamwise[:], (spectra_x[2,t,k,:] / ustar**2.), color=colors[t],linewidth=2.0, label=str(t))
    
    xlabel(r'$\kappa \ [-]$',fontsize = 20)
    ylabel(r'$E \ [-]$',fontsize = 20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([1, 250, 0.00001, 3])
    tight_layout()
    savefig(input_dir + "spectrax_w_z_" + str(indexes_local[k]) + ".png")
    close()
    #
    figure()
    grid()
    colors = cm.Blues(np.linspace(0.4,1,nt))
    for t in range(nt):
        hist(pdf_fields[0,t,k,:,:].flatten(), bins = bin_edges_u, color=colors[t], density = True, histtype = 'step', label =str(t))
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([0, 0.16, 0, 100])
    tight_layout()
    savefig(input_dir + "pdf_u_z_" + str(indexes_local[k]) + ".png")
    close()
    #
    figure()
    grid()
    colors = cm.Blues(np.linspace(0.4,1,nt))
    for t in range(nt):
        hist(pdf_fields[1,t,k,:,:].flatten(), bins = bin_edges_v, color=colors[t], density = True, histtype = 'step', label =str(t))
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([-0.04,0.04,0,100])
    tight_layout()
    savefig(input_dir + "pdf_v_z_" + str(indexes_local[k]) + ".png")
    close()
    #
    figure()
    grid()
    colors = cm.Blues(np.linspace(0.4,1,nt))
    for t in range(nt):
        hist(pdf_fields[2,t,k,:,:].flatten(), bins = bin_edges_w, color=colors[t], density = True, histtype = 'step', label =str(t))
    ylabel(r'$\rm Normalized\ Density\ [-]$', fontsize=20)
    xlabel(r'$\rm Wind\ velocity\ {[m\ s^{-1}]}$', fontsize=20)
    #legend(loc=0, frameon=False,fontsize=16)
    xticks(fontsize = 16, rotation = 90)
    yticks(fontsize = 16, rotation = 0)
    grid()
    axis([-0.03,0.03,0,100])
    tight_layout()
    savefig(input_dir + "pdf_w_z_" + str(indexes_local[k]) + ".png")
    close()