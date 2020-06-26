import sys
import numpy
import struct
import netCDF4
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *
sys.path.append("/home/robinst/microhh/python")
from microhh_tools_robinst import *

nx = 768
ny = 384
nz = 256

#nx = 96
#ny = 48
#nz = 64

#iter = 60000
#iterstep = 500
#nt   = 7
iter = 1200
iterstep = 600
nt = 11
#nt = 1

# read Moser's data
Moserxspec_z5 = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/spectra/chan590.xspec.5", skiprows=25)
Moserzspec_z5 = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/spectra/chan590.zspec.5", skiprows=25)
Moserxspec_z99 = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/spectra/chan590.xspec.99", skiprows=25)
Moserzspec_z99 = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/spectra/chan590.zspec.99", skiprows=25)

kMoser = Moserxspec_z5[1:,0]  # +1; Should be the same for all spectra, +1 to get the actual wave number kappa
Euu_xspec_z5_moser = Moserxspec_z5[1:,1]
Evv_xspec_z5_moser = Moserxspec_z5[1:,3] #Note: v and w switched in Moser, here taken into account
Eww_xspec_z5_moser = Moserxspec_z5[1:,2]
Epp_xspec_z5_moser = Moserxspec_z5[1:,4] 

Euu_zspec_z5_moser = Moserzspec_z5[1:,1]
Evv_zspec_z5_moser = Moserzspec_z5[1:,3] #Note: v and w switched in Moser, here taken into account
Eww_zspec_z5_moser = Moserzspec_z5[1:,2]
Epp_zspec_z5_moser = Moserzspec_z5[1:,4]

Euu_xspec_z99_moser = Moserxspec_z99[1:,1]
Evv_xspec_z99_moser = Moserxspec_z99[1:,3] #Note: v and w switched in Moser, here taken into account
Eww_xspec_z99_moser = Moserxspec_z99[1:,2]
Epp_xspec_z99_moser = Moserxspec_z99[1:,4]

Euu_zspec_z99_moser = Moserzspec_z99[1:,1]
Evv_zspec_z99_moser = Moserzspec_z99[1:,3] #Note: v and w switched in Moser, here taken into account
Eww_zspec_z99_moser = Moserzspec_z99[1:,2]
Epp_zspec_z99_moser = Moserzspec_z99[1:,4]

# read the grid data
n = nx*ny*nz

fin = open("/projects/1/flowsim/simulation1/grid.{:07d}".format(0),"rb")
raw = fin.read(nx*8)
x   = numpy.array(struct.unpack('<{}d'.format(nx), raw))
raw = fin.read(nx*8)
xh  = numpy.array(struct.unpack('<{}d'.format(nx), raw))
raw = fin.read(ny*8)
y   = numpy.array(struct.unpack('<{}d'.format(ny), raw))
raw = fin.read(ny*8)
yh  = numpy.array(struct.unpack('<{}d'.format(ny), raw))
raw = fin.read(nz*8)
z   = numpy.array(struct.unpack('<{}d'.format(nz), raw))
raw = fin.read(nz*8)
zh  = numpy.array(struct.unpack('<{}d'.format(nz), raw))
fin.close()

# read the 3d data and process it
uavgt = numpy.zeros((nt, nz))
vavgt = numpy.zeros((nt, nz))

for t in range(nt):
	prociter = iter + iterstep*t
	print("Processing iter = {:07d}".format(prociter))
	
	fin = open("/projects/1/flowsim/simulation1/u.{:07d}".format(prociter),"rb")
	raw = fin.read(n*8)
	tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
	del(raw)
	u   = tmp.reshape((nz, ny, nx))
	del(tmp)
	fin.close()
	
	uavgt[t,:] = numpy.nanmean(numpy.nanmean(u,2),1)
	
	fin = open("/projects/1/flowsim/simulation1/v.{:07d}".format(prociter),"rb")
	raw = fin.read(n*8)
	tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
	del(raw)
	v   = tmp.reshape((nz, ny, nx))
	del(tmp)
	fin.close()
	
	vavgt[t,:] = numpy.nanmean(numpy.nanmean(v,2),1)

utotavgt = (uavgt**2. + vavgt**2.)**.5
visc     = 1.0e-5
ustart   = (visc * utotavgt[:,0] / z[0])**0.5

uavg = numpy.nanmean(uavgt,0)
vavg = numpy.nanmean(vavgt,0)
utotavg = numpy.nanmean(utotavgt,0)
ustar = numpy.nanmean(ustart)

print('u_tau  = %.6f' % ustar)
print('Re_tau = %.2f' % (ustar / visc))

#Define height data
yplus  = z  * ustar / visc
yplush = zh * ustar / visc

starty = 0
endy   = int(z.size / 2)

# read cross-sections
variables=["u","v","w","p"]
nwave_modes_x = int(nx * 0.5)
nwave_modes_y = int(ny * 0.5)
spectra_x5t = numpy.zeros((4,nt,nwave_modes_x))
spectra_y5t = numpy.zeros((4,nt,nwave_modes_y))
spectra_x99t = numpy.zeros((4,nt,nwave_modes_x))
spectra_y99t = numpy.zeros((4,nt,nwave_modes_y))
index_spectra = 0
input_dir = '/projects/1/flowsim/simulation1/'
for crossname in variables:

	if(crossname == 'u'): loc = [1,0,0]
	elif(crossname=='v'): loc = [0,1,0]
	elif(crossname=='w'): loc = [0,0,1]
	else:                 loc = [0,0,0]
	
	locx = 'x' if loc[0] == 0 else 'xh'
	locy = 'y' if loc[1] == 0 else 'yh'
	locz = 'z' if loc[2] == 0 else 'zh'
	
	indexes_local = get_cross_indices(crossname, 'xy', filepath=input_dir)
	stop = False
	for t in range(nt):
		prociter = iter + iterstep*t
		print("Processing iter = {:07d}".format(prociter))
		if (stop):
			break
		height_diff1 = 99999 #Initialize parameters to use for selection height
		height_diff2 = 99999 
		for k in range(np.size(indexes_local)):
			index = indexes_local[k]
			zplus = yplus if locz=='z' else yplush
			f_in  = "{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, prociter)
			try:
				fin = open(input_dir + f_in, "rb")
			except:
				print('Stopping: cannot find file {}'.format(f_in))
				crossfile.sync()
				stop = True
				break
		
			print("Processing %8s, time=%7i, index=%4i"%(crossname, prociter, index))

			#fin = open("{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, prociter), "rb")
			raw = fin.read(nx*ny*8)
			tmp = np.array(st.unpack('{0}{1}d'.format("<", nx*ny), raw))
			del(raw)
			s = tmp.reshape((ny, nx))
			del(tmp)
			if (abs(zplus[index]-5) < height_diff1): #Store spectrum of crosssection closest to z+=5 
				fftx = numpy.fft.rfft(s,axis=1)*(1/nx)
				ffty = numpy.fft.rfft(s,axis=0)*(1/ny)
				Px = fftx[:,1:] * numpy.conjugate(fftx[:,1:])
				Py = ffty[1:,:] * numpy.conjugate(ffty[1:,:])
				if int(nx % 2) == 0:
					Ex = np.append(2*Px[:,:-1],np.reshape(Px[:,-1],(ny,1)),axis=1)
				else:
					Ex = 2*Px[:,:]
				
				if int(ny % 2) == 0:
					Ey = np.append(2*Py[:-1,:],np.reshape(Py[-1,:],(1,nx)),axis=0)
				else:
					Ey = 2*Py[:,:]

				spectra_x5t[index_spectra,t,:] = numpy.nanmean(Ex,axis=0) #Average Fourier transorm over the direction where it was not calculated
				spectra_y5t[index_spectra,t,:] = numpy.nanmean(Ey,axis=1)
				height_diff1 = abs(zplus[index]-5)            
			if (abs(zplus[index]-99) < height_diff2): #Store spectrum crosssection closest to z+=99
				fftx = numpy.fft.rfft(s,axis=1)*(1/nx)
				ffty = numpy.fft.rfft(s,axis=0)*(1/ny)
				Px = fftx[:,1:] * numpy.conjugate(fftx[:,1:])
				Py = ffty[1:,:] * numpy.conjugate(ffty[1:,:])
				if int(nx % 2) == 0:
					Ex = np.append(2*Px[:,:-1],np.reshape(Px[:,-1],(ny,1)),axis=1)
				else:
					Ex = 2*Px[:,:]

				if int(ny % 2) == 0:
					Ey = np.append(2*Py[:-1,:],np.reshape(Py[-1,:],(1,nx)),axis=0)
				else:
					Ey = (ffty[1:,:] * numpy.conjugate(ffty[1:,:]))


				spectra_x99t[index_spectra,t,:] = numpy.nanmean(Ex,axis=0)
				spectra_y99t[index_spectra,t,:] = numpy.nanmean(Ey,axis=1)
				height_diff2 = abs(zplus[index]-99)
			del(s)
			fin.close()

	index_spectra +=1

spectra_x5 = numpy.nanmean(spectra_x5t,axis=1)
spectra_y5 = numpy.nanmean(spectra_y5t,axis=1)
spectra_x99 = numpy.nanmean(spectra_x99t,axis=1)
spectra_y99 = numpy.nanmean(spectra_y99t,axis=1)

k_streamwise = np.arange(1,nwave_modes_x+1)
k_spanwise = np.arange(1,nwave_modes_y+1)

#Plot balances
figure()
loglog(k_streamwise[:], (spectra_x5[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x5[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x5[2,:] / ustar**2.), 'b-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x5[3,:] / ustar**4.), 'g-',linewidth=2.0)

loglog(kMoser[:],(Euu_xspec_z5_moser[:]), 'ko',label = r'$E_{uu}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Evv_xspec_z5_moser[:]), 'ro',label = r'$E_{vv}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Eww_xspec_z5_moser[:]), 'bo',label = r'$E_{ww}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Epp_xspec_z5_moser[:]), 'go',label = r'$E_{pp}$',fillstyle = 'none',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("moser600spectra_xz5.png")
close()
#
figure()
loglog(k_spanwise[:], (spectra_y5[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y5[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y5[2,:] / ustar**2.), 'b-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y5[3,:] / ustar**4.), 'g-',linewidth=2.0)

loglog(kMoser[:],(Euu_zspec_z5_moser[:]), 'ko',label = r'$E_{uu}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Evv_zspec_z5_moser[:]), 'ro',label = r'$E_{vv}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Eww_zspec_z5_moser[:]), 'bo',label = r'$E_{ww}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Epp_zspec_z5_moser[:]), 'go',label = r'$E_{pp}$',fillstyle = 'none',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("moser600spectra_yz5.png")
close()
#
figure()
loglog(k_streamwise[:], (spectra_x99[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x99[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x99[2,:] / ustar**2.), 'b-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x99[3,:] / ustar**4.), 'g-',linewidth=2.0)

loglog(kMoser[:],(Euu_xspec_z99_moser[:]), 'ko',label = r'$E_{uu}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Evv_xspec_z99_moser[:]), 'ro',label = r'$E_{vv}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Eww_xspec_z99_moser[:]), 'bo',label = r'$E_{ww}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Epp_xspec_z99_moser[:]), 'go',label = r'$E_{pp}$',fillstyle = 'none',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("moser600spectra_xz99.png")
close()
#
figure()
loglog(k_spanwise[:], (spectra_y99[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y99[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y99[2,:] / ustar**2.), 'b-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y99[3,:] / ustar**4.), 'g-',linewidth=2.0)

loglog(kMoser[:],(Euu_zspec_z99_moser[:]), 'ko',label = r'$E_{uu}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Evv_zspec_z99_moser[:]), 'ro',label = r'$E_{vv}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Eww_zspec_z99_moser[:]), 'bo',label = r'$E_{ww}$',fillstyle = 'none',linewidth=2.0)
loglog(kMoser[:],(Epp_zspec_z99_moser[:]), 'go',label = r'$E_{pp}$',fillstyle = 'none',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("moser600spectra_yz99.png")
close()

