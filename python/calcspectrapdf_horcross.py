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

#nx = 768
#ny = 384
#nz = 256

nx = 96
ny = 48
nz = 64

#iter = 60000
#iterstep = 500
#nt   = 7

iter = 1200

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
print("Processing iter = {:07d}".format(iter))

fin = open("/projects/1/flowsim/simulation1/u.{:07d}".format(iter),"rb")
raw = fin.read(n*8)
tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
del(raw)
u   = tmp.reshape((nz, ny, nx))
del(tmp)
fin.close()

uavg = numpy.nanmean(numpy.nanmean(u,2),1)

fin = open("/projects/1/flowsim/simulation1/v.{:07d}".format(iter),"rb")
raw = fin.read(n*8)
tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
del(raw)
v   = tmp.reshape((nz, ny, nx))
del(tmp)
fin.close()

vavg = numpy.nanmean(numpy.nanmean(v,2),1)

utotavg = (uavg**2. + vavg**2.)**.5
visc    = 1.0e-5
ustar   = (visc * utotavg[0] / z[0])**0.5

print('u_tau  = %.6f' % ustar)
print('Re_tau = %.2f' % (ustar / visc))

#Define height data
yplus  = z  * ustar / visc
yplush = zh * ustar / visc

starty = 0
endy   = int(z.size / 2)

# read cross-sections
variables=["u","v","w"]
nwave_modes_x = int(nx * 0.5)
nwave_modes_y = int(ny * 0.5)
spectra_x = numpy.zeros((3,nz,nwave_modes_x))
spectra_y = numpy.zeros((3,nz,nwave_modes_y))
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

		spectra_x[index_spectra,k,:] = numpy.nanmean(Ex,axis=0) #Average Fourier transorm over the direction where it was not calculated
		spectra_y[index_spectra,k,:] = numpy.nanmean(Ey,axis=1)
		fin.close()

	index_spectra +=1

k_streamwise = np.arange(1,nwave_modes_x+1)
k_spanwise = np.arange(1,nwave_modes_y+1)

#Plot balances
figure()
loglog(k_streamwise[:], (spectra_x[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_streamwise[:], (spectra_x[2,:] / ustar**2.), 'b-',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("spectra_xz.png")
close()
#
figure()
loglog(k_spanwise[:], (spectra_y[0,:] / ustar**2.), 'k-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y[1,:] / ustar**2.), 'r-',linewidth=2.0)
loglog(k_spanwise[:], (spectra_y[2,:] / ustar**2.), 'b-',linewidth=2.0)

xlabel(r'$\kappa \ [-]$',fontsize = 20)
ylabel(r'$E \ [-]$',fontsize = 20)
legend(loc=0, frameon=False,fontsize=16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([1, 250, 0.000001, 3])
tight_layout()
savefig("spectra_yz.png")
close()
