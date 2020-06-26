import numpy
import struct
import netCDF4
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
mpl.rcParams.update({'figure.autolayout':True})
from matplotlib.pyplot import *

nx = 768
ny = 384
nz = 256

#iter = 60000
#iterstep = 500
#nt   = 7
iter = 1200
iterstep = 600
nt = 11

# read Moser's data
Mosermean = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590.means", skiprows=25)
Moserrey  = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590.reystress", skiprows=25)

yplusMoser = Mosermean[:,1]
uavgMoser  = Mosermean[:,2]
uvarMoser  = Moserrey[:,2]
vvarMoser  = Moserrey[:,4]#In Moser data v and w are switched
wvarMoser  = Moserrey[:,3]

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

uvart = numpy.zeros((nt, nz))
vvart = numpy.zeros((nt, nz))
wvart = numpy.zeros((nt, nz))

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
  for k in range(nz):
    uvart[t,k] = numpy.var(u[k,:,:] - uavgt[t,k])
  del(u)

  fin = open("/projects/1/flowsim/simulation1/v.{:07d}".format(prociter),"rb")
  raw = fin.read(n*8)
  tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
  del(raw)
  v   = tmp.reshape((nz, ny, nx))
  del(tmp)
  fin.close()

  vavgt[t,:] = numpy.nanmean(numpy.nanmean(v,2),1)
  for k in range(nz):
    vvart[t,k] = numpy.var(v[k,:,:] - vavgt[t,k])
  del(v)

  fin = open("/projects/1/flowsim/simulation1/w.{:07d}".format(prociter),"rb")
  raw = fin.read(n*8)
  tmp = numpy.array(struct.unpack('<{}d'.format(n), raw))
  del(raw)
  w   = tmp.reshape((nz, ny, nx))
  del(tmp)
  fin.close()

  for k in range(nz):
    wvart[t,k] = numpy.var(w[k,:,:])
  del(w)

utotavgt = (uavgt**2. + vavgt**2.)**.5
visc     = 1.0e-5
ustart   = (visc * utotavgt[:,0] / z[0])**0.5

uavg = numpy.nanmean(uavgt,0)
vavg = numpy.nanmean(vavgt,0)
uvar = numpy.nanmean(uvart,0)
vvar = numpy.nanmean(vvart,0)
wvar = numpy.nanmean(wvart,0)

utotavg = numpy.nanmean(utotavgt,0)
#pdb.set_trace()
ustar = numpy.nanmean(ustart)

print('u_tau  = %.6f' % ustar)
print('Re_tau = %.2f' % (ustar / visc))

# create the theoretical lines
ypluslin = numpy.arange(0.5,15., 0.1)
ypluslog = numpy.arange(5.,800., 1.)
ulin     = ypluslin
ulog     = 2.5 * numpy.log( ypluslog ) + 5.

yplus  = z  * ustar / visc
yplush = zh * ustar / visc

starty = 0
endy   = int(z.size / 2)

close('all')
figure()
for t in range(nt):
  semilogx(yplus[starty:endy], utotavgt[t,starty:endy] / ustar, color='#cccccc',linewidth=2.0)
semilogx(yplus[starty:endy], utotavg[starty:endy] / ustar, 'k-',label="MicroHH",linewidth=2.0)
semilogx(yplusMoser, uavgMoser, 'ko', label="Moser",fillstyle = 'none',linewidth=2.0)
semilogx(ypluslin, ulin, 'k:',linewidth=2.0)
semilogx(ypluslog, ulog, 'k:',linewidth=2.0)
#gca().axis('Equal')
xlabel(r'$z \ [-]$',fontsize = 20)
ylabel(r'$\overline{u} \ [-]$',fontsize = 20)
legend(loc=2, frameon=False,fontsize = 16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([0.3, 600, 0, 22])
tight_layout()
savefig("moser600profiles_u.png")

figure()
for t in range(nt):
  semilogx(yplus [starty:endy], (uvart[t,starty:endy] / ustar**2.)**0.5, color='#cccccc',linewidth=2.0)
  semilogx(yplus [starty:endy], (vvart[t,starty:endy] / ustar**2.)**0.5, color='#cccccc',linewidth=2.0)
  semilogx(yplush[starty:endy], (wvart[t,starty:endy] / ustar**2.)**0.5, color='#cccccc',linewidth=2.0)
semilogx(yplus [starty:endy], (uvar[starty:endy] / ustar**2.)**0.5, 'k-',linewidth=2.0)
semilogx(yplus [starty:endy], (vvar[starty:endy] / ustar**2.)**0.5, 'r-',linewidth=2.0)
semilogx(yplush[starty:endy], (wvar[starty:endy] / ustar**2.)**0.5, 'b-',linewidth=2.0)
semilogx(yplusMoser, numpy.sqrt(uvarMoser), 'ko', label=r"$\overline{u'u'}$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, numpy.sqrt(vvarMoser), 'ro', label=r"$\overline{v'v'}$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, numpy.sqrt(wvarMoser), 'bo', label=r"$\overline{w'w'}$",fillstyle = 'none',linewidth=2.0)
#gca().axis('Equal')
xlabel(r'$z \ [-]$',fontsize = 20)
ylabel(r"$\overline{u'_{i}u'_{i}}^{1/2} \ [-]$",fontsize = 20)
legend(loc=0, frameon=False,fontsize = 16)
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
grid()
axis([0.3, 600, 0, 3.5])
tight_layout()
savefig("moser600profiles_rms.png") 
