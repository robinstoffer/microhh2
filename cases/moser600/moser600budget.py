import sys
import numpy
import struct
import netCDF4
#import pdb
#import tkinter
import matplotlib as mpl
mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
from matplotlib.pyplot import *
sys.path.append("/home/robinst/microhh/python")
from microhh_tools import *

nx = 768
ny = 384
nz = 256

#iter = 60000
#iterstep = 500
#nt   = 7
iter = 1200
iterstep = 600
nt = 11
nbegin_nc = 20 #Don't take average over all time samples for variables from netCDF files (i.e. exclude spin-up period).
nend_nc = 120

# read Moser's data
Moseruubal = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/balances/chan590.uubal", skiprows=25)
Moservvbal = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/balances/chan590.wwbal", skiprows=25) #In Moser case v and w are switched
Moserwwbal = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/balances/chan590.vvbal", skiprows=25)
Moserkbal = numpy.loadtxt("/home/robinst/microhh/cases/moser600/chan590/balances/chan590.kbal", skiprows=25)

yplusMoser = Moseruubal[:,1] #Should be the same for u,v,w,TKE
P_umoser = Moseruubal[:,4]
S_umoser = Moseruubal[:,3]
Tt_umoser = Moseruubal[:,6]
Tv_umoser = Moseruubal[:,7]
eps_umoser = Moseruubal[:,2]

P_vmoser = Moservvbal[:,4]
S_vmoser = Moservvbal[:,3]
Tt_vmoser = Moservvbal[:,6]
Tv_vmoser = Moservvbal[:,7]
eps_vmoser = Moservvbal[:,2]

P_wmoser = Moserwwbal[:,4]
Tt_wmoser = Moserwwbal[:,6]
Tv_wmoser = Moserwwbal[:,7]
eps_wmoser = Moserwwbal[:,2]
Tp_wmoser = Moserwwbal[:,5]

S_kmoser = Moserkbal[:,3]
Tt_kmoser = Moserkbal[:,6]
Tv_kmoser = Moserkbal[:,7]
eps_kmoser = Moserkbal[:,2]
Tp_kmoser = Moserkbal[:,5]

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

# read statistics file
f = Read_statistics("/projects/1/flowsim/simulation1/moser600.default.0000000.nc")
P_ut = numpy.array(f['u2_rdstr'][nbegin_nc:nend_nc,:])
S_ut = numpy.array(f['u2_shear'][nbegin_nc:nend_nc,:])
Tt_ut = numpy.array(f['u2_turb'][nbegin_nc:nend_nc,:])
Tv_ut = numpy.array(f['u2_visc'][nbegin_nc:nend_nc,:])
eps_ut = numpy.array(f['u2_diss'][nbegin_nc:nend_nc,:])
#
P_vt = numpy.array(f['v2_rdstr'][nbegin_nc:nend_nc,:])
S_vt = numpy.array(f['v2_shear'][nbegin_nc:nend_nc,:])
Tt_vt = numpy.array(f['v2_turb'][nbegin_nc:nend_nc,:])
Tv_vt = numpy.array(f['v2_visc'][nbegin_nc:nend_nc,:])
eps_vt = numpy.array(f['v2_diss'][nbegin_nc:nend_nc,:])
#
P_wt = numpy.array(f['w2_rdstr'][nbegin_nc:nend_nc,:])
Tp_wt = numpy.array(f['w2_pres'][nbegin_nc:nend_nc,:])
Tt_wt = numpy.array(f['w2_turb'][nbegin_nc:nend_nc,:])
Tv_wt = numpy.array(f['w2_visc'][nbegin_nc:nend_nc,:])
eps_wt = numpy.array(f['w2_diss'][nbegin_nc:nend_nc,:])
#
S_kt = numpy.array(f['tke_shear'][nbegin_nc:nend_nc,:])
Tp_kt = numpy.array(f['tke_pres'][nbegin_nc:nend_nc,:])
Tt_kt = numpy.array(f['tke_turb'][nbegin_nc:nend_nc,:])
Tv_kt = numpy.array(f['tke_visc'][nbegin_nc:nend_nc,:])
eps_kt = numpy.array(f['tke_diss'][nbegin_nc:nend_nc,:])

#Average over time
P_u = numpy.nanmean(P_ut,0)
S_u = numpy.nanmean(S_ut,0)
Tt_u = numpy.nanmean(Tt_ut,0)
Tv_u = numpy.nanmean(Tv_ut,0)
eps_u = numpy.nanmean(eps_ut,0)
#
P_v = numpy.nanmean(P_vt,0)
S_v = numpy.nanmean(S_vt,0)
Tt_v = numpy.nanmean(Tt_vt,0)
Tv_v = numpy.nanmean(Tv_vt,0)
eps_v = numpy.nanmean(eps_vt,0)
#
P_w = numpy.nanmean(P_wt,0)
Tp_w = numpy.nanmean(Tp_wt,0)
Tt_w = numpy.nanmean(Tt_wt,0)
Tv_w = numpy.nanmean(Tv_wt,0)
eps_w = numpy.nanmean(eps_wt,0)
#
Tp_k = numpy.nanmean(Tp_kt,0)
S_k = numpy.nanmean(S_kt,0)
Tt_k = numpy.nanmean(Tt_kt,0)
Tv_k = numpy.nanmean(Tv_kt,0)
eps_k = numpy.nanmean(eps_kt,0)

#Plot balances
figure()
semilogx(yplus[starty:endy], (P_u[starty:endy] * visc / ustar**4.), 'm-',linewidth=2.0)
semilogx(yplus[starty:endy], (S_u[starty:endy] * visc / ustar**4.), 'r-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tt_u[starty:endy] * visc / ustar**4.), 'b-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tv_u[starty:endy] * visc / ustar**4.), 'g-',linewidth=2.0)
semilogx(yplus[starty:endy], (eps_u[starty:endy] * visc / ustar**4.), 'k-',linewidth=2.0)

semilogx(yplusMoser, P_umoser, 'mo', label=r"$P$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, S_umoser, 'ro', label=r"$S$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tt_umoser, 'bo', label=r"$T_t$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tv_umoser, 'go', label=r"$T_v$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, eps_umoser, 'ko', label=r"$\epsilon$",fillstyle = 'none',linewidth=2.0)

xlabel(r'$z \ [-]$', fontsize = 20)
ylabel(r"$\partial \ \overline{u'u'} \ / \partial t \ [-]$",fontsize=20)
legend(loc=4, frameon=False,fontsize=16)
grid()
axis([0.3, 600, -0.5, 0.5])
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
tight_layout()
savefig("moser600budget_uvar.png")
close()
#
figure()
semilogx(yplus[starty:endy], (P_v[starty:endy] * visc / ustar**4.), 'm-',linewidth=2.0)
semilogx(yplus[starty:endy], (S_v[starty:endy] * visc / ustar**4.), 'r-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tt_v[starty:endy] * visc / ustar**4.), 'b-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tv_v[starty:endy] * visc / ustar**4.), 'g-',linewidth=2.0)
semilogx(yplus[starty:endy], (eps_v[starty:endy] * visc / ustar**4.), 'k-',linewidth=2.0)

semilogx(yplusMoser, P_vmoser, 'mo', label=r"$P$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, S_vmoser, 'ro', label=r"$S$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tt_vmoser, 'bo', label=r"$T_t$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tv_vmoser, 'go', label=r"$T_v$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, eps_vmoser, 'ko', label=r"$\epsilon$",fillstyle = 'none',linewidth=2.0)

xlabel(r'$z \ [-]$', fontsize = 20)
ylabel(r"$\partial \ \overline{v'v'} \ / \partial t \ [-]$",fontsize=20)
legend(loc=4, frameon=False,fontsize=16)
grid()
axis([0.3, 600, -0.15, 0.15])
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
tight_layout()
savefig("moser600budget_vvar.png")
close()
#
figure()
semilogx(yplush[starty+1:endy], (P_w[starty+1:endy] * visc / ustar**4.), 'm-',linewidth=2.0)
semilogx(yplush[starty+1:endy], (Tp_w[starty+1:endy] * visc / ustar**4.), 'y-',linewidth=2.0)
semilogx(yplush[starty+1:endy], (Tt_w[starty+1:endy] * visc / ustar**4.), 'b-',linewidth=2.0)
semilogx(yplush[starty+1:endy], (Tv_w[starty+1:endy] * visc / ustar**4.), 'g-',linewidth=2.0)
semilogx(yplush[starty+1:endy], (eps_w[starty+1:endy] * visc / ustar**4.), 'k-',linewidth=2.0)

semilogx(yplusMoser, P_wmoser, 'mo', label=r"$P$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tp_wmoser, 'yo', label=r"$T_p$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tt_wmoser, 'bo', label=r"$T_t$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tv_wmoser, 'go', label=r"$T_v$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, eps_wmoser, 'ko', label=r"$\epsilon$",fillstyle = 'none',linewidth=2.0)

xlabel(r'$z \ [-]$', fontsize = 20)
ylabel(r"$\partial \ \overline{w'w'} \ / \partial t \ [-]$",fontsize=20)
legend(loc=4, frameon=False,fontsize=16)
grid()
axis([0.3, 600, -0.06, 0.05])
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
tight_layout()
savefig("moser600budget_wvar.png")
close()
#
figure()
semilogx(yplus[starty:endy], (Tp_k[starty:endy] * visc / ustar**4.), 'y-',linewidth=2.0)
semilogx(yplus[starty:endy], (S_k[starty:endy] * visc / ustar**4.), 'r-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tt_k[starty:endy] * visc / ustar**4.), 'b-',linewidth=2.0)
semilogx(yplus[starty:endy], (Tv_k[starty:endy] * visc / ustar**4.), 'g-',linewidth=2.0)
semilogx(yplus[starty:endy], (eps_k[starty:endy] * visc / ustar**4.), 'k-',linewidth=2.0)

semilogx(yplusMoser, Tp_kmoser, 'yo', label=r"$T_p$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, S_kmoser, 'ro', label=r"$S$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tt_kmoser, 'bo', label=r"$T_t$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, Tv_kmoser, 'go', label=r"$T_v$",fillstyle = 'none',linewidth=2.0)
semilogx(yplusMoser, eps_kmoser, 'ko', label=r"$\epsilon$",fillstyle = 'none',linewidth=2.0)

xlabel(r'$z \ [-]$', fontsize = 20)
ylabel(r"$\partial \ \overline{TKE} \ / \partial t \ [-]$",fontsize=20)
legend(loc=4, frameon=False,fontsize=16)
grid()
axis([0.3, 600, -0.35, 0.3])
xticks(fontsize = 16, rotation = 90)
yticks(fontsize = 16, rotation = 0)
tight_layout()
savefig("moser600budget_kvar.png")
close()
