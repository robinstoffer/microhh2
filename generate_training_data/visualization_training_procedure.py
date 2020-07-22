#Visualize panels in 4-step filtering procedure, final figure in paper made with powerpoint
#NOTE: requires pre-processing: DNS binary snapshot of u,v at time step 1200s have to be converted to a netCDF file
import netCDF4 as nc
import numpy as np
#from scipy.interpolate import RectBivariateSpline as rbs
import matplotlib as mpl
mpl.use('PDF')
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}", r"\usepackage[utf8]{inputenc}"] #r"\usepackage{dvipng}"]
import matplotlib.pyplot as plt
import subprocess

#Select time step used for visualization
t_dns = 1200 #Refers to time step of corresponding DNS binary fields
t_training = 0 #Refers to time step of corresponding training file
#NOTE: ensure that t_dns and t_training refer to the same time step

#Perform pre-processing needed for visualizing the training data generation procedure: for one of the training snapshots (here chosen to be the one at 1200s simulation time), convert the binary high-resolution DNS fields to netCDF-files
subprocess.run(["python3", "../python/3d_to_nc.py", "--directory", "../cases/moser600" ,"--filename", "moser600.ini", "--vars", "u", "--precision", "double", "-t0", str(t_dns), "-t1", str(t_dns), "-tstep", "1"])
subprocess.run(["python3", "../python/3d_to_nc.py", "--directory", "../cases/moser600" ,"--filename", "moser600.ini", "--vars", "v", "--precision", "double", "-t0", str(t_dns), "-t1", str(t_dns), "-tstep", "1"])


#Load datasets
training_data = nc.Dataset('../cases/moser600/training_data/training_data.nc','r')
highres_data = nc.Dataset('../cases/moser600/u.nc','r')
highres_coord_data = nc.Dataset('../cases/moser600/v.nc','r') #Only some of the coordinates stored in this file are needed

#Define settings for normalization, currently reflect chosen Moser case
delta = 1
reynolds_tau_moser = 590
mvisc_ref_moser = training_data['mvisc'][:]
utau_ref_moser = (reynolds_tau_moser * mvisc_ref_moser) / delta

#Define indices to select height and only part horizontal domain, les and dns indices chosen such that they cover the same region of the domain
k_selecles = 3
iend_selecles = 24
jend_selecles = 12
k_selecdns = 46
iend_selecdns = 192
jend_selecdns = 96

#Load high-resolution DNS velocity and coordinates, adjust x and yh for plotting
u = highres_data.variables['u'][0,k_selecdns,:jend_selecdns,:iend_selecdns] / utau_ref_moser #NOTE: assumes that high-res u.nc file only contains the desired time step, not other ones (and therefore simply is always the first one). This is valid for the current settings used in the pre-processing.
x = highres_coord_data.variables['x'][:iend_selecdns]*delta
xh = highres_data.variables['xh'][:iend_selecdns]*delta
x = np.insert(x,0,x[0]-xh[1]) 
y = highres_data.variables['y'][:jend_selecdns]*delta
yh = highres_coord_data.variables['yh'][:jend_selecdns+1]*delta
z = highres_data.variables['z'][:]*delta
print(z[k_selecdns])

#load LES velocity and coordinates, at some coordinates additional grid cells are selected for the plot
uc = training_data.variables['uc'][t_training,training_data.variables['kgc_center'][:]:training_data.variables['kend'][:],training_data.variables['jgc'][:]:training_data.variables['jend'][:],training_data.variables['igc'][:]:training_data.variables['ihend'][:]] / utau_ref_moser
uc = uc[k_selecles,:jend_selecles,:iend_selecles]
xc = training_data.variables['xgc'][training_data.variables['igc'][:]-1:training_data.variables['igc'][:]+iend_selecles]*delta #Ghost cell used to capture boundary chosen domain
xhc =  training_data.variables['xhc'][:iend_selecles+1]*delta
yc  = training_data.variables['yc'][:jend_selecles]*delta
yhc = training_data.variables['yhc'][:jend_selecles+1]*delta
zc  = training_data.variables['zc'][:]*delta
print(zc[k_selecles])

#Extract contributions from both the turbulent advection term and viscous stress term
total_tau_xu = (training_data.variables['total_tau_xu_turb'][t_training,k_selecles,:jend_selecles,:iend_selecles] + training_data.variables['total_tau_xu_visc'][t_training,k_selecles,:jend_selecles,:iend_selecles]) / (utau_ref_moser ** 2)
res_tau_xu = (training_data.variables['res_tau_xu_turb'][t_training,k_selecles,:jend_selecles,:iend_selecles] + training_data.variables['res_tau_xu_visc'][t_training,k_selecles,:jend_selecles,:iend_selecles]) / (utau_ref_moser ** 2)
unres_tau_xu = training_data.variables['unres_tau_xu_tot'][t_training,k_selecles,:jend_selecles,:iend_selecles] / (utau_ref_moser ** 2)

#Plot individual panels, powerpoint was used to create from these panels to eventual figure
plt.figure()
plt.pcolormesh(x, yh, u, vmin=5., vmax=20.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[-]$',rotation=270,fontsize=20,labelpad=30)
plt.xlabel(r'$x \delta^{-1} [-]$',fontsize=20)
plt.ylabel(r'$y \delta^{-1} [-]$',fontsize=20)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'a) $u$', loc='center', fontsize=20, y=1.08)
fig = plt.gcf()
fig.set_tight_layout(True) #set_tight_layout method required to prevent that the backend falls back to Agg
fig.savefig('method_high_res_u.pdf',bbox_inches='tight')
#
plt.figure()
plt.pcolormesh(xc, yhc, uc, vmin=5., vmax=20.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[-]$',rotation=270,fontsize=20,labelpad=30)
plt.xlabel(r'$x \delta^{-1} [-]$',fontsize=20)
plt.ylabel(r'$y \delta^{-1} [-]$',fontsize=20)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'b) $\overline{u}$', loc='center', fontsize=20, y=1.08)
fig = plt.gcf()
fig.set_tight_layout(True) #set_tight_layout method required to prevent that the backend falls back to Agg
fig.savefig('method_low_res_u.pdf',bbox_inches='tight')
#
plt.figure()
plt.pcolormesh(xhc, yhc, total_tau_xu, vmin=100., vmax=400.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[-]$',rotation=270,fontsize=20,labelpad=30)
plt.xlabel(r'$x \delta^{-1} [-]$',fontsize=20)
plt.ylabel(r'$y \delta^{-1} [-]$',fontsize=20)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r"c) $\frac{1}{\Delta y \Delta z} \int_{\partial\Omega_{u}^{in}} (u u - \nu \frac{\partial u}{\partial x}) \,\ \mathrm{d}y'\mathrm{d}z'$", loc='center', fontsize=20, y=1.08)
fig = plt.gcf()
fig.set_tight_layout(True) #set_tight_layout method required to prevent that the backend falls back to Agg
fig.savefig('method_total_transport.pdf',bbox_inches='tight')
#
plt.figure()
plt.pcolormesh(xhc, yhc, res_tau_xu, vmin=100., vmax=400.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[-]$',rotation=270,fontsize=20,labelpad=30)
plt.xlabel(r'$x \delta^{-1} [-]$',fontsize=20)
plt.ylabel(r'$y \delta^{-1} [-]$',fontsize=20)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r"d) $(\frac{\overline{u}(l-1,m,n) + \overline{u}(l,m,n)}{2})^{2} - \nu \frac{\overline{u}(l,m,n) - \overline{u}(l-1,m,n)}{\Delta x}$", loc='center', fontsize=20, y=1.08)
fig = plt.gcf()
fig.set_tight_layout(True) #set_tight_layout method required to prevent that the backend falls back to Agg
fig.savefig('method_res_transport.pdf',bbox_inches='tight')
#
plt.figure()
plt.pcolormesh(xhc, yhc, unres_tau_xu, vmin=-12.5, vmax=12.5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$[-]$',rotation=270,fontsize=20,labelpad=30)
plt.xlabel(r'$x \delta^{-1} [-]$',fontsize=20)
plt.ylabel(r'$y \delta^{-1} [-]$',fontsize=20)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'e) $\tau_{uu}^{in}$', loc='center', fontsize=20, y=1.08)
fig = plt.gcf()
fig.set_tight_layout(True) #set_tight_layout method required to prevent that the backend falls back to Agg
fig.savefig('method_unres_transport.pdf',bbox_inches='tight')

#Close netCDF-files
training_data.close()
highres_data.close()
highres_coord_data.close()
