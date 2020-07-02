import netCDF4 as nc
import numpy as np
#from scipy.interpolate import RectBivariateSpline as rbs
import matplotlib as mpl
mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}", r"\usepackage[utf8]{inputenc}"]#,r"\usepackage{dvipng}"]
import matplotlib.pyplot as plt

training_data = nc.Dataset('/projects/1/flowsim/simulation1/lesscoarse_prefsampling/training_data.nc','r')
highres_data = nc.Dataset('/projects/1/flowsim/simulation1/u.nc','r')
highres_coord_data = nc.Dataset('/projects/1/flowsim/simulation1/v.nc','r')
#delta = 1500 #Half-channel width in [m], 500 m is representative for a realistic ABL
delta = 1 #NOTE: half-channel width for Moser case
#utau_ref = 0.2 #Friction velocity in [m/s], 0.2 m/s is representative for a realistic ABL. This is applied to undo the normalisation and rescale the values to realistic numbers.
utau_ref_moser = training_data['utau_ref'][:] #NOTE: friction velocity for Moser case, needed to normalize high-res velocity fields. The low-resolution fields were already normalized.
#utau_ref_moser = 0.0059

#Define indices to select only part domain, les and dns indices chosen such that they cover the same region of the domain
iend_selecles = 24;
jend_selecles = 12;
iend_selecdns = 192;
jend_selecdns = 96;

u = (highres_data.variables['u'][0,38,:jend_selecdns,:iend_selecdns] / utau_ref_moser) #* utau_ref
xh = highres_data.variables['xh'][:iend_selecdns]*delta
y = highres_data.variables['y'][:jend_selecdns]*delta
z = highres_data.variables['z'][:]*delta
x = highres_coord_data.variables['x'][:iend_selecdns]*delta
yh = highres_coord_data.variables['yh'][:jend_selecdns]*delta
#yh = np.insert(yh,0,0)
x = np.insert(x,0,x[0]-xh[1])
yh = np.append(yh, highres_coord_data.variables['yh'][jend_selecdns]*delta)
#yh = np.append(yh,np.pi*delta) #NOTE: select this one when jend_selecdns is NOT used!
print(z[38])

uc = training_data.variables['uc'][27,training_data.variables['kgc_center'][:]:training_data.variables['kend'][:],training_data.variables['jgc'][:]:training_data.variables['jend'][:],training_data.variables['igc'][:]:training_data.variables['ihend'][:]]
uc = uc[2,:jend_selecles,:iend_selecles]# * utau_ref
#xhc =  training_data.variables['xhc'][training_data.variables['igc'][:]:training_data.variables['ihend'][:]]*delta
xc = training_data.variables['xgc'][training_data.variables['igc'][:]-1:training_data.variables['igc'][:]+iend_selecles]*delta
#yhc = training_data.variables['yghc'][training_data.variables['jgc'][:]:training_data.variables['jhend'][:]]*delta
#yc = training_data.variables['ygc'][training_data.variables['jgc'][:]:training_data.variables['jend'][:]+1]*delta
zgc = training_data.variables['zgc'][training_data.variables['kgc_center'][:]:training_data.variables['kend'][:]]*delta
xhc =  training_data.variables['xhgc'][training_data.variables['igc'][:]-1:training_data.variables['igc'][:]+iend_selecles]*delta #-1 needed to select edge of chosen region in domain
#xc  = training_data.variables['xc'][:]*delta
yhc = training_data.variables['yhc'][:jend_selecles+1]*delta
yc  = training_data.variables['yc'][:jend_selecles]*delta
zc  = training_data.variables['zc'][:]*delta
#yhc = np.append(yhc,np.pi*delta)
print(zgc[2])

#NOTE: only turbulence contribution considered, not the total transport as the viscous contribution should not be added.
total_tau_xu = training_data.variables['total_tau_xu_turb'][27,2,:jend_selecles,:iend_selecles] + training_data.variables['total_tau_xu_visc'][27,2,:jend_selecles,:iend_selecles]
res_tau_xu = training_data.variables['res_tau_xu_turb'][27,2,:jend_selecles,:iend_selecles] + training_data.variables['res_tau_xu_visc'][27,2,:jend_selecles,:iend_selecles]
unres_tau_xu = training_data.variables['unres_tau_xu_tot'][27,2,:jend_selecles,:iend_selecles] #* (utau_ref ** 2)
#print(total_tau_xu.shape)
#print(xhc.shape)
#print(yc.shape)

lines = []
#itot = int(x[-1] / 10)
#jtot = int(y[-1] / 10)
for i in xhc:
    lx = np.array([i, i])
    ly = np.array([0, yhc[-1]])
    lines.append((lx, ly))
for j in yhc:
    lx = np.array([0, xc[-1]])
    ly = np.array([j, j])
    lines.append((lx, ly))

plt.figure()
#plt.subplot(121)
plt.pcolormesh(x, yh, u, vmin=5., vmax=20.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('[-]',rotation=270,fontsize=20,labelpad=30)
#cbar.set_label(r'$\rm {[m\ s^{-1}}]$',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=1.0)
plt.xlabel(r'$x/\delta [-]$',fontsize=20)
plt.ylabel(r'$y/\delta [-]$',fontsize=20)
plt.xlim(0, x[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'a) $u$', loc='center', fontsize=20, y=1.08)
#plt.title(r'a) high-resolution flow field u', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_high_res_u.png',bbox_inches='tight')


#plt.subplot(122)
plt.figure()
plt.pcolormesh(xc, yhc, uc, vmin=5., vmax=20.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('[-]',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel(r'$x/\delta [-]$',fontsize=20)
plt.ylabel(r'$y/\delta [-]$',fontsize=20)
plt.xlim(0, xc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'b) $\overline{u}$', loc='center', fontsize=20, y=1.08)
#plt.title(r'b) coarse-grained flow field', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_low_res_u.png',bbox_inches='tight')

plt.figure()
plt.pcolormesh(xhc, yhc, total_tau_xu, vmin=100., vmax=400.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('[-]',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel(r'$x/\delta [-]$',fontsize=20)
plt.ylabel(r'$y/\delta [-]$',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r"c) $\frac{1}{\Delta x \Delta y \Delta z} \int_{\partial\Omega_{u \mathbf{k}}^{in}} (u u - \nu \frac{\partial u}{\partial x}) \mathrm{d}\mathbf{x'}$", loc='center', fontsize=20, y=1.08)
#plt.title(r'c) Total turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_total_transport.png',bbox_inches='tight')

plt.figure()
plt.pcolormesh(xhc, yhc, res_tau_xu, vmin=100., vmax=400.)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('[-]',rotation=270,fontsize=20,labelpad=30)
#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel(r'$x/\delta [-]$',fontsize=20)
plt.ylabel(r'$y/\delta [-]$',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'd) $\frac{\overline{u}^{\mathbf{k}} + \overline{u}^{\mathbf{k} - \mathbf{e_u}}}{2}$' " " r'$\frac{\overline{u}^{\mathbf{k}} + \overline{u}^{\mathbf{k} - \mathbf{e_u }}}{2} - \nu \frac{\overline{u}^{\mathbf{k}} - \overline{u}^{\mathbf{k} - \mathbf{e_u} }}{\Delta x}$', loc='center', fontsize=20, y=1.08)
#plt.title(r'd) Resolved turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_res_transport.png',bbox_inches='tight')

plt.figure()
plt.pcolormesh(xhc, yhc, unres_tau_xu, vmin=-12.5, vmax=12.5)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('[-]',rotation=270,fontsize=20,labelpad=30)

#for l in lines:
#    plt.plot(*l, '#dddddd', linewidth=2.0)
plt.xlabel(r'$x/\delta [-]$',fontsize=20)
plt.ylabel(r'$y/\delta [-]$',fontsize=20)
plt.xlim(0, xhc[-1])
#plt.ylim(0, 36)
plt.xticks(fontsize = 16, rotation = 0)
plt.yticks(fontsize = 16, rotation = 0)
plt.title(r'e) $\tau_{uu}^{\mathbf{k}, in}$', loc='center', fontsize=20, y=1.08)
#plt.title(r'e) Unresolved/subgrid turbulent transport', loc='center', fontsize=20)
plt.tight_layout()
plt.savefig('method_unres_transport.png',bbox_inches='tight')

training_data.close()
