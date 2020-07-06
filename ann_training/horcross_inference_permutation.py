import numpy as np
import netCDF4 as nc
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--permute_file', default=None, \
        help='NetCDF file that contains the feature importances for the features')
parser.add_argument('--only_zu', dest='only_zu', default=None, \
        action='store_true', \
        help='Flag to specify that only the zu-component is considered, such that some features can be left out that are not used.')
parser.add_argument('--upstream', dest='upstream', default=None, \
        action='store_true', \
        help='Flag to specify that upstream component is considered.')
parser.add_argument('--downstream', dest='downstream', default=None, \
        action='store_true', \
        help='Flag to specify that downstream component is considered.')
args = parser.parse_args()

#Check whether either upstream or downstream is specified (should be the case)
flag_upstream = False
if args.upstream:
    flag_upstream = True
flag_downstream = False
if args.downstream:
    flag_downstream = True
if flag_upstream and flag_downstream:
    raise RuntimeError("Please specify with single flag whether upstream or downstream component is considered.")
if (not flag_upstream) and (not flag_downstream):
    raise RuntimeError("Please specify with single flag whether upstream or downstream component is considered.")

#Fetch feature importances
a = nc.Dataset(args.permute_file,'r')
u_fi = np.array(a['u_fi'][:,:,:,:])
v_fi = np.array(a['v_fi'][:,:,:,:])
w_fi = np.array(a['w_fi'][:,:,:,:])

#Define boundaries for grid cell indices
i_ind  = np.arange(-0.5,5.5)
iv_ind = np.arange(-0.5,5.5)
iw_ind = np.arange(-0.5,5.5)
j_ind  = np.arange(-0.5,5.5)
jv_ind = np.arange(-0.5,5.5)
k_ind  = np.arange(0,5)
kw_ind = np.arange(0,5)

i_ticks  = np.arange(0,5)
iv_ticks = np.arange(0,5)
iw_ticks = np.arange(0,5)
j_ticks  = np.arange(0,5)
jv_ticks = np.arange(0,5)
k_ticks  = np.arange(0,5)
kw_ticks = np.arange(0,5)

#Remove not used features when only zu-component is considered
if args.only_zu:
    v_fi = v_fi[:,:,1:,:-1]
    iv_ind = np.arange(-0.5,4.5)
    jv_ind = np.arange(0.5,5.5)
    w_fi = w_fi[:,1:,:,:-1]
    iw_ind = np.arange(-0.5,4.5)
    kw_ind = np.arange(1,5)
    iv_ticks = np.arange(0,4)
    jv_ticks = np.arange(1,5)
    iw_ticks = np.arange(0,4)
    kw_ticks = np.arange(1,5)

#Take time average
u_fi = np.mean(u_fi, axis=0)
v_fi = np.mean(v_fi, axis=0)
w_fi = np.mean(w_fi, axis=0)

#Make horizontal cross-sections:
#u-velocity
for k in k_ind:
    u_fi_height = u_fi[k,:,:]

    plt.figure()
    plt.pcolormesh(i_ind,j_ind,u_fi_height, vmin=1.0, vmax=2.5, cmap=plt.get_cmap('Blues'))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1i'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1i'))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Permutation importance [-]',rotation=270,fontsize=20,labelpad=30)
    plt.xlabel('i', fontsize=16)
    plt.ylabel('j', fontsize=16)
    plt.xticks(i_ticks, fontsize=16, rotation=0)
    plt.yticks(j_ticks, fontsize=16, rotation=0)
    plt.tight_layout()
    if flag_upstream:
        plt.savefig("u_fi_upstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    elif flag_downstream:
        plt.savefig("u_fi_downstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    plt.close()

#v-velocity
for k in k_ind:
    v_fi_height = v_fi[k,:,:]

    plt.figure()
    plt.pcolormesh(iv_ind,jv_ind,v_fi_height, vmin=1.0, vmax=1.5, cmap=plt.get_cmap('Blues'))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1i'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1i'))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Permutation importance [-]',rotation=270,fontsize=20,labelpad=30)
    plt.xlabel('i', fontsize=16)
    plt.ylabel('j', fontsize=16)
    plt.xticks(iv_ticks, fontsize=16, rotation=0)
    plt.yticks(jv_ticks, fontsize=16, rotation=0)
    plt.tight_layout()
    if flag_upstream:
        plt.savefig("v_fi_upstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    elif flag_downstream:
        plt.savefig("v_fi_downstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    plt.close()

#w-velocity
for k in kw_ind:
    k_ind = k
    if args.only_zu:
        k_ind = k-1
    w_fi_height = w_fi[k_ind,:,:]

    plt.figure()
    plt.pcolormesh(iw_ind,j_ind, w_fi_height, vmin=1.0, vmax=2.2, cmap=plt.get_cmap('Blues'))

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1i'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1i'))
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Permutation importance [-]',rotation=270,fontsize=20,labelpad=30)
    plt.xlabel('i', fontsize=16)
    plt.ylabel('j', fontsize=16)
    plt.xticks(iw_ticks, fontsize=16, rotation=0)
    plt.yticks(j_ticks, fontsize=16, rotation=0)
    plt.tight_layout()
    if flag_upstream:
        plt.savefig("w_fi_upstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    elif flag_downstream:
        plt.savefig("w_fi_downstream_horcross_heightindex" + str(k)+ ".png", dpi=200)
    plt.close()
