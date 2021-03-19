import numpy as np
import pandas as pd
import netCDF4 as nc
#import tensorflow as tf
import matplotlib as mpl
import copy
mpl.use('PDF')
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}", r"\usepackage[utf8]{inputenc}"]
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import argparse
import csv

from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--prediction_file', default=None, \
        help='NetCDF file that contains the predictions')
parser.add_argument('--smagorinsky_file', default=None, \
        help='NetCDF file that contains the calculated sub-grid scale transports according to the Smagorinsky-Lilly subgrid model')
parser.add_argument('--training_file', default=None, \
        help='NetCDF file that contains the training data, including the actual unresolved transports.')
parser.add_argument('--make_plots', dest='make_plots', default=None, \
        action='store_true', \
        help='Make plots at each height for the predictions of the MLP, Smagorinsky, and the training data')
parser.add_argument('--make_table', dest='make_table', default=None, \
        action='store_true', \
        help='Make table with all correlation coefficients between the predictions of the MLP, Smagorinsky, and the training data. Note: if --make_table flag is specified, netCDF files called "reconstructed_fields.nc" and "dissipation.nc" should be present. These are created with the --reconstruct_fields and --calc_dissipation flags.')
parser.add_argument('--reconstruct_fields', dest='reconstruct_fields', default=None, \
        action='store_true', \
        help="Reconstruct the corresponding transport fields for the predictions of the MLP. If not specified, a netCDF file called 'reconstructed_fields.nc' should be present in the current directory.")
parser.add_argument('--calc_dissipation', dest='calc_dissipation', default=None, \
        action='store_true', \
        help="Calculate the corresponding dissipation fields for the predictions of the MLP, the Smagorinsky SGS model, and the filtered DNS fields. If specified, a netCDF file called 'reconstructed_fields.nc' should be present in the current directory. (created with --reconstruct_fields flag). Creates netCDF-file called 'dissipation.nc' in current directory.")
parser.add_argument('--plot_dissipation', dest='plot_dissipation', default=None, \
        action='store_true', \
        help="Make plots of the dissipation fields inferred from the MLP predictions, Smagorinsky SGS model, and the filtered DNS fields. If specified, a netCDF file called 'dissipation.nc' should be present in the current directory. (created with --calc_dissipation flag). Note: if --make_table flag is specified, a correlation table is made of the dissipation.")
args = parser.parse_args()

###Fetch Smagorinsky fluxes, training fluxes, MLP predictions, and heights. Next, calculate isotropic part subgrid-scale stress and subtract it.###
a = nc.Dataset(args.prediction_file,'r')
b = nc.Dataset(args.smagorinsky_file,'r')
c = nc.Dataset(args.training_file,'r')

#Define settings for selected channel flow Moser case, and subsequently calculate representative friction velocity
delta = 1.
mvisc_ref_moser = np.array(c['mvisc'][:])
reynolds_tau_moser = 590
utau_ref_moser = (reynolds_tau_moser * mvisc_ref_moser) / delta

#Specify time steps reserved for a priori testing.
tstart = 28
tend   = 31
nt = tend - tstart

#Extract smagorinsky fluxes, training fluxes (including resolved and total fluxes), MLP fluxes.
#NOTE1:remove previously added ghost cells in staggered directions
#NOTE2: in staggered directions, ghost cells added previously are removed, except in the z-direction for the zu and zv components
#NOTE3: In the isotropic directions the arrays in the training file contain one additional ghost cell, which we remove here.
#NOTE4: on purpose both the ANN and smag predictions are compared to the fluxes resulting from the advection AND the viscous term. Both these terms have unresolved contributions in our FV-method. It is also to demonstrate that the correlation of smag is (very) low in that case, as smag does not take the visouc term and made numerical errors explicitly into account.
#NOTE5: for the smag and ANN predictions, we do not subtract the deviatoric part for the istropic components (although common practice when smag is used, it is not necessary for our ANN SGS model. Not subtracting the deviatoric part with the Smagorinsky preds thus keeps it consistent with our ANN SGS model.)
smag_tau_xu  = np.array(b['smag_tau_xu'][:nt,:,:,:])
smag_tau_yu  = np.array(b['smag_tau_yu'][:nt,:,:-1,:-1])
smag_tau_zu  = np.array(b['smag_tau_zu'][:nt,:,:,:-1])
smag_tau_xv  = np.array(b['smag_tau_xv'][:nt,:,:-1,:-1])
smag_tau_yv  = np.array(b['smag_tau_yv'][:nt,:,:,:])
smag_tau_zv  = np.array(b['smag_tau_zv'][:nt,:,:-1,:])
smag_tau_xw  = np.array(b['smag_tau_xw'][:nt,:-1,:,:-1])
smag_tau_yw  = np.array(b['smag_tau_yw'][:nt,:-1,:-1,:])
smag_tau_zw  = np.array(b['smag_tau_zw'][:nt,:,:,:])
#
unres_tau_xu = np.array(c['unres_tau_xu_tot'][tstart:tend,:,:,1:])
unres_tau_yu = np.array(c['unres_tau_yu_tot'][tstart:tend,:,:-1,:-1])
unres_tau_zu = np.array(c['unres_tau_zu_tot'][tstart:tend,:,:,:-1])
unres_tau_xv = np.array(c['unres_tau_xv_tot'][tstart:tend,:,:-1,:-1])
unres_tau_yv = np.array(c['unres_tau_yv_tot'][tstart:tend,:,1:,:])
unres_tau_zv = np.array(c['unres_tau_zv_tot'][tstart:tend,:,:-1,:])
unres_tau_xw = np.array(c['unres_tau_xw_tot'][tstart:tend,:-1,:,:-1])
unres_tau_yw = np.array(c['unres_tau_yw_tot'][tstart:tend,:-1,:-1,:])
unres_tau_zw = np.array(c['unres_tau_zw_tot'][tstart:tend,1:,:,:])
#res_tau_xu   = np.array(c['res_tau_xu_turb']  [tstart:tend,:,:,1:]) + np.array(c['res_tau_xu_visc']   [tstart:tend,:,:,1:])
#res_tau_yu   = np.array(c['res_tau_yu_turb']  [tstart:tend,:,:-1,:-1]) + np.array(c['res_tau_yu_visc']   [tstart:tend,:,:-1,:-1]) 
#res_tau_zu   = np.array(c['res_tau_zu_turb']  [tstart:tend,:,:,:-1]) + np.array(c['res_tau_zu_visc']   [tstart:tend,:,:,:-1]) 
#res_tau_xv   = np.array(c['res_tau_xv_turb']  [tstart:tend,:,:-1,:-1]) + np.array(c['res_tau_xv_visc']   [tstart:tend,:,:-1,:-1]) 
#res_tau_yv   = np.array(c['res_tau_yv_turb']  [tstart:tend,:,1:,:]) + np.array(c['res_tau_yv_visc']   [tstart:tend,:,1:,:])
#res_tau_zv   = np.array(c['res_tau_zv_turb']  [tstart:tend,:,:-1,:]) + np.array(c['res_tau_zv_visc']   [tstart:tend,:,:-1,:]) 
#res_tau_xw   = np.array(c['res_tau_xw_turb']  [tstart:tend,:-1,:,:-1]) + np.array(c['res_tau_xw_visc']   [tstart:tend,:-1,:,:-1]) 
#res_tau_yw   = np.array(c['res_tau_yw_turb']  [tstart:tend,:-1,:-1,:]) + np.array(c['res_tau_yw_visc']   [tstart:tend,:-1,:-1,:]) 
#res_tau_zw   = np.array(c['res_tau_zw_turb']  [tstart:tend,1:,:,:]) + np.array(c['res_tau_zw_visc']   [tstart:tend,1:,:,:])
#tot_tau_xu   = np.array(c['total_tau_xu_turb'][tstart:tend,:,:,1:]) + np.array(c['total_tau_xu_visc'] [tstart:tend,:,:,1:])
#tot_tau_yu   = np.array(c['total_tau_yu_turb'][tstart:tend,:,:-1,:-1]) + np.array(c['total_tau_yu_visc'] [tstart:tend,:,:-1,:-1]) 
#tot_tau_zu   = np.array(c['total_tau_zu_turb'][tstart:tend,:,:,:-1]) + np.array(c['total_tau_zu_visc'] [tstart:tend,:,:,:-1])
#tot_tau_xv   = np.array(c['total_tau_xv_turb'][tstart:tend,:,:-1,:-1]) + np.array(c['total_tau_xv_visc'] [tstart:tend,:,:-1,:-1])
#tot_tau_yv   = np.array(c['total_tau_yv_turb'][tstart:tend,:,1:,:]) + np.array(c['total_tau_yv_visc'] [tstart:tend,:,1:,:])
#tot_tau_zv   = np.array(c['total_tau_zv_turb'][tstart:tend,:,:-1,:]) + np.array(c['total_tau_zv_visc'] [tstart:tend,:,:-1,:])
#tot_tau_xw   = np.array(c['total_tau_xw_turb'][tstart:tend,:-1,:,:-1]) + np.array(c['total_tau_xw_visc'] [tstart:tend,:-1,:,:-1])
#tot_tau_yw   = np.array(c['total_tau_yw_turb'][tstart:tend,:-1,:-1,:]) + np.array(c['total_tau_yw_visc'] [tstart:tend,:-1,:-1,:])
#tot_tau_zw   = np.array(c['total_tau_zw_turb'][tstart:tend,1:,:,:]) + np.array(c['total_tau_zw_visc'] [tstart:tend,1:,:,:])
##
#res_tau_xu_visc = np.array(c['res_tau_xu_visc'][tstart:tend,:,:,1:])
#res_tau_yu_visc = np.array(c['res_tau_yu_visc'][tstart:tend,:,:-1,:-1])
#res_tau_zu_visc = np.array(c['res_tau_zu_visc'][tstart:tend,:,:,:-1])
#res_tau_xv_visc = np.array(c['res_tau_xv_visc'][tstart:tend,:,:-1,:-1])
#res_tau_yv_visc = np.array(c['res_tau_yv_visc'][tstart:tend,:,1:,:])
#res_tau_zv_visc = np.array(c['res_tau_zv_visc'][tstart:tend,:,:-1,:])
#res_tau_xw_visc = np.array(c['res_tau_xw_visc'][tstart:tend,:-1,:,:-1])
#res_tau_yw_visc = np.array(c['res_tau_yw_visc'][tstart:tend,:-1,:-1,:])
#res_tau_zw_visc = np.array(c['res_tau_zw_visc'][tstart:tend,1:,:,:])
#
if args.reconstruct_fields:
    preds_values_xu_upstream   = np.array(a['preds_values_tau_xu_upstream'][:])   
    lbls_values_xu_upstream    = np.array(a['lbls_values_tau_xu_upstream'][:])    
    preds_values_xu_downstream = np.array(a['preds_values_tau_xu_downstream'][:]) 
    lbls_values_xu_downstream  = np.array(a['lbls_values_tau_xu_downstream'][:])  
    preds_values_yu_upstream   = np.array(a['preds_values_tau_yu_upstream'][:])   
    lbls_values_yu_upstream    = np.array(a['lbls_values_tau_yu_upstream'][:])    
    preds_values_yu_downstream = np.array(a['preds_values_tau_yu_downstream'][:]) 
    lbls_values_yu_downstream  = np.array(a['lbls_values_tau_yu_downstream'][:])  
    preds_values_zu_upstream   = np.array(a['preds_values_tau_zu_upstream'][:])   
    lbls_values_zu_upstream    = np.array(a['lbls_values_tau_zu_upstream'][:])    
    preds_values_zu_downstream = np.array(a['preds_values_tau_zu_downstream'][:]) 
    lbls_values_zu_downstream  = np.array(a['lbls_values_tau_zu_downstream'][:])  
    preds_values_xv_upstream   = np.array(a['preds_values_tau_xv_upstream'][:])   
    lbls_values_xv_upstream    = np.array(a['lbls_values_tau_xv_upstream'][:])    
    preds_values_xv_downstream = np.array(a['preds_values_tau_xv_downstream'][:]) 
    lbls_values_xv_downstream  = np.array(a['lbls_values_tau_xv_downstream'][:])  
    preds_values_yv_upstream   = np.array(a['preds_values_tau_yv_upstream'][:])   
    lbls_values_yv_upstream    = np.array(a['lbls_values_tau_yv_upstream'][:])    
    preds_values_yv_downstream = np.array(a['preds_values_tau_yv_downstream'][:]) 
    lbls_values_yv_downstream  = np.array(a['lbls_values_tau_yv_downstream'][:])  
    preds_values_zv_upstream   = np.array(a['preds_values_tau_zv_upstream'][:])   
    lbls_values_zv_upstream    = np.array(a['lbls_values_tau_zv_upstream'][:])    
    preds_values_zv_downstream = np.array(a['preds_values_tau_zv_downstream'][:]) 
    lbls_values_zv_downstream  = np.array(a['lbls_values_tau_zv_downstream'][:])  
    preds_values_xw_upstream   = np.array(a['preds_values_tau_xw_upstream'][:])   
    lbls_values_xw_upstream    = np.array(a['lbls_values_tau_xw_upstream'][:])    
    preds_values_xw_downstream = np.array(a['preds_values_tau_xw_downstream'][:]) 
    lbls_values_xw_downstream  = np.array(a['lbls_values_tau_xw_downstream'][:])  
    preds_values_yw_upstream   = np.array(a['preds_values_tau_yw_upstream'][:])
    lbls_values_yw_upstream    = np.array(a['lbls_values_tau_yw_upstream'][:])
    preds_values_yw_downstream = np.array(a['preds_values_tau_yw_downstream'][:]) 
    lbls_values_yw_downstream  = np.array(a['lbls_values_tau_yw_downstream'][:])  
    preds_values_zw_upstream   = np.array(a['preds_values_tau_zw_upstream'][:])
    lbls_values_zw_upstream    = np.array(a['lbls_values_tau_zw_upstream'][:]) 
    preds_values_zw_downstream = np.array(a['preds_values_tau_zw_downstream'][:]) 
    lbls_values_zw_downstream  = np.array(a['lbls_values_tau_zw_downstream'][:])  
    zhloc_values    = np.array(a['zhloc_samples'][:])
    zloc_values     = np.array(a['zloc_samples'][:])
    yhloc_values    = np.array(a['yhloc_samples'][:])
    yloc_values     = np.array(a['yloc_samples'][:])
    xhloc_values    = np.array(a['xhloc_samples'][:])
    xloc_values     = np.array(a['xloc_samples'][:])
    tstep_values    = np.array(a['tstep_samples'][:]).astype('int') #make sure time steps are stored as integers, not as floats

#Extract coordinates
nt, _, _, _ = unres_tau_xu.shape
zc  = np.array(c['zc'][:])
nz = len(zc)
zhc = np.array(c['zhc'][:])
zgcextra = np.array(c['zgcextra'][:])
yc  = np.array(c['yc'][:])
ny = len(yc)
yhc = np.array(c['yhc'][:])
ygcextra = np.array(c['ygcextra'][:])
xc  = np.array(c['xc'][:])
nx = len(xc)
xhc = np.array(c['xhc'][:])
xgcextra = np.array(c['xgcextra'][:])
zhcless = zhc[:-1]
yhcless = yhc[:-1]
xhcless = xhc[:-1]

##Calculate trace part of subgrid-stress, and subtract this from labels for comparison with Smagorinsky where deviatoric part is subtracted
#trace_train = (unres_tau_xu + unres_tau_yv + unres_tau_zw) * (1./3.)
#unres_tau_xu_traceless = unres_tau_xu - trace_train
#unres_tau_yv_traceless = unres_tau_yv - trace_train
#unres_tau_zw_traceless = unres_tau_zw - trace_train

#Close files
a.close()
b.close()
c.close()

###Reconstruct flow fields###
def reconstruct_field(preds, x, xs_unique, y, ys_unique, z, zs_unique, tstep, tsteps_unique):
    
    #Initialize empty array for storage
    preds_rec = np.empty((len(tsteps_unique), len(zs_unique), len(ys_unique), len(xs_unique)))

    #For each unique combination of x, y, tstep find the corresponding value and store it
    t = 0
    for tstep_unique in tsteps_unique:
        tstep_indices = (tstep == tstep_unique)
        preds1 = preds[tstep_indices]
        z1 = z[tstep_indices]
        y1 = y[tstep_indices]
        x1 = x[tstep_indices]
        k = 0
        for z_unique in zs_unique:
            z_indices = (z1 == z_unique)
            preds2 = preds1[z_indices]
            y2 = y1[z_indices]
            x2 = x1[z_indices]
            j = 0
            for y_unique in ys_unique:
                y_indices = (y2 == y_unique)
                preds3 = preds2[y_indices]
                x3 = x2[y_indices]
                i = 0
                for x_unique in xs_unique:
                    x_index = (x3 == x_unique)
                    if len(preds3[x_index]) > 1:
                        preds_rec[t,k,j,i] = preds3[x_index][0] #Preferential sampling in validation set: some components are predicted more often, but simply take the first one: they should all be identical (NN does not change during evalution validation set)
                    else:
                        preds_rec[t,k,j,i] = preds3[x_index]
                    i += 1
                j += 1
            k += 1
        t += 1

    return preds_rec

#Reconstruct flow fields if specified to do so
if args.reconstruct_fields:

    #Create netCDF-file to store reconstructed fields
    d = nc.Dataset('reconstructed_fields.nc','w')

    #Extract additional coordinates and time steps
    tstep_unique = np.unique(tstep_values)
    zloc_unique  = np.unique(zloc_values)
    zhloc_unique = np.unique(zhloc_values)
    yloc_unique  = np.unique(yloc_values)
    yhloc_unique = np.unique(yhloc_values)
    xloc_unique  = np.unique(xloc_values)
    xhloc_unique = np.unique(xhloc_values)


    #Create dimensions for storage in nc-file
    d.createDimension("zc", len(zc))
    d.createDimension("zhc",len(zhc))
    d.createDimension("zhcless",len(zhcless))
    d.createDimension("yc", len(yc))
    d.createDimension("yhcless",len(yhcless))
    d.createDimension("xc", len(xc))
    d.createDimension("xhcless",len(xhcless))
    d.createDimension("tstep_unique",len(tstep_unique))

    #Create variables for dimensions and store them
    var_zc           = d.createVariable("zc",           "f8", ("zc",))
    var_zhc          = d.createVariable("zhc",          "f8", ("zhc",))
    var_zhcless      = d.createVariable("zhcless",      "f8", ("zhcless",))
    var_yc           = d.createVariable("yc",           "f8", ("yc",))
    var_yhcless      = d.createVariable("yhcless",      "f8", ("yhcless",))
    var_xc           = d.createVariable("xc",           "f8", ("xc",))
    var_xhcless      = d.createVariable("xhcless",      "f8", ("xhcless",))
    var_tstep_unique = d.createVariable("tstep_unique", "f8", ("tstep_unique",))

    var_zc[:]            = zc
    var_zhc[:]           = zhc
    var_zhcless[:]       = zhcless
    var_yc[:]            = yc
    var_yhcless[:]       = yhcless
    var_xc[:]            = xc
    var_xhcless[:]       = xhcless
    var_tstep_unique[:]  = tstep_unique

    #Create variables for storage labels
    var_unres_tau_xu_lbls = d.createVariable("unres_tau_xu_lbls","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_lbls = d.createVariable("unres_tau_xv_lbls","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_lbls = d.createVariable("unres_tau_xw_lbls","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_lbls = d.createVariable("unres_tau_yu_lbls","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_lbls = d.createVariable("unres_tau_yv_lbls","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_lbls = d.createVariable("unres_tau_yw_lbls","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_lbls = d.createVariable("unres_tau_zu_lbls","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_lbls = d.createVariable("unres_tau_zv_lbls","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_lbls = d.createVariable("unres_tau_zw_lbls","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_lbls_upstream = d.createVariable("unres_tau_xu_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_lbls_upstream = d.createVariable("unres_tau_xv_lbls_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_lbls_upstream = d.createVariable("unres_tau_xw_lbls_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_lbls_upstream = d.createVariable("unres_tau_yu_lbls_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_lbls_upstream = d.createVariable("unres_tau_yv_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_lbls_upstream = d.createVariable("unres_tau_yw_lbls_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_lbls_upstream = d.createVariable("unres_tau_zu_lbls_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_lbls_upstream = d.createVariable("unres_tau_zv_lbls_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_lbls_upstream = d.createVariable("unres_tau_zw_lbls_upstream","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_lbls_downstream = d.createVariable("unres_tau_xu_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_lbls_downstream = d.createVariable("unres_tau_xv_lbls_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_lbls_downstream = d.createVariable("unres_tau_xw_lbls_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_lbls_downstream = d.createVariable("unres_tau_yu_lbls_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_lbls_downstream = d.createVariable("unres_tau_yv_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_lbls_downstream = d.createVariable("unres_tau_yw_lbls_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_lbls_downstream = d.createVariable("unres_tau_zu_lbls_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_lbls_downstream = d.createVariable("unres_tau_zv_lbls_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_lbls_downstream = d.createVariable("unres_tau_zw_lbls_downstream","f8",("tstep_unique","zc","yc","xc"))
    #

    #Call function to recontruct fields of labels for all nine components and both upstream/downstream components
    print('start reconstructing labels')
    unres_tau_xu_lbls_upstream = reconstruct_field(lbls_values_xu_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    unres_tau_xu_lbls_downstream = reconstruct_field(lbls_values_xu_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    unres_tau_xv_lbls_upstream = reconstruct_field(lbls_values_xv_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_xv_lbls_downstream = reconstruct_field(lbls_values_xv_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('second component done')
    unres_tau_xw_lbls_upstream = reconstruct_field(lbls_values_xw_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    unres_tau_xw_lbls_downstream = reconstruct_field(lbls_values_xw_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    print('third component done')
    unres_tau_yu_lbls_upstream = reconstruct_field(lbls_values_yu_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_yu_lbls_downstream = reconstruct_field(lbls_values_yu_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    unres_tau_yv_lbls_upstream = reconstruct_field(lbls_values_yv_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_yv_lbls_downstream = reconstruct_field(lbls_values_yv_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fifth component done')
    unres_tau_yw_lbls_upstream = reconstruct_field(lbls_values_yw_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_yw_lbls_downstream = reconstruct_field(lbls_values_yw_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    unres_tau_zu_lbls_upstream = reconstruct_field(lbls_values_zu_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_zu_lbls_downstream = reconstruct_field(lbls_values_zu_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    unres_tau_zv_lbls_upstream = reconstruct_field(lbls_values_zv_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    unres_tau_zv_lbls_downstream = reconstruct_field(lbls_values_zv_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    unres_tau_zw_lbls_upstream = reconstruct_field(lbls_values_zw_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    unres_tau_zw_lbls_downstream = reconstruct_field(lbls_values_zw_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')

    #Create variables for storage reconstructed fields of predictions
    var_unres_tau_xu_MLP = d.createVariable("unres_tau_xu_MLP","f8",("tstep_unique","zc","yc","xc"))                
    var_unres_tau_xv_MLP = d.createVariable("unres_tau_xv_MLP","f8",("tstep_unique","zc","yhcless","xhcless"))     
    var_unres_tau_xw_MLP = d.createVariable("unres_tau_xw_MLP","f8",("tstep_unique","zhcless","yc","xhcless"))     
    var_unres_tau_yu_MLP = d.createVariable("unres_tau_yu_MLP","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_MLP = d.createVariable("unres_tau_yv_MLP","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_MLP = d.createVariable("unres_tau_yw_MLP","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_MLP = d.createVariable("unres_tau_zu_MLP","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_MLP = d.createVariable("unres_tau_zv_MLP","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_MLP = d.createVariable("unres_tau_zw_MLP","f8",("tstep_unique","zc","yc","xc")) 
    #
    var_unres_tau_xu_MLP_upstream = d.createVariable("unres_tau_xu_MLP_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_MLP_upstream = d.createVariable("unres_tau_xv_MLP_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_MLP_upstream = d.createVariable("unres_tau_xw_MLP_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_MLP_upstream = d.createVariable("unres_tau_yu_MLP_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_MLP_upstream = d.createVariable("unres_tau_yv_MLP_upstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_MLP_upstream = d.createVariable("unres_tau_yw_MLP_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_MLP_upstream = d.createVariable("unres_tau_zu_MLP_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_MLP_upstream = d.createVariable("unres_tau_zv_MLP_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_MLP_upstream = d.createVariable("unres_tau_zw_MLP_upstream","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_MLP_downstream = d.createVariable("unres_tau_xu_MLP_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_MLP_downstream = d.createVariable("unres_tau_xv_MLP_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_MLP_downstream = d.createVariable("unres_tau_xw_MLP_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_MLP_downstream = d.createVariable("unres_tau_yu_MLP_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_MLP_downstream = d.createVariable("unres_tau_yv_MLP_downstream","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_MLP_downstream = d.createVariable("unres_tau_yw_MLP_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_MLP_downstream = d.createVariable("unres_tau_zu_MLP_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_zv_MLP_downstream = d.createVariable("unres_tau_zv_MLP_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zw_MLP_downstream = d.createVariable("unres_tau_zw_MLP_downstream","f8",("tstep_unique","zc","yc","xc"))
    #

    #Call function to recontruct fields of predictions for all nine components
    print('start reconstructing predictions')
    preds_values_xu_upstream = reconstruct_field(preds_values_xu_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_xu_downstream = reconstruct_field(preds_values_xu_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('first component done')
    preds_values_xv_upstream = reconstruct_field(preds_values_xv_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_xv_downstream = reconstruct_field(preds_values_xv_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('second component done')
    preds_values_xw_upstream = reconstruct_field(preds_values_xw_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    preds_values_xw_downstream = reconstruct_field(preds_values_xw_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique)
    print('third component done')
    preds_values_yu_upstream = reconstruct_field(preds_values_yu_upstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    preds_values_yu_downstream = reconstruct_field(preds_values_yu_downstream, xhloc_values, xhloc_unique, yhloc_values, yhloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('fourth component done')
    preds_values_yv_upstream = reconstruct_field(preds_values_yv_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    preds_values_yv_downstream = reconstruct_field(preds_values_yv_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique)
    print('fifth component done')
    preds_values_yw_upstream = reconstruct_field(preds_values_yw_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_yw_downstream = reconstruct_field(preds_values_yw_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('sixth component done')
    preds_values_zu_upstream = reconstruct_field(preds_values_zu_upstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_zu_downstream = reconstruct_field(preds_values_zu_downstream, xhloc_values, xhloc_unique, yloc_values, yloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('seventh component done')
    preds_values_zv_upstream = reconstruct_field(preds_values_zv_upstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    preds_values_zv_downstream = reconstruct_field(preds_values_zv_downstream, xloc_values, xloc_unique, yhloc_values, yhloc_unique, zhloc_values, zhloc_unique, tstep_values, tstep_unique) 
    print('eighth component done')
    preds_values_zw_upstream = reconstruct_field(preds_values_zw_upstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    preds_values_zw_downstream = reconstruct_field(preds_values_zw_downstream, xloc_values, xloc_unique, yloc_values, yloc_unique, zloc_values, zloc_unique, tstep_values, tstep_unique) 
    print('nineth component done')
    
    #Combine upstream and downstream components together, where at each height the upstream and downstream components are selected alternately (as should be done during inference).
    #NOTE: zw_flag is a boolean flag for the zw turbulent transport component, because this component needs special treatment to allow for symmetric inference
    print('Start combining upstream and downstream components.')
    def _combine_upstream_downstream(upstream_field, downstream_field, sample_dir, dims, zw_flag = False, zhless_wallszero = False, isotrop_flag = False):
        
        #Choose sample indices and output shape based on specified sample_dir
        if not (sample_dir == 'x' or sample_dir == 'y' or sample_dir == 'z'):
            raise RuntimeError("The specified sampling direction should be 'x', 'y', or 'z'.")
        
        if sample_dir   == 'x':
            sample_dim = dims[3]
            dim1_ind = np.arange(dims[1])
            dim2_ind = np.arange(dims[2])
        elif sample_dir == 'y':
            sample_dim = dims[2]
            dim1_ind = np.arange(dims[1])
            dim2_ind = np.arange(dims[3])
        else:
            if not zw_flag:
                dims[1] = dims[1] + 1 #zw transport component does not have one additional grid cell in sample_dir
            sample_dim = dims[1]
            dim1_ind = np.arange(dims[2])
            dim2_ind = np.arange(dims[3])

        combined_field = np.zeros((dims[0],dims[1],dims[2],dims[3]))
        #Define where the sum of the indices for x- and y-direction are odd, such that the fields can be alternately stored in specified sample_dir.
        dim1cor, dim2cor  = np.meshgrid(dim2_ind, dim1_ind)
        twodimcor         = dim1cor + dim2cor
        twodimcor_odd     = twodimcor % 2 != 0
        twodimcor_odd     = np.tile(twodimcor_odd, (nt,1,1))
        twodimcor_even    = twodimcor % 2 == 0
        twodimcor_even    = np.tile(twodimcor_even, (nt,1,1))

        for i in range(sample_dim):
            #Determine index slices for upstream and downstream fields
            if sample_dir == 'x' and isotrop_flag and i == (sample_dim-1):
                indices_up   = np.s_[:,:,:,0]
                indices_down = np.s_[:,:,:,i]
                indices_combined = np.s_[:,:,:,i]
            elif sample_dir == 'x' and isotrop_flag:
                indices_up   = np.s_[:,:,:,i+1]
                indices_down = np.s_[:,:,:,i]
                indices_combined = np.s_[:,:,:,i]
            elif sample_dir == 'x':
                indices_up   = np.s_[:,:,:,i]
                indices_down = np.s_[:,:,:,i-1]
                indices_combined = np.s_[:,:,:,i]
            elif sample_dir == 'y' and isotrop_flag and i == (sample_dim-1):
                indices_up   = np.s_[:,:,0,:]
                indices_down = np.s_[:,:,i,:]
                indices_combined = np.s_[:,:,i,:]
            elif sample_dir == 'y' and isotrop_flag:
                indices_up   = np.s_[:,:,i+1,:]
                indices_down = np.s_[:,:,i,:]
                indices_combined = np.s_[:,:,i,:]
            elif sample_dir == 'y':
                indices_up   = np.s_[:,:,i,:]
                indices_down = np.s_[:,:,i-1,:]
                indices_combined = np.s_[:,:,i,:]
            elif zw_flag:
                indices_up   = np.s_[:,i+1,:,:]
                indices_down = np.s_[:,i,:,:]
                indices_combined = np.s_[:,i,:,:]
            else:
                indices_up   = np.s_[:,i,:,:]
                indices_down = np.s_[:,i-1,:,:]
                indices_combined = np.s_[:,i,:,:]

            #Compensate for shifted indices in the alternate sampling when zw_flag=true or isotrop_flag=true
            if zw_flag or isotrop_flag:
                twodimcor_odd2   = twodimcor_even
                twodimcor_even2  = twodimcor_odd
            else:
                twodimcor_odd2   = twodimcor_odd
                twodimcor_even2  = twodimcor_even
            
            #Assign upstream and downstream values to combined field, depends on whether the vertical or horizontal directions are considered.
            #NOTE: for the bottom and top layer when sampled in the z-direction, only the upstream and downstream field respectively are used.
            if (i == 0) and (sample_dir == 'z'):
                combined_field[indices_combined] = upstream_field[indices_up]
            elif (i == (sample_dim-1)) and (sample_dir == 'z'):
                combined_field[indices_combined] = downstream_field[indices_down]
            
            #Make distinction between i=even and i=odd to alternate storage
            elif i % 2 == 0: #i is even
                combined_field[indices_combined][twodimcor_odd2]  = upstream_field[indices_up][twodimcor_odd2]
                combined_field[indices_combined][twodimcor_even2] = downstream_field[indices_down][twodimcor_even2]
            elif i % 2 != 0: #i is odd
                combined_field[indices_combined][twodimcor_even2] = upstream_field[indices_up][twodimcor_even2]
                combined_field[indices_combined][twodimcor_odd2]  = downstream_field[indices_down][twodimcor_odd2]
            else:
                raise RuntimeError("Error occured in script as this line should not have been reached. Check carefully for bugs.")

        #Set bottom wall to 0 if zhless_wallszero is true to take into account that in the inference within MicroHH those values are effectively not used
        if zhless_wallszero:
            combined_field[:,0,:,:] = 0.

        return combined_field

    #Call function above for all preds and labels
    #NOTE: reconstructed fields isotropic components xu and yv on purpose no ghost cell at upstream side included, to prevent that values are counted twice in the PDF and/or correlation coefficient
    preds_values_xu =  _combine_upstream_downstream(preds_values_xu_upstream, preds_values_xu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x', isotrop_flag = True)
    preds_values_yu =  _combine_upstream_downstream(preds_values_yu_upstream, preds_values_yu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    preds_values_zu =  _combine_upstream_downstream(preds_values_zu_upstream, preds_values_zu_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    preds_values_xv =  _combine_upstream_downstream(preds_values_xv_upstream, preds_values_xv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    preds_values_yv =  _combine_upstream_downstream(preds_values_yv_upstream, preds_values_yv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y', isotrop_flag = True)
    preds_values_zv =  _combine_upstream_downstream(preds_values_zv_upstream, preds_values_zv_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    preds_values_xw =  _combine_upstream_downstream(preds_values_xw_upstream, preds_values_xw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x', zhless_wallszero = True)
    preds_values_yw =  _combine_upstream_downstream(preds_values_yw_upstream, preds_values_yw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y', zhless_wallszero = True)
    preds_values_zw =  _combine_upstream_downstream(preds_values_zw_upstream, preds_values_zw_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z', zw_flag = True)
    #
    unres_tau_xu_lbls =  _combine_upstream_downstream(unres_tau_xu_lbls_upstream, unres_tau_xu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x', isotrop_flag = True)
    unres_tau_yu_lbls =  _combine_upstream_downstream(unres_tau_yu_lbls_upstream, unres_tau_yu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    unres_tau_zu_lbls =  _combine_upstream_downstream(unres_tau_zu_lbls_upstream, unres_tau_zu_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    unres_tau_xv_lbls =  _combine_upstream_downstream(unres_tau_xv_lbls_upstream, unres_tau_xv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    unres_tau_yv_lbls =  _combine_upstream_downstream(unres_tau_yv_lbls_upstream, unres_tau_yv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y', isotrop_flag = True)
    unres_tau_zv_lbls =  _combine_upstream_downstream(unres_tau_zv_lbls_upstream, unres_tau_zv_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z')
    unres_tau_xw_lbls =  _combine_upstream_downstream(unres_tau_xw_lbls_upstream, unres_tau_xw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'x')
    unres_tau_yw_lbls =  _combine_upstream_downstream(unres_tau_yw_lbls_upstream, unres_tau_yw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'y')
    unres_tau_zw_lbls =  _combine_upstream_downstream(unres_tau_zw_lbls_upstream, unres_tau_zw_lbls_downstream, dims = [nt,nz,ny,nx], sample_dir = 'z', zw_flag = True)

    #Store variables in netCDF file
    #Labels
    var_unres_tau_xu_lbls[:,:,:,:] = unres_tau_xu_lbls[:,:,:,:]
    var_unres_tau_xv_lbls[:,:,:,:] = unres_tau_xv_lbls[:,:,:,:]
    var_unres_tau_xw_lbls[:,:,:,:] = unres_tau_xw_lbls[:,:,:,:]
    var_unres_tau_yu_lbls[:,:,:,:] = unres_tau_yu_lbls[:,:,:,:]
    var_unres_tau_yv_lbls[:,:,:,:] = unres_tau_yv_lbls[:,:,:,:]
    var_unres_tau_yw_lbls[:,:,:,:] = unres_tau_yw_lbls[:,:,:,:]
    var_unres_tau_zu_lbls[:,:,:,:] = unres_tau_zu_lbls[:,:,:,:]
    var_unres_tau_zv_lbls[:,:,:,:] = unres_tau_zv_lbls[:,:,:,:]
    var_unres_tau_zw_lbls[:,:,:,:] = unres_tau_zw_lbls[:,:,:,:]
    #
    var_unres_tau_xu_lbls_upstream[:,:,:,:] = unres_tau_xu_lbls_upstream[:,:,:,:]
    var_unres_tau_xv_lbls_upstream[:,:,:,:] = unres_tau_xv_lbls_upstream[:,:,:,:]
    var_unres_tau_xw_lbls_upstream[:,:,:,:] = unres_tau_xw_lbls_upstream[:,:,:,:]
    var_unres_tau_yu_lbls_upstream[:,:,:,:] = unres_tau_yu_lbls_upstream[:,:,:,:]
    var_unres_tau_yv_lbls_upstream[:,:,:,:] = unres_tau_yv_lbls_upstream[:,:,:,:]
    var_unres_tau_yw_lbls_upstream[:,:,:,:] = unres_tau_yw_lbls_upstream[:,:,:,:]
    var_unres_tau_zu_lbls_upstream[:,:,:,:] = unres_tau_zu_lbls_upstream[:,:,:,:]
    var_unres_tau_zv_lbls_upstream[:,:,:,:] = unres_tau_zv_lbls_upstream[:,:,:,:]
    var_unres_tau_zw_lbls_upstream[:,:,:,:] = unres_tau_zw_lbls_upstream[:,:,:,:]
    #
    var_unres_tau_xu_lbls_downstream[:,:,:,:] = unres_tau_xu_lbls_downstream[:,:,:,:]
    var_unres_tau_xv_lbls_downstream[:,:,:,:] = unres_tau_xv_lbls_downstream[:,:,:,:]
    var_unres_tau_xw_lbls_downstream[:,:,:,:] = unres_tau_xw_lbls_downstream[:,:,:,:]
    var_unres_tau_yu_lbls_downstream[:,:,:,:] = unres_tau_yu_lbls_downstream[:,:,:,:]
    var_unres_tau_yv_lbls_downstream[:,:,:,:] = unres_tau_yv_lbls_downstream[:,:,:,:]
    var_unres_tau_yw_lbls_downstream[:,:,:,:] = unres_tau_yw_lbls_downstream[:,:,:,:]
    var_unres_tau_zu_lbls_downstream[:,:,:,:] = unres_tau_zu_lbls_downstream[:,:,:,:]
    var_unres_tau_zv_lbls_downstream[:,:,:,:] = unres_tau_zv_lbls_downstream[:,:,:,:]
    var_unres_tau_zw_lbls_downstream[:,:,:,:] = unres_tau_zw_lbls_downstream[:,:,:,:]
    #Predictions
    var_unres_tau_xu_MLP[:,:,:,:] = preds_values_xu[:,:,:,:]
    var_unres_tau_xv_MLP[:,:,:,:] = preds_values_xv[:,:,:,:]
    var_unres_tau_xw_MLP[:,:,:,:] = preds_values_xw[:,:,:,:]
    var_unres_tau_yu_MLP[:,:,:,:] = preds_values_yu[:,:,:,:]
    var_unres_tau_yv_MLP[:,:,:,:] = preds_values_yv[:,:,:,:]
    var_unres_tau_yw_MLP[:,:,:,:] = preds_values_yw[:,:,:,:]
    var_unres_tau_zu_MLP[:,:,:,:] = preds_values_zu[:,:,:,:]
    var_unres_tau_zv_MLP[:,:,:,:] = preds_values_zv[:,:,:,:]
    var_unres_tau_zw_MLP[:,:,:,:] = preds_values_zw[:,:,:,:]
    #
    var_unres_tau_xu_MLP_upstream[:,:,:,:] = preds_values_xu_upstream[:,:,:,:]
    var_unres_tau_xv_MLP_upstream[:,:,:,:] = preds_values_xv_upstream[:,:,:,:]
    var_unres_tau_xw_MLP_upstream[:,:,:,:] = preds_values_xw_upstream[:,:,:,:]
    var_unres_tau_yu_MLP_upstream[:,:,:,:] = preds_values_yu_upstream[:,:,:,:]
    var_unres_tau_yv_MLP_upstream[:,:,:,:] = preds_values_yv_upstream[:,:,:,:]
    var_unres_tau_yw_MLP_upstream[:,:,:,:] = preds_values_yw_upstream[:,:,:,:]
    var_unres_tau_zu_MLP_upstream[:,:,:,:] = preds_values_zu_upstream[:,:,:,:]
    var_unres_tau_zv_MLP_upstream[:,:,:,:] = preds_values_zv_upstream[:,:,:,:]
    var_unres_tau_zw_MLP_upstream[:,:,:,:] = preds_values_zw_upstream[:,:,:,:]
    #
    var_unres_tau_xu_MLP_downstream[:,:,:,:] = preds_values_xu_downstream[:,:,:,:]
    var_unres_tau_xv_MLP_downstream[:,:,:,:] = preds_values_xv_downstream[:,:,:,:]
    var_unres_tau_xw_MLP_downstream[:,:,:,:] = preds_values_xw_downstream[:,:,:,:]
    var_unres_tau_yu_MLP_downstream[:,:,:,:] = preds_values_yu_downstream[:,:,:,:]
    var_unres_tau_yv_MLP_downstream[:,:,:,:] = preds_values_yv_downstream[:,:,:,:]
    var_unres_tau_yw_MLP_downstream[:,:,:,:] = preds_values_yw_downstream[:,:,:,:]
    var_unres_tau_zu_MLP_downstream[:,:,:,:] = preds_values_zu_downstream[:,:,:,:]
    var_unres_tau_zv_MLP_downstream[:,:,:,:] = preds_values_zv_downstream[:,:,:,:]
    var_unres_tau_zw_MLP_downstream[:,:,:,:] = preds_values_zw_downstream[:,:,:,:]
    var_unres_tau_zw_MLP_downstream[:,:,:,:] = preds_values_zw_downstream[:,:,:,:]
    
    
    #Create variables for storage unresolved, resolved, and total transports
    #var_unres_tau_xu_traceless = d.createVariable("unres_tau_xu_traceless","f8",("tstep_unique","zc","yc","xc"))
    #var_unres_tau_yv_traceless = d.createVariable("unres_tau_yv_traceless","f8",("tstep_unique","zc","yc","xc"))
    #var_unres_tau_zw_traceless = d.createVariable("unres_tau_zw_traceless","f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu = d.createVariable("unres_tau_xu","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv = d.createVariable("unres_tau_xv","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw = d.createVariable("unres_tau_xw","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu = d.createVariable("unres_tau_yu","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv = d.createVariable("unres_tau_yv","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw = d.createVariable("unres_tau_yw","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu = d.createVariable("unres_tau_zu","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv = d.createVariable("unres_tau_zv","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw = d.createVariable("unres_tau_zw","f8",("tstep_unique","zc","yc","xc"))
    #var_res_tau_xu   = d.createVariable("res_tau_xu"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_res_tau_xv   = d.createVariable("res_tau_xv"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_res_tau_xw   = d.createVariable("res_tau_xw"  ,"f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_res_tau_yu   = d.createVariable("res_tau_yu"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_res_tau_yv   = d.createVariable("res_tau_yv"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_res_tau_yw   = d.createVariable("res_tau_yw"  ,"f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_res_tau_zu   = d.createVariable("res_tau_zu"  ,"f8",("tstep_unique","zhc","yc","xhcless"))
    #var_res_tau_zv   = d.createVariable("res_tau_zv"  ,"f8",("tstep_unique","zhc","yhcless","xc"))
    #var_res_tau_zw   = d.createVariable("res_tau_zw"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_xu   = d.createVariable("tot_tau_xu"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_xv   = d.createVariable("tot_tau_xv"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_xw   = d.createVariable("tot_tau_xw"  ,"f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_tot_tau_yu   = d.createVariable("tot_tau_yu"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_yv   = d.createVariable("tot_tau_yv"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_yw   = d.createVariable("tot_tau_yw"  ,"f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_tot_tau_zu   = d.createVariable("tot_tau_zu"  ,"f8",("tstep_unique","zhc","yc","xhcless"))
    #var_tot_tau_zv   = d.createVariable("tot_tau_zv"  ,"f8",("tstep_unique","zhc","yhcless","xc"))
    #var_tot_tau_zw   = d.createVariable("tot_tau_zw"  ,"f8",("tstep_unique","zc","yc","xc"))
    #
    var_unres_tau_xu_smag = d.createVariable("unres_tau_xu_smag","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_xv_smag = d.createVariable("unres_tau_xv_smag","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_xw_smag = d.createVariable("unres_tau_xw_smag","f8",("tstep_unique","zhcless","yc","xhcless"))
    var_unres_tau_yu_smag = d.createVariable("unres_tau_yu_smag","f8",("tstep_unique","zc","yhcless","xhcless"))
    var_unres_tau_yv_smag = d.createVariable("unres_tau_yv_smag","f8",("tstep_unique","zc","yc","xc"))
    var_unres_tau_yw_smag = d.createVariable("unres_tau_yw_smag","f8",("tstep_unique","zhcless","yhcless","xc"))
    var_unres_tau_zu_smag = d.createVariable("unres_tau_zu_smag","f8",("tstep_unique","zhc","yc","xhcless"))
    var_unres_tau_zv_smag = d.createVariable("unres_tau_zv_smag","f8",("tstep_unique","zhc","yhcless","xc"))
    var_unres_tau_zw_smag = d.createVariable("unres_tau_zw_smag","f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_xu_smag   = d.createVariable("tot_tau_xu_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_xv_smag   = d.createVariable("tot_tau_xv_smag"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_xw_smag   = d.createVariable("tot_tau_xw_smag"  ,"f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_tot_tau_yu_smag   = d.createVariable("tot_tau_yu_smag"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_yv_smag   = d.createVariable("tot_tau_yv_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_yw_smag   = d.createVariable("tot_tau_yw_smag"  ,"f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_tot_tau_zu_smag   = d.createVariable("tot_tau_zu_smag"  ,"f8",("tstep_unique","zhc","yc","xhcless"))
    #var_tot_tau_zv_smag   = d.createVariable("tot_tau_zv_smag"  ,"f8",("tstep_unique","zhc","yhcless","xc"))
    #var_tot_tau_zw_smag   = d.createVariable("tot_tau_zw_smag"  ,"f8",("tstep_unique","zc","yc","xc"))
    #
    #var_tot_tau_xu_MLP   = d.createVariable("tot_tau_xu_MLP"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_xv_MLP   = d.createVariable("tot_tau_xv_MLP"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_xw_MLP   = d.createVariable("tot_tau_xw_MLP"  ,"f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_tot_tau_yu_MLP   = d.createVariable("tot_tau_yu_MLP"  ,"f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_tot_tau_yv_MLP   = d.createVariable("tot_tau_yv_MLP"  ,"f8",("tstep_unique","zc","yc","xc"))
    #var_tot_tau_yw_MLP   = d.createVariable("tot_tau_yw_MLP"  ,"f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_tot_tau_zu_MLP   = d.createVariable("tot_tau_zu_MLP"  ,"f8",("tstep_unique","zhc","yc","xhcless"))
    #var_tot_tau_zv_MLP   = d.createVariable("tot_tau_zv_MLP"  ,"f8",("tstep_unique","zhc","yhcless","xc"))
    #var_tot_tau_zw_MLP   = d.createVariable("tot_tau_zw_MLP"  ,"f8",("tstep_unique","zc","yc","xc"))

    #Store values for unresolved, resolved, and total fluxes training data, Smagorinsky, and MLP
    #var_unres_tau_xu_traceless[:,:,:,:] = unres_tau_xu_traceless
    #var_unres_tau_yv_traceless[:,:,:,:] = unres_tau_yv_traceless
    #var_unres_tau_zw_traceless[:,:,:,:] = unres_tau_zw_traceless
    #
    var_unres_tau_xu[:,:,:,:] = unres_tau_xu
    var_unres_tau_xv[:,:,:,:] = unres_tau_xv
    var_unres_tau_xw[:,:,:,:] = unres_tau_xw
    var_unres_tau_yu[:,:,:,:] = unres_tau_yu
    var_unres_tau_yv[:,:,:,:] = unres_tau_yv
    var_unres_tau_yw[:,:,:,:] = unres_tau_yw
    var_unres_tau_zu[:,:,:,:] = unres_tau_zu
    var_unres_tau_zv[:,:,:,:] = unres_tau_zv
    var_unres_tau_zw[:,:,:,:] = unres_tau_zw
    #
    #var_res_tau_xu[:,:,:,:]   = res_tau_xu
    #var_res_tau_xv[:,:,:,:]   = res_tau_xv
    #var_res_tau_xw[:,:,:,:]   = res_tau_xw
    #var_res_tau_yu[:,:,:,:]   = res_tau_yu
    #var_res_tau_yv[:,:,:,:]   = res_tau_yv
    #var_res_tau_yw[:,:,:,:]   = res_tau_yw
    #var_res_tau_zu[:,:,:,:]   = res_tau_zu
    #var_res_tau_zv[:,:,:,:]   = res_tau_zv
    #var_res_tau_zw[:,:,:,:]   = res_tau_zw
    #var_tot_tau_xu[:,:,:,:]   = tot_tau_xu 
    #var_tot_tau_xv[:,:,:,:]   = tot_tau_xv 
    #var_tot_tau_xw[:,:,:,:]   = tot_tau_xw 
    #var_tot_tau_yu[:,:,:,:]   = tot_tau_yu 
    #var_tot_tau_yv[:,:,:,:]   = tot_tau_yv 
    #var_tot_tau_yw[:,:,:,:]   = tot_tau_yw 
    #var_tot_tau_zu[:,:,:,:]   = tot_tau_zu 
    #var_tot_tau_zv[:,:,:,:]   = tot_tau_zv 
    #var_tot_tau_zw[:,:,:,:]   = tot_tau_zw 
    #
    var_unres_tau_xu_smag[:,:,:,:] = smag_tau_xu
    var_unres_tau_xv_smag[:,:,:,:] = smag_tau_xv
    var_unres_tau_xw_smag[:,:,:,:] = smag_tau_xw
    var_unres_tau_yu_smag[:,:,:,:] = smag_tau_yu
    var_unres_tau_yv_smag[:,:,:,:] = smag_tau_yv
    var_unres_tau_yw_smag[:,:,:,:] = smag_tau_yw
    var_unres_tau_zu_smag[:,:,:,:] = smag_tau_zu
    var_unres_tau_zv_smag[:,:,:,:] = smag_tau_zv
    var_unres_tau_zw_smag[:,:,:,:] = smag_tau_zw
    #var_tot_tau_xu_smag[:,:,:,:]   = smag_tau_xu + res_tau_xu 
    #var_tot_tau_xv_smag[:,:,:,:]   = smag_tau_xv + res_tau_xv 
    #var_tot_tau_xw_smag[:,:,:,:]   = smag_tau_xw + res_tau_xw 
    #var_tot_tau_yu_smag[:,:,:,:]   = smag_tau_yu + res_tau_yu 
    #var_tot_tau_yv_smag[:,:,:,:]   = smag_tau_yv + res_tau_yv 
    #var_tot_tau_yw_smag[:,:,:,:]   = smag_tau_yw + res_tau_yw 
    #var_tot_tau_zu_smag[:,:,:,:]   = smag_tau_zu + res_tau_zu 
    #var_tot_tau_zv_smag[:,:,:,:]   = smag_tau_zv + res_tau_zv 
    #var_tot_tau_zw_smag[:,:,:,:]   = smag_tau_zw + res_tau_zw 
    #var_tot_tau_xu_MLP[:,:,:,:]    = preds_values_xu  + res_tau_xu 
    #var_tot_tau_xv_MLP[:,:,:,:]    = preds_values_xv  + res_tau_xv 
    #var_tot_tau_xw_MLP[:,:,:,:]    = preds_values_xw  + res_tau_xw 
    #var_tot_tau_yu_MLP[:,:,:,:]    = preds_values_yu  + res_tau_yu 
    #var_tot_tau_yv_MLP[:,:,:,:]    = preds_values_yv  + res_tau_yv 
    #var_tot_tau_yw_MLP[:,:,:,:]    = preds_values_yw  + res_tau_yw 
    #var_tot_tau_zu_MLP[:,:,:,:]    = preds_values_zu  + res_tau_zu 
    #var_tot_tau_zv_MLP[:,:,:,:]    = preds_values_zv  + res_tau_zv 
    #var_tot_tau_zw_MLP[:,:,:,:]    = preds_values_zw  + res_tau_zw 

    #Create variables for storage horizontal averages
    var_unres_tau_xu_horavg = d.createVariable("unres_tau_xu_horavg","f8",("tstep_unique","zc"))
    #var_unres_tau_xu_traceless_horavg = d.createVariable("unres_tau_xu_traceless_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_horavg = d.createVariable("unres_tau_xv_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_horavg = d.createVariable("unres_tau_xw_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_yu_horavg = d.createVariable("unres_tau_yu_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_horavg = d.createVariable("unres_tau_yv_horavg","f8",("tstep_unique","zc"))
    #var_unres_tau_yv_traceless_horavg = d.createVariable("unres_tau_yv_traceless_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_horavg = d.createVariable("unres_tau_yw_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_zu_horavg = d.createVariable("unres_tau_zu_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_horavg = d.createVariable("unres_tau_zv_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_horavg = d.createVariable("unres_tau_zw_horavg","f8",("tstep_unique","zc"))
    #var_unres_tau_zw_traceless_horavg = d.createVariable("unres_tau_zw_traceless_horavg","f8",("tstep_unique","zc"))
    #var_res_tau_xu_horavg   = d.createVariable("res_tau_xu_horavg","f8",  ("tstep_unique","zc"))
    #var_res_tau_xv_horavg   = d.createVariable("res_tau_xv_horavg","f8",  ("tstep_unique","zc"))
    #var_res_tau_xw_horavg   = d.createVariable("res_tau_xw_horavg","f8",  ("tstep_unique","zhcless"))
    #var_res_tau_yu_horavg   = d.createVariable("res_tau_yu_horavg","f8",  ("tstep_unique","zc"))
    #var_res_tau_yv_horavg   = d.createVariable("res_tau_yv_horavg","f8",  ("tstep_unique","zc"))
    #var_res_tau_yw_horavg   = d.createVariable("res_tau_yw_horavg","f8",  ("tstep_unique","zhcless"))
    #var_res_tau_zu_horavg   = d.createVariable("res_tau_zu_horavg","f8",  ("tstep_unique","zhc"))
    #var_res_tau_zv_horavg   = d.createVariable("res_tau_zv_horavg","f8",  ("tstep_unique","zhc"))
    #var_res_tau_zw_horavg   = d.createVariable("res_tau_zw_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xu_horavg   = d.createVariable("tot_tau_xu_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xv_horavg   = d.createVariable("tot_tau_xv_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xw_horavg   = d.createVariable("tot_tau_xw_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_yu_horavg   = d.createVariable("tot_tau_yu_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yv_horavg   = d.createVariable("tot_tau_yv_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yw_horavg   = d.createVariable("tot_tau_yw_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_zu_horavg   = d.createVariable("tot_tau_zu_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zv_horavg   = d.createVariable("tot_tau_zv_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zw_horavg   = d.createVariable("tot_tau_zw_horavg","f8",  ("tstep_unique","zc"))
    #
    var_unres_tau_xu_smag_horavg = d.createVariable("unres_tau_xu_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_smag_horavg = d.createVariable("unres_tau_xv_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_smag_horavg = d.createVariable("unres_tau_xw_smag_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_yu_smag_horavg = d.createVariable("unres_tau_yu_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_smag_horavg = d.createVariable("unres_tau_yv_smag_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_smag_horavg = d.createVariable("unres_tau_yw_smag_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_zu_smag_horavg = d.createVariable("unres_tau_zu_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_smag_horavg = d.createVariable("unres_tau_zv_smag_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_smag_horavg = d.createVariable("unres_tau_zw_smag_horavg","f8",("tstep_unique","zc"))
    #var_tot_tau_xu_smag_horavg   = d.createVariable("tot_tau_xu_smag_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xv_smag_horavg   = d.createVariable("tot_tau_xv_smag_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xw_smag_horavg   = d.createVariable("tot_tau_xw_smag_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_yu_smag_horavg   = d.createVariable("tot_tau_yu_smag_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yv_smag_horavg   = d.createVariable("tot_tau_yv_smag_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yw_smag_horavg   = d.createVariable("tot_tau_yw_smag_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_zu_smag_horavg   = d.createVariable("tot_tau_zu_smag_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zv_smag_horavg   = d.createVariable("tot_tau_zv_smag_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zw_smag_horavg   = d.createVariable("tot_tau_zw_smag_horavg","f8",  ("tstep_unique","zc"))
    #
    var_unres_tau_xu_MLP_horavg = d.createVariable("unres_tau_xu_MLP_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xv_MLP_horavg = d.createVariable("unres_tau_xv_MLP_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_xw_MLP_horavg = d.createVariable("unres_tau_xw_MLP_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_yu_MLP_horavg = d.createVariable("unres_tau_yu_MLP_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yv_MLP_horavg = d.createVariable("unres_tau_yv_MLP_horavg","f8",("tstep_unique","zc"))
    var_unres_tau_yw_MLP_horavg = d.createVariable("unres_tau_yw_MLP_horavg","f8",("tstep_unique","zhcless"))
    var_unres_tau_zu_MLP_horavg = d.createVariable("unres_tau_zu_MLP_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zv_MLP_horavg = d.createVariable("unres_tau_zv_MLP_horavg","f8",("tstep_unique","zhc"))
    var_unres_tau_zw_MLP_horavg = d.createVariable("unres_tau_zw_MLP_horavg","f8",("tstep_unique","zc"))
    #var_tot_tau_xu_MLP_horavg   = d.createVariable("tot_tau_xu_MLP_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xv_MLP_horavg   = d.createVariable("tot_tau_xv_MLP_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_xw_MLP_horavg   = d.createVariable("tot_tau_xw_MLP_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_yu_MLP_horavg   = d.createVariable("tot_tau_yu_MLP_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yv_MLP_horavg   = d.createVariable("tot_tau_yv_MLP_horavg","f8",  ("tstep_unique","zc"))
    #var_tot_tau_yw_MLP_horavg   = d.createVariable("tot_tau_yw_MLP_horavg","f8",  ("tstep_unique","zhcless"))
    #var_tot_tau_zu_MLP_horavg   = d.createVariable("tot_tau_zu_MLP_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zv_MLP_horavg   = d.createVariable("tot_tau_zv_MLP_horavg","f8",  ("tstep_unique","zhc"))
    #var_tot_tau_zw_MLP_horavg   = d.createVariable("tot_tau_zw_MLP_horavg","f8",  ("tstep_unique","zc"))
 
    ##Create variables for storage fractions sub-grid fluxes
    #var_frac_unres_tau_xu = d.createVariable("frac_unres_tau_xu","f8",("tstep_unique","zc","yc","xc"))
    #var_frac_unres_tau_xv = d.createVariable("frac_unres_tau_xv","f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_frac_unres_tau_xw = d.createVariable("frac_unres_tau_xw","f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_frac_unres_tau_yu = d.createVariable("frac_unres_tau_yu","f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_frac_unres_tau_yv = d.createVariable("frac_unres_tau_yv","f8",("tstep_unique","zc","yc","xc"))
    #var_frac_unres_tau_yw = d.createVariable("frac_unres_tau_yw","f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_frac_unres_tau_zu = d.createVariable("frac_unres_tau_zu","f8",("tstep_unique","zhc","yc","xhcless"))
    #var_frac_unres_tau_zv = d.createVariable("frac_unres_tau_zv","f8",("tstep_unique","zhc","yhcless","xc"))
    #var_frac_unres_tau_zw = d.createVariable("frac_unres_tau_zw","f8",("tstep_unique","zc","yc","xc"))
    #var_frac_unres_tau_xu_horavg = d.createVariable("frac_unres_tau_xu_horavg","f8",("tstep_unique","zc"))
    #var_frac_unres_tau_xv_horavg = d.createVariable("frac_unres_tau_xv_horavg","f8",("tstep_unique","zc"))
    #var_frac_unres_tau_xw_horavg = d.createVariable("frac_unres_tau_xw_horavg","f8",("tstep_unique","zhcless"))
    #var_frac_unres_tau_yu_horavg = d.createVariable("frac_unres_tau_yu_horavg","f8",("tstep_unique","zc"))
    #var_frac_unres_tau_yv_horavg = d.createVariable("frac_unres_tau_yv_horavg","f8",("tstep_unique","zc"))
    #var_frac_unres_tau_yw_horavg = d.createVariable("frac_unres_tau_yw_horavg","f8",("tstep_unique","zhcless"))
    #var_frac_unres_tau_zu_horavg = d.createVariable("frac_unres_tau_zu_horavg","f8",("tstep_unique","zhc"))
    #var_frac_unres_tau_zv_horavg = d.createVariable("frac_unres_tau_zv_horavg","f8",("tstep_unique","zhc"))
    #var_frac_unres_tau_zw_horavg = d.createVariable("frac_unres_tau_zw_horavg","f8",("tstep_unique","zc"))

    #calculate horizontally averaged profiles.
    ##Calculate fraction unresolved compared to resolved fluxes, both point-by-point and horizontally averaged
    ##NOTE1: Some extreme outliers in the fractions occur when the total momentum transport reaches 0. To preven this from happening, the fractions are confined to the range -10 to 10.
    var_unres_tau_xu_horavg[:,:]          = np.mean(unres_tau_xu,                 axis=(2,3), keepdims=False)
    #var_unres_tau_xu_traceless_horavg[:,:] = np.mean(unres_tau_xu_traceless,       axis=(2,3), keepdims=False)
    #var_res_tau_xu_horavg[:,:]            = np.mean(res_tau_xu,                   axis=(2,3), keepdims=False)
    #var_tot_tau_xu_horavg[:,:]            = np.mean(tot_tau_xu,                   axis=(2,3), keepdims=False)
    var_unres_tau_xu_smag_horavg[:,:]     = np.mean(smag_tau_xu,                  axis=(2,3), keepdims=False)
    var_unres_tau_xu_MLP_horavg[:,:]      = np.mean(preds_values_xu,              axis=(2,3), keepdims=False)
    #var_tot_tau_xu_smag_horavg[:,:]       = np.mean(smag_tau_xu + res_tau_xu,     axis=(2,3), keepdims=False)
    #var_tot_tau_xu_MLP_horavg[:,:]        = np.mean(preds_values_xu + res_tau_xu, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_xu[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_xu[:,:,:,:]) / np.array(var_res_tau_xu[:,:,:,:])),-10)
    #var_frac_unres_tau_xu_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_xu_horavg[:,:]) / np.array(var_res_tau_xu_horavg[:,:])),-10)
    #
    var_unres_tau_xv_horavg[:,:]          = np.mean(unres_tau_xv,                 axis=(2,3), keepdims=False)
    #var_res_tau_xv_horavg[:,:]            = np.mean(res_tau_xv,                   axis=(2,3), keepdims=False)
    #var_tot_tau_xv_horavg[:,:]            = np.mean(tot_tau_xv,                   axis=(2,3), keepdims=False)
    var_unres_tau_xv_smag_horavg[:,:]     = np.mean(smag_tau_xv,                  axis=(2,3), keepdims=False)
    var_unres_tau_xv_MLP_horavg[:,:]      = np.mean(preds_values_xv,              axis=(2,3), keepdims=False)
    #var_tot_tau_xv_smag_horavg[:,:]       = np.mean(smag_tau_xv + res_tau_xv,     axis=(2,3), keepdims=False)
    #var_tot_tau_xv_MLP_horavg[:,:]        = np.mean(preds_values_xv + res_tau_xv, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_xv[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_xv[:,:,:,:]) / np.array(var_res_tau_xv[:,:,:,:])),-10)
    #var_frac_unres_tau_xv_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_xv_horavg[:,:]) / np.array(var_res_tau_xv_horavg[:,:])),-10)
    #
    var_unres_tau_xw_horavg[:,:]          = np.mean(unres_tau_xw,                 axis=(2,3), keepdims=False)
    #var_res_tau_xw_horavg[:,:]            = np.mean(res_tau_xw,                   axis=(2,3), keepdims=False)
    #var_tot_tau_xw_horavg[:,:]            = np.mean(tot_tau_xw,                   axis=(2,3), keepdims=False)
    var_unres_tau_xw_smag_horavg[:,:]     = np.mean(smag_tau_xw,                  axis=(2,3), keepdims=False)
    var_unres_tau_xw_MLP_horavg[:,:]      = np.mean(preds_values_xw,              axis=(2,3), keepdims=False)
    #var_tot_tau_xw_smag_horavg[:,:]       = np.mean(smag_tau_xw + res_tau_xw,     axis=(2,3), keepdims=False)
    #var_tot_tau_xw_MLP_horavg[:,:]        = np.mean(preds_values_xw + res_tau_xw, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_xw[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_xw[:,:,:,:]) / np.array(var_res_tau_xw[:,:,:,:])),-10)
    #var_frac_unres_tau_xw_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_xw_horavg[:,:]) / np.array(var_res_tau_xw_horavg[:,:])),-10)
    #
    var_unres_tau_yu_horavg[:,:]          = np.mean(unres_tau_yu,                 axis=(2,3), keepdims=False)
    #var_res_tau_yu_horavg[:,:]            = np.mean(res_tau_yu,                   axis=(2,3), keepdims=False)
    #var_tot_tau_yu_horavg[:,:]            = np.mean(tot_tau_yu,                   axis=(2,3), keepdims=False)
    var_unres_tau_yu_smag_horavg[:,:]     = np.mean(smag_tau_yu,                  axis=(2,3), keepdims=False)
    var_unres_tau_yu_MLP_horavg[:,:]      = np.mean(preds_values_yu,              axis=(2,3), keepdims=False)
    #var_tot_tau_yu_smag_horavg[:,:]       = np.mean(smag_tau_yu + res_tau_yu,     axis=(2,3), keepdims=False)
    #var_tot_tau_yu_MLP_horavg[:,:]        = np.mean(preds_values_yu + res_tau_yu, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_yu[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_yu[:,:,:,:]) / np.array(var_res_tau_yu[:,:,:,:])),-10)
    #var_frac_unres_tau_yu_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_yu_horavg[:,:]) / np.array(var_res_tau_yu_horavg[:,:])),-10)
    #
    var_unres_tau_yv_horavg[:,:]          = np.mean(unres_tau_yv,                 axis=(2,3), keepdims=False)
    #var_unres_tau_yv_traceless_horavg[:,:] = np.mean(unres_tau_yv_traceless,       axis=(2,3), keepdims=False)
    #var_res_tau_yv_horavg[:,:]            = np.mean(res_tau_yv,                   axis=(2,3), keepdims=False)
    #var_tot_tau_yv_horavg[:,:]            = np.mean(tot_tau_yv,                   axis=(2,3), keepdims=False)
    var_unres_tau_yv_smag_horavg[:,:]     = np.mean(smag_tau_yv,                  axis=(2,3), keepdims=False)
    var_unres_tau_yv_MLP_horavg[:,:]      = np.mean(preds_values_yv,              axis=(2,3), keepdims=False)
    #var_tot_tau_yv_smag_horavg[:,:]       = np.mean(smag_tau_yv + res_tau_yv,     axis=(2,3), keepdims=False)
    #var_tot_tau_yv_MLP_horavg[:,:]        = np.mean(preds_values_yv + res_tau_yv, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_yv[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_yv[:,:,:,:]) / np.array(var_res_tau_yv[:,:,:,:])),-10)
    #var_frac_unres_tau_yv_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_yv_horavg[:,:]) / np.array(var_res_tau_yv_horavg[:,:])),-10)
    #
    var_unres_tau_yw_horavg[:,:]          = np.mean(unres_tau_yw,                 axis=(2,3), keepdims=False)
    #var_res_tau_yw_horavg[:,:]            = np.mean(res_tau_yw,                   axis=(2,3), keepdims=False)
    #var_tot_tau_yw_horavg[:,:]            = np.mean(tot_tau_yw,                   axis=(2,3), keepdims=False)
    var_unres_tau_yw_smag_horavg[:,:]     = np.mean(smag_tau_yw,                  axis=(2,3), keepdims=False)
    var_unres_tau_yw_MLP_horavg[:,:]      = np.mean(preds_values_yw,              axis=(2,3), keepdims=False)
    #var_tot_tau_yw_smag_horavg[:,:]       = np.mean(smag_tau_yw + res_tau_yw,     axis=(2,3), keepdims=False)
    #var_tot_tau_yw_MLP_horavg[:,:]        = np.mean(preds_values_yw + res_tau_yw, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_yw[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_yw[:,:,:,:]) / np.array(var_res_tau_yw[:,:,:,:])),-10)
    #var_frac_unres_tau_yw_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_yw_horavg[:,:]) / np.array(var_res_tau_yw_horavg[:,:])),-10)
    #
    var_unres_tau_zu_horavg[:,:]          = np.mean(unres_tau_zu,                 axis=(2,3), keepdims=False)
    #var_res_tau_zu_horavg[:,:]            = np.mean(res_tau_zu,                   axis=(2,3), keepdims=False)
    #var_tot_tau_zu_horavg[:,:]            = np.mean(tot_tau_zu,                   axis=(2,3), keepdims=False)
    var_unres_tau_zu_smag_horavg[:,:]     = np.mean(smag_tau_zu,                  axis=(2,3), keepdims=False)
    var_unres_tau_zu_MLP_horavg[:,:]      = np.mean(preds_values_zu,              axis=(2,3), keepdims=False)
    #var_tot_tau_zu_smag_horavg[:,:]       = np.mean(smag_tau_zu + res_tau_zu,     axis=(2,3), keepdims=False)
    #var_tot_tau_zu_MLP_horavg[:,:]        = np.mean(preds_values_zu + res_tau_zu, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_zu[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_zu[:,:,:,:]) / np.array(var_res_tau_zu[:,:,:,:])),-10)
    #var_frac_unres_tau_zu_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_zu_horavg[:,:]) / np.array(var_res_tau_zu_horavg[:,:])),-10)
    #
    var_unres_tau_zv_horavg[:,:]          = np.mean(unres_tau_zv,                 axis=(2,3), keepdims=False)
    #var_res_tau_zv_horavg[:,:]            = np.mean(res_tau_zv,                   axis=(2,3), keepdims=False)
    #var_tot_tau_zv_horavg[:,:]            = np.mean(tot_tau_zv,                   axis=(2,3), keepdims=False)
    var_unres_tau_zv_smag_horavg[:,:]     = np.mean(smag_tau_zv,                  axis=(2,3), keepdims=False)
    var_unres_tau_zv_MLP_horavg[:,:]      = np.mean(preds_values_zv,              axis=(2,3), keepdims=False)
    #var_tot_tau_zv_smag_horavg[:,:]       = np.mean(smag_tau_zv + res_tau_zv,     axis=(2,3), keepdims=False)
    #var_tot_tau_zv_MLP_horavg[:,:]        = np.mean(preds_values_zv + res_tau_zv, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_zv[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_zv[:,:,:,:]) / np.array(var_res_tau_zv[:,:,:,:])),-10)
    #var_frac_unres_tau_zv_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_zv_horavg[:,:]) / np.array(var_res_tau_zv_horavg[:,:])),-10)
    #
    var_unres_tau_zw_horavg[:,:]          = np.mean(unres_tau_zw,                 axis=(2,3), keepdims=False)
    #var_unres_tau_zw_traceless_horavg[:,:] = np.mean(unres_tau_zw_traceless,       axis=(2,3), keepdims=False)
    #var_res_tau_zw_horavg[:,:]            = np.mean(res_tau_zw,                   axis=(2,3), keepdims=False)
    #var_tot_tau_zw_horavg[:,:]            = np.mean(tot_tau_zw,                   axis=(2,3), keepdims=False)
    var_unres_tau_zw_smag_horavg[:,:]     = np.mean(smag_tau_zw,                  axis=(2,3), keepdims=False)
    var_unres_tau_zw_MLP_horavg[:,:]      = np.mean(preds_values_zw,              axis=(2,3), keepdims=False)
    #var_tot_tau_zw_smag_horavg[:,:]       = np.mean(smag_tau_zw + res_tau_zw,     axis=(2,3), keepdims=False)
    #var_tot_tau_zw_MLP_horavg[:,:]        = np.mean(preds_values_zw + res_tau_zw, axis=(2,3), keepdims=False)
    #var_frac_unres_tau_zw[:,:,:,:]        = np.maximum(np.minimum(10,np.array(var_unres_tau_zw[:,:,:,:]) / np.array(var_res_tau_zw[:,:,:,:])),-10)
    #var_frac_unres_tau_zw_horavg[:,:]     = np.maximum(np.minimum(10,np.array(var_unres_tau_zw_horavg[:,:]) / np.array(var_res_tau_zw_horavg[:,:])),-10)

    #Close netCDF-files
    d.close()

#Calculate and store dissipation if specified
if args.calc_dissipation:
    
    print('Start reading variables needed to calculate dissipation.')
    print('Note: the dissipation is calculated under the assumption that the flow fields have periodic BCs in the horizontal directions, and a no-slip BC in the vertical. In addition, it is assumed that the coarse-grained grid is equidistant. If this is not the case, change the script accordingly.')
   
    #Read and create corresponding files
    fields = nc.Dataset("reconstructed_fields.nc", "r")
    training = nc.Dataset(args.training_file, "r")
    dissipation = nc.Dataset("dissipation.nc", "w")

    #Read coordinates with ghostcells associated with velocity fields, extract ghostcells
    zhgc = np.array(training['zhgc'][:])
    zgc = np.array(training['zgc'][:])
    yhgc = np.array(training['yhgc'][:])
    ygc = np.array(training['ygc'][:])
    xhgc = np.array(training['xhgc'][:])
    xgc = np.array(training['xgc'][:])
    #
    igc = int(training['igc'][:])
    jgc = int(training['jgc'][:])
    kgc_center = int(training['kgc_center'][:])
    kgc_edge = int(training['kgc_edge'][:])
    ihend          = int(training['ihend'][:])
    iend           = int(training['iend'][:])
    jhend          = int(training['jhend'][:])
    jend           = int(training['jend'][:])
    khend          = int(training['khend'][:])
    kend           = int(training['kend'][:])

    #Define time steps, based on previously defined tstart, tend that designate the testing time steps.
    tstep = np.arange(tstart, tend, dtype=np.int64)

    #Raise error if kgc_center and kgc_edge are not equal; script not designed for this scenario
    if kgc_center != kgc_center:
        raise RuntimeError("The number of ghostcells in the vertical are not equal for the centered and staggered variables. This script does not take such a difference into account. Please revise the script if necessary.")
   
    #Previously already extracted coordinates, don't read them again
    #zc  = np.array(c['zc'][:])
    #nz = len(zc)
    #zhc = np.array(c['zhc'][:])
    #zgcextra = np.array(c['zgcextra'][:])
    #yc  = np.array(c['yc'][:])
    #ny = len(yc)
    #yhc = np.array(c['yhc'][:])
    #ygcextra = np.array(c['ygcextra'][:])
    #xc  = np.array(c['xc'][:])
    #nx = len(xc)
    #xhc = np.array(c['xhc'][:])
    #xgcextra = np.array(c['xgcextra'][:])
    #zhcless = zhc[:-1]
    #yhcless = yhc[:-1]
    #xhcless = xhc[:-1]

    #Extract filtered velocity fields, use previously defined tstart, tend (which designate the testing period of the training dataset)
    uc = np.array(training['uc'][tstart:tend,:,:,:])
    vc = np.array(training['vc'][tstart:tend,:,:,:])
    wc = np.array(training['wc'][tstart:tend,:,:,:])
    
    #Extract MLP fluxes
    unres_tau_xu_MLP = np.array(fields['unres_tau_xu_MLP'][:,:,:,:])
    unres_tau_xv_MLP = np.array(fields['unres_tau_xv_MLP'][:,:,:,:])
    unres_tau_xw_MLP = np.array(fields['unres_tau_xw_MLP'][:,:,:,:])
    unres_tau_yu_MLP = np.array(fields['unres_tau_yu_MLP'][:,:,:,:])
    unres_tau_yv_MLP = np.array(fields['unres_tau_yv_MLP'][:,:,:,:])
    unres_tau_yw_MLP = np.array(fields['unres_tau_yw_MLP'][:,:,:,:])
    unres_tau_zu_MLP = np.array(fields['unres_tau_zu_MLP'][:,:,:,:])
    unres_tau_zv_MLP = np.array(fields['unres_tau_zv_MLP'][:,:,:,:])
    unres_tau_zw_MLP = np.array(fields['unres_tau_zw_MLP'][:,:,:,:])
    
    #Extract Smagorinsky fluxes
    unres_tau_xu_smag = np.array(fields['unres_tau_xu_smag'][:,:,:,:])
    unres_tau_xv_smag = np.array(fields['unres_tau_xv_smag'][:,:,:,:])
    unres_tau_xw_smag = np.array(fields['unres_tau_xw_smag'][:,:,:,:])
    unres_tau_yu_smag = np.array(fields['unres_tau_yu_smag'][:,:,:,:])
    unres_tau_yv_smag = np.array(fields['unres_tau_yv_smag'][:,:,:,:])
    unres_tau_yw_smag = np.array(fields['unres_tau_yw_smag'][:,:,:,:])
    unres_tau_zu_smag = np.array(fields['unres_tau_zu_smag'][:,:,:,:])
    unres_tau_zv_smag = np.array(fields['unres_tau_zv_smag'][:,:,:,:])
    unres_tau_zw_smag = np.array(fields['unres_tau_zw_smag'][:,:,:,:])
    
    #Extract training fluxes
    #unres_tau_xu_traceless = np.array(fields['unres_tau_xu_traceless'][:,:,:,:])
    #unres_tau_yv_traceless = np.array(fields['unres_tau_yv_traceless'][:,:,:,:])
    #unres_tau_zw_traceless = np.array(fields['unres_tau_zw_traceless'][:,:,:,:])
    #
    unres_tau_xu_lbl = np.array(fields['unres_tau_xu'][:,:,:,:])
    unres_tau_xv_lbl = np.array(fields['unres_tau_xv'][:,:,:,:])
    unres_tau_xw_lbl = np.array(fields['unres_tau_xw'][:,:,:,:])
    unres_tau_yu_lbl = np.array(fields['unres_tau_yu'][:,:,:,:])
    unres_tau_yv_lbl = np.array(fields['unres_tau_yv'][:,:,:,:])
    unres_tau_yw_lbl = np.array(fields['unres_tau_yw'][:,:,:,:])
    unres_tau_zu_lbl = np.array(fields['unres_tau_zu'][:,:,:,:])
    unres_tau_zv_lbl = np.array(fields['unres_tau_zv'][:,:,:,:])
    unres_tau_zw_lbl = np.array(fields['unres_tau_zw'][:,:,:,:])

    #Create dimensions for storage in nc-file
    dissipation.createDimension("zc", len(zc))
    dissipation.createDimension("zhc",len(zhc))
    #dissipation.createDimension("zhcless",len(zhcless))
    dissipation.createDimension("yc", len(yc))
    dissipation.createDimension("yhc",len(yhc))
    dissipation.createDimension("xc", len(xc))
    dissipation.createDimension("xhc",len(xhc))
    dissipation.createDimension("tstep",len(tstep))

    #Create and store dimension variables in nc-file
    var_zc = dissipation.createVariable("zc","f8",("zc",))
    var_zc[:]  = zc
    var_zhc = dissipation.createVariable("zhc","f8",("zhc",))
    var_zhc[:] = zhc
    var_yc = dissipation.createVariable("yc","f8",("yc",))
    var_yc[:]  = yc
    var_yhc = dissipation.createVariable("yhc","f8",("yhc",))
    var_yhc[:] = yhc
    var_xc = dissipation.createVariable("xc","f8",("xc",))
    var_xc[:]  = xc
    var_xhc = dissipation.createVariable("xhc","f8",("xhc",))
    var_xhc[:] = xhc
    var_tstep = dissipation.createVariable("tstep","f8",("tstep",))
    var_tstep[:] = tstep
    
    #Create variables for storage in nc-file
    var_diss_tau_xu_MLP     = dissipation.createVariable("diss_tau_xu_MLP","f8",("tstep","zc","yc","xc"))
    var_diss_tau_xv_MLP     = dissipation.createVariable("diss_tau_xv_MLP","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_xw_MLP     = dissipation.createVariable("diss_tau_xw_MLP","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_yu_MLP     = dissipation.createVariable("diss_tau_yu_MLP","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_yv_MLP     = dissipation.createVariable("diss_tau_yv_MLP","f8",("tstep","zc","yc","xc"))
    var_diss_tau_yw_MLP     = dissipation.createVariable("diss_tau_yw_MLP","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zu_MLP     = dissipation.createVariable("diss_tau_zu_MLP","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_zv_MLP     = dissipation.createVariable("diss_tau_zv_MLP","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zw_MLP     = dissipation.createVariable("diss_tau_zw_MLP","f8",("tstep","zc","yc","xc"))
    var_diss_tot_MLP        = dissipation.createVariable("diss_tot_MLP","f8",("tstep","zc","yc","xc"))
    var_diss_tot_MLP_horavg = dissipation.createVariable("diss_tot_MLP_horavg","f8",("tstep","zc"))
    #
    var_diss_tau_xu_smag     = dissipation.createVariable("diss_tau_xu_smag","f8",("tstep","zc","yc","xc"))
    var_diss_tau_xv_smag     = dissipation.createVariable("diss_tau_xv_smag","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_xw_smag     = dissipation.createVariable("diss_tau_xw_smag","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_yu_smag     = dissipation.createVariable("diss_tau_yu_smag","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_yv_smag     = dissipation.createVariable("diss_tau_yv_smag","f8",("tstep","zc","yc","xc"))
    var_diss_tau_yw_smag     = dissipation.createVariable("diss_tau_yw_smag","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zu_smag     = dissipation.createVariable("diss_tau_zu_smag","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_zv_smag     = dissipation.createVariable("diss_tau_zv_smag","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zw_smag     = dissipation.createVariable("diss_tau_zw_smag","f8",("tstep","zc","yc","xc"))
    var_diss_tot_smag        = dissipation.createVariable("diss_tot_smag","f8",("tstep","zc","yc","xc"))
    var_diss_tot_smag_horavg = dissipation.createVariable("diss_tot_smag_horavg","f8",("tstep","zc"))
    #
    var_diss_tau_xu_lbl     = dissipation.createVariable("diss_tau_xu_lbl","f8",("tstep","zc","yc","xc"))
    var_diss_tau_xv_lbl     = dissipation.createVariable("diss_tau_xv_lbl","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_xw_lbl     = dissipation.createVariable("diss_tau_xw_lbl","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_yu_lbl     = dissipation.createVariable("diss_tau_yu_lbl","f8",("tstep","zc","yhc","xhc"))
    var_diss_tau_yv_lbl     = dissipation.createVariable("diss_tau_yv_lbl","f8",("tstep","zc","yc","xc"))
    var_diss_tau_yw_lbl     = dissipation.createVariable("diss_tau_yw_lbl","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zu_lbl     = dissipation.createVariable("diss_tau_zu_lbl","f8",("tstep","zhc","yc","xhc"))
    var_diss_tau_zv_lbl     = dissipation.createVariable("diss_tau_zv_lbl","f8",("tstep","zhc","yhc","xc"))
    var_diss_tau_zw_lbl     = dissipation.createVariable("diss_tau_zw_lbl","f8",("tstep","zc","yc","xc"))
    var_diss_tot_lbl        = dissipation.createVariable("diss_tot_lbl","f8",("tstep","zc","yc","xc"))
    var_diss_tot_lbl_horavg = dissipation.createVariable("diss_tot_lbl_horavg","f8",("tstep","zc"))

    #Add ghostcells in horizontal (periodic BCs) and vertical (no-slip BC). At the same time, intialize dissipation components
    diss_tau_xu_MLP = copy.deepcopy(unres_tau_xu_MLP)
    diss_tau_yv_MLP = copy.deepcopy(unres_tau_yv_MLP)
    diss_tau_zw_MLP = copy.deepcopy(unres_tau_zw_MLP)
    #
    diss_tau_xv_MLP = copy.deepcopy(unres_tau_xv_MLP)
    diss_tau_xv_MLP = np.append(diss_tau_xv_MLP, diss_tau_xv_MLP[:,:,np.newaxis,0,:], axis=2)
    diss_tau_xv_MLP = np.append(diss_tau_xv_MLP, diss_tau_xv_MLP[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_xw_MLP = copy.deepcopy(unres_tau_xw_MLP)
    diss_tau_xw_MLP = np.append(diss_tau_xw_MLP, np.zeros(diss_tau_xw_MLP.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_xw_MLP = np.append(diss_tau_xw_MLP, diss_tau_xw_MLP[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yu_MLP = copy.deepcopy(unres_tau_yu_MLP)
    diss_tau_yu_MLP = np.append(diss_tau_yu_MLP, diss_tau_yu_MLP[:,:,np.newaxis,0,:], axis=2)
    diss_tau_yu_MLP = np.append(diss_tau_yu_MLP, diss_tau_yu_MLP[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yw_MLP = copy.deepcopy(unres_tau_yw_MLP)
    diss_tau_yw_MLP = np.append(diss_tau_yw_MLP, np.zeros(diss_tau_yw_MLP.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_yw_MLP = np.append(diss_tau_yw_MLP, diss_tau_yw_MLP[:,:,np.newaxis,0,:], axis=2)
    #
    diss_tau_zu_MLP = copy.deepcopy(unres_tau_zu_MLP)
    diss_tau_zu_MLP = np.append(diss_tau_zu_MLP, diss_tau_zu_MLP[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_zv_MLP = copy.deepcopy(unres_tau_zv_MLP)
    diss_tau_zv_MLP = np.append(diss_tau_zv_MLP, diss_tau_zv_MLP[:,:,np.newaxis,0,:], axis=2)
    #
    diss_tau_xu_lbl = copy.deepcopy(unres_tau_xu_lbl)
    diss_tau_yv_lbl = copy.deepcopy(unres_tau_yv_lbl)
    diss_tau_zw_lbl = copy.deepcopy(unres_tau_zw_lbl)
    #
    diss_tau_xv_lbl = copy.deepcopy(unres_tau_xv_lbl)
    diss_tau_xv_lbl = np.append(diss_tau_xv_lbl, diss_tau_xv_lbl[:,:,np.newaxis,0,:], axis=2)
    diss_tau_xv_lbl = np.append(diss_tau_xv_lbl, diss_tau_xv_lbl[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_xw_lbl = copy.deepcopy(unres_tau_xw_lbl)
    diss_tau_xw_lbl = np.append(diss_tau_xw_lbl, np.zeros(diss_tau_xw_lbl.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_xw_lbl = np.append(diss_tau_xw_lbl, diss_tau_xw_lbl[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yu_lbl = copy.deepcopy(unres_tau_yu_lbl)
    diss_tau_yu_lbl = np.append(diss_tau_yu_lbl, diss_tau_yu_lbl[:,:,np.newaxis,0,:], axis=2)
    diss_tau_yu_lbl = np.append(diss_tau_yu_lbl, diss_tau_yu_lbl[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yw_lbl = copy.deepcopy(unres_tau_yw_lbl)
    diss_tau_yw_lbl = np.append(diss_tau_yw_lbl, np.zeros(diss_tau_yw_lbl.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_yw_lbl = np.append(diss_tau_yw_lbl, diss_tau_yw_lbl[:,:,np.newaxis,0,:], axis=2)
    #
    diss_tau_zu_lbl = copy.deepcopy(unres_tau_zu_lbl)
    diss_tau_zu_lbl = np.append(diss_tau_zu_lbl, diss_tau_zu_lbl[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_zv_lbl = copy.deepcopy(unres_tau_zv_lbl)
    diss_tau_zv_lbl = np.append(diss_tau_zv_lbl, diss_tau_zv_lbl[:,:,np.newaxis,0,:], axis=2)
    #
    diss_tau_xu_smag = copy.deepcopy(unres_tau_xu_smag)
    diss_tau_yv_smag = copy.deepcopy(unres_tau_yv_smag)
    diss_tau_zw_smag = copy.deepcopy(unres_tau_zw_smag)
    #
    diss_tau_xv_smag = copy.deepcopy(unres_tau_xv_smag)
    diss_tau_xv_smag = np.append(diss_tau_xv_smag, diss_tau_xv_smag[:,:,np.newaxis,0,:], axis=2)
    diss_tau_xv_smag = np.append(diss_tau_xv_smag, diss_tau_xv_smag[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_xw_smag = copy.deepcopy(unres_tau_xw_smag)
    diss_tau_xw_smag = np.append(diss_tau_xw_smag, np.zeros(diss_tau_xw_smag.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_xw_smag = np.append(diss_tau_xw_smag, diss_tau_xw_smag[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yu_smag = copy.deepcopy(unres_tau_yu_smag)
    diss_tau_yu_smag = np.append(diss_tau_yu_smag, diss_tau_yu_smag[:,:,np.newaxis,0,:], axis=2)
    diss_tau_yu_smag = np.append(diss_tau_yu_smag, diss_tau_yu_smag[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_yw_smag = copy.deepcopy(unres_tau_yw_smag)
    diss_tau_yw_smag = np.append(diss_tau_yw_smag, np.zeros(diss_tau_yw_smag.shape, dtype=np.float64)[:,np.newaxis,0,:,:], axis=1)
    diss_tau_yw_smag = np.append(diss_tau_yw_smag, diss_tau_yw_smag[:,:,np.newaxis,0,:], axis=2)
    #
    diss_tau_zu_smag = copy.deepcopy(unres_tau_zu_smag)
    diss_tau_zu_smag = np.append(diss_tau_zu_smag, diss_tau_zu_smag[:,:,:,np.newaxis,0], axis=3)
    #
    diss_tau_zv_smag = copy.deepcopy(unres_tau_zv_smag)
    diss_tau_zv_smag = np.append(diss_tau_zv_smag, diss_tau_zv_smag[:,:,np.newaxis,0,:], axis=2)
    #

    #Initialize arrays for dissipation calculation
    diss_tot_MLP  = np.zeros((len(tstep),len(zc),len(yc),len(xc)), dtype=np.float64)
    diss_tot_smag = np.zeros((len(tstep),len(zc),len(yc),len(xc)), dtype=np.float64)
    diss_tot_lbl  = np.zeros((len(tstep),len(zc),len(yc),len(xc)), dtype=np.float64)

    #Calculate grid distances
    nxc = len(xc)
    nyc = len(yc)
    nzc = len(zc)
    zlen = zhc[-1]
    ylen = yhc[-1]
    xlen = xhc[-1]
    dzi = nzc / zlen
    dyi = nyc / ylen
    dxi = nxc / xlen

    #Calculate all the individual components
    #xu
    diss_tau_xu_MLP = - diss_tau_xu_MLP * (uc[:,kgc_center:kend,jgc:jend,igc+1:ihend] - uc[:,kgc_center:kend,jgc:jend,igc:ihend-1]) * dxi
    diss_tau_xu_smag = - diss_tau_xu_smag * (uc[:,kgc_center:kend,jgc:jend,igc+1:ihend] - uc[:,kgc_center:kend,jgc:jend,igc:ihend-1]) * dxi
    diss_tau_xu_lbl = - diss_tau_xu_lbl * (uc[:,kgc_center:kend,jgc:jend,igc+1:ihend] - uc[:,kgc_center:kend,jgc:jend,igc:ihend-1]) * dxi
    #
    diss_tot_MLP[:,:,:,:]  += diss_tau_xu_MLP[:,:,:,:]
    diss_tot_smag[:,:,:,:] += diss_tau_xu_smag[:,:,:,:]
    diss_tot_lbl[:,:,:,:]  += diss_tau_xu_lbl[:,:,:,:]

    #yu
    diss_tau_yu_MLP = - diss_tau_yu_MLP * (uc[:,kgc_center:kend,jgc:jend+1,igc:ihend] - uc[:,kgc_center:kend,jgc-1:jend,igc:ihend]) * dyi
    diss_tau_yu_smag = - diss_tau_yu_smag * (uc[:,kgc_center:kend,jgc:jend+1,igc:ihend] - uc[:,kgc_center:kend,jgc-1:jend,igc:ihend]) * dyi
    diss_tau_yu_lbl = - diss_tau_yu_lbl * (uc[:,kgc_center:kend,jgc:jend+1,igc:ihend] - uc[:,kgc_center:kend,jgc-1:jend,igc:ihend]) * dyi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_yu_MLP[:,:,:-1,:-1] + diss_tau_yu_MLP[:,:,:-1,1:] + diss_tau_yu_MLP[:,:,1:,:-1]  + diss_tau_yu_MLP[:,:,1:,1:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_yu_smag[:,:,:-1,:-1] + diss_tau_yu_smag[:,:,:-1,1:] + diss_tau_yu_smag[:,:,1:,:-1]  + diss_tau_yu_smag[:,:,1:,1:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_yu_lbl[:,:,:-1,:-1] + diss_tau_yu_lbl[:,:,:-1,1:] + diss_tau_yu_lbl[:,:,1:,:-1]  + diss_tau_yu_lbl[:,:,1:,1:])
    
    #zu
    diss_tau_zu_MLP = - diss_tau_zu_MLP * (uc[:,kgc_center:kend+1,jgc:jend,igc:ihend] - uc[:,kgc_center-1:kend,jgc:jend,igc:ihend]) * dzi
    diss_tau_zu_smag = - diss_tau_zu_smag * (uc[:,kgc_center:kend+1,jgc:jend,igc:ihend] - uc[:,kgc_center-1:kend,jgc:jend,igc:ihend]) * dzi
    diss_tau_zu_lbl = - diss_tau_zu_lbl * (uc[:,kgc_center:kend+1,jgc:jend,igc:ihend] - uc[:,kgc_center-1:kend,jgc:jend,igc:ihend]) * dzi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_zu_MLP[:,:-1,:,:-1] + diss_tau_zu_MLP[:,:-1,:,1:] + diss_tau_zu_MLP[:,1:,:,:-1]  + diss_tau_zu_MLP[:,1:,:,1:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_zu_smag[:,:-1,:,:-1] + diss_tau_zu_smag[:,:-1,:,1:] + diss_tau_zu_smag[:,1:,:,:-1]  + diss_tau_zu_smag[:,1:,:,1:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_zu_lbl[:,:-1,:,:-1] + diss_tau_zu_lbl[:,:-1,:,1:] + diss_tau_zu_lbl[:,1:,:,:-1]  + diss_tau_zu_lbl[:,1:,:,1:])
    
    #xv
    diss_tau_xv_MLP = - diss_tau_xv_MLP * (vc[:,kgc_center:kend,jgc:jhend,igc:iend+1] - vc[:,kgc_center:kend,jgc:jhend,igc-1:iend]) * dxi
    diss_tau_xv_smag = - diss_tau_xv_smag * (vc[:,kgc_center:kend,jgc:jhend,igc:iend+1] - vc[:,kgc_center:kend,jgc:jhend,igc-1:iend]) * dxi
    diss_tau_xv_lbl = - diss_tau_xv_lbl * (vc[:,kgc_center:kend,jgc:jhend,igc:iend+1] - vc[:,kgc_center:kend,jgc:jhend,igc-1:iend]) * dxi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_xv_MLP[:,:,:-1,:-1] + diss_tau_xv_MLP[:,:,:-1,1:] + diss_tau_xv_MLP[:,:,1:,:-1]  + diss_tau_xv_MLP[:,:,1:,1:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_xv_smag[:,:,:-1,:-1] + diss_tau_xv_smag[:,:,:-1,1:] + diss_tau_xv_smag[:,:,1:,:-1]  + diss_tau_xv_smag[:,:,1:,1:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_xv_lbl[:,:,:-1,:-1] + diss_tau_xv_lbl[:,:,:-1,1:] + diss_tau_xv_lbl[:,:,1:,:-1]  + diss_tau_xv_lbl[:,:,1:,1:])
    
    #yv
    diss_tau_yv_MLP = - diss_tau_yv_MLP * (vc[:,kgc_center:kend,jgc+1:jhend,igc:iend] - vc[:,kgc_center:kend,jgc:jhend-1,igc:iend]) * dyi
    diss_tau_yv_smag = - diss_tau_yv_smag * (vc[:,kgc_center:kend,jgc+1:jhend,igc:iend] - vc[:,kgc_center:kend,jgc:jhend-1,igc:iend]) * dyi
    diss_tau_yv_lbl = - diss_tau_yv_lbl * (vc[:,kgc_center:kend,jgc+1:jhend,igc:iend] - vc[:,kgc_center:kend,jgc:jhend-1,igc:iend]) * dyi
    #
    diss_tot_MLP[:,:,:,:]  += diss_tau_yv_MLP[:,:,:,:]
    diss_tot_smag[:,:,:,:] += diss_tau_yv_smag[:,:,:,:]
    diss_tot_lbl[:,:,:,:]  += diss_tau_yv_lbl[:,:,:,:]
    
    #zv
    diss_tau_zv_MLP = - diss_tau_zv_MLP * (vc[:,kgc_center:kend+1,jgc:jhend,igc:iend] - vc[:,kgc_center-1:kend,jgc:jhend,igc:iend]) * dzi
    diss_tau_zv_smag = - diss_tau_zv_smag * (vc[:,kgc_center:kend+1,jgc:jhend,igc:iend] - vc[:,kgc_center-1:kend,jgc:jhend,igc:iend]) * dzi
    diss_tau_zv_lbl = - diss_tau_zv_lbl * (vc[:,kgc_center:kend+1,jgc:jhend,igc:iend] - vc[:,kgc_center-1:kend,jgc:jhend,igc:iend]) * dzi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_zv_MLP[:,:-1,:-1,:] + diss_tau_zv_MLP[:,:-1,1:,:] + diss_tau_zv_MLP[:,1:,:-1,:]  + diss_tau_zv_MLP[:,1:,1:,:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_zv_smag[:,:-1,:-1,:] + diss_tau_zv_smag[:,:-1,1:,:] + diss_tau_zv_smag[:,1:,:-1,:]  + diss_tau_zv_smag[:,1:,1:,:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_zv_lbl[:,:-1,:-1,:] + diss_tau_zv_lbl[:,:-1,1:,:] + diss_tau_zv_lbl[:,1:,:-1,:]  + diss_tau_zv_lbl[:,1:,1:,:])
    
    #xw
    diss_tau_xw_MLP = - diss_tau_xw_MLP * (wc[:,kgc_edge:khend,jgc:jend,igc:iend+1] - wc[:,kgc_edge:khend,jgc:jend,igc-1:iend]) * dxi
    diss_tau_xw_smag = - diss_tau_xw_smag * (wc[:,kgc_edge:khend,jgc:jend,igc:iend+1] - wc[:,kgc_edge:khend,jgc:jend,igc-1:iend]) * dxi
    diss_tau_xw_lbl = - diss_tau_xw_lbl * (wc[:,kgc_edge:khend,jgc:jend,igc:iend+1] - wc[:,kgc_edge:khend,jgc:jend,igc-1:iend]) * dxi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_xw_MLP[:,:-1,:,:-1] + diss_tau_xw_MLP[:,:-1,:,1:] + diss_tau_xw_MLP[:,1:,:,:-1]  + diss_tau_xw_MLP[:,1:,:,1:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_xw_smag[:,:-1,:,:-1] + diss_tau_xw_smag[:,:-1,:,1:] + diss_tau_xw_smag[:,1:,:,:-1]  + diss_tau_xw_smag[:,1:,:,1:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_xw_lbl[:,:-1,:,:-1] + diss_tau_xw_lbl[:,:-1,:,1:] + diss_tau_xw_lbl[:,1:,:,:-1]  + diss_tau_xw_lbl[:,1:,:,1:])
    
    #yw
    diss_tau_yw_MLP = - diss_tau_yw_MLP * (wc[:,kgc_edge:khend,jgc:jend+1,igc:iend] - wc[:,kgc_edge:khend,jgc-1:jend,igc:iend]) * dyi
    diss_tau_yw_smag = - diss_tau_yw_smag * (wc[:,kgc_edge:khend,jgc:jend+1,igc:iend] - wc[:,kgc_edge:khend,jgc-1:jend,igc:iend]) * dyi
    diss_tau_yw_lbl = - diss_tau_yw_lbl * (wc[:,kgc_edge:khend,jgc:jend+1,igc:iend] - wc[:,kgc_edge:khend,jgc-1:jend,igc:iend]) * dyi
    #
    diss_tot_MLP[:,:,:,:]   += 0.25 * (diss_tau_yw_MLP[:,:-1,:-1,:] + diss_tau_yw_MLP[:,:-1,1:,:] + diss_tau_yw_MLP[:,1:,:-1,:]  + diss_tau_yw_MLP[:,1:,1:,:])
    diss_tot_smag[:,:,:,:]  += 0.25 * (diss_tau_yw_smag[:,:-1,:-1,:] + diss_tau_yw_smag[:,:-1,1:,:] + diss_tau_yw_smag[:,1:,:-1,:]  + diss_tau_yw_smag[:,1:,1:,:])
    diss_tot_lbl[:,:,:,:]   += 0.25 * (diss_tau_yw_lbl[:,:-1,:-1,:] + diss_tau_yw_lbl[:,:-1,1:,:] + diss_tau_yw_lbl[:,1:,:-1,:]  + diss_tau_yw_lbl[:,1:,1:,:])
    
    #zw
    diss_tau_zw_MLP = - diss_tau_zw_MLP * (wc[:,kgc_edge+1:khend,jgc:jend,igc:iend] - wc[:,kgc_edge:khend-1,jgc:jend,igc:iend]) * dzi
    diss_tau_zw_smag = - diss_tau_zw_smag * (wc[:,kgc_edge+1:khend,jgc:jend,igc:iend] - wc[:,kgc_edge:khend-1,jgc:jend,igc:iend]) * dzi
    diss_tau_zw_lbl = - diss_tau_zw_lbl * (wc[:,kgc_edge+1:khend,jgc:jend,igc:iend] - wc[:,kgc_edge:khend-1,jgc:jend,igc:iend]) * dzi
    #
    diss_tot_MLP[:,:,:,:]  += diss_tau_zw_MLP[:,:,:,:]
    diss_tot_smag[:,:,:,:] += diss_tau_zw_smag[:,:,:,:]
    diss_tot_lbl[:,:,:,:]  += diss_tau_zw_lbl[:,:,:,:]

    #Calculate horizontally averaged vertical profiles dissipation
    diss_tot_MLP_horavg = np.mean(diss_tot_MLP, axis=(2,3))
    diss_tot_smag_horavg = np.mean(diss_tot_smag, axis=(2,3))
    diss_tot_lbl_horavg = np.mean(diss_tot_lbl, axis=(2,3))

    #Write variables
    var_diss_tau_xu_MLP[:,:,:,:]     =     diss_tau_xu_MLP[:,:,:,:] 
    var_diss_tau_xv_MLP[:,:,:,:]     =     diss_tau_xv_MLP[:,:,:,:]
    var_diss_tau_xw_MLP[:,:,:,:]     =     diss_tau_xw_MLP[:,:,:,:]
    var_diss_tau_yu_MLP[:,:,:,:]     =     diss_tau_yu_MLP[:,:,:,:]
    var_diss_tau_yv_MLP[:,:,:,:]     =     diss_tau_yv_MLP[:,:,:,:]
    var_diss_tau_yw_MLP[:,:,:,:]     =     diss_tau_yw_MLP[:,:,:,:]
    var_diss_tau_zu_MLP[:,:,:,:]     =     diss_tau_zu_MLP[:,:,:,:]
    var_diss_tau_zv_MLP[:,:,:,:]     =     diss_tau_zv_MLP[:,:,:,:]
    var_diss_tau_zw_MLP[:,:,:,:]     =     diss_tau_zw_MLP[:,:,:,:]
    var_diss_tot_MLP[:,:,:,:]        =        diss_tot_MLP[:,:,:,:]
    var_diss_tot_MLP_horavg[:,:]     = diss_tot_MLP_horavg[:,:]
    #
    var_diss_tau_xu_smag[:,:,:,:]     =     diss_tau_xu_smag[:,:,:,:]
    var_diss_tau_xv_smag[:,:,:,:]     =     diss_tau_xv_smag[:,:,:,:]
    var_diss_tau_xw_smag[:,:,:,:]     =     diss_tau_xw_smag[:,:,:,:]
    var_diss_tau_yu_smag[:,:,:,:]     =     diss_tau_yu_smag[:,:,:,:]
    var_diss_tau_yv_smag[:,:,:,:]     =     diss_tau_yv_smag[:,:,:,:]
    var_diss_tau_yw_smag[:,:,:,:]     =     diss_tau_yw_smag[:,:,:,:]
    var_diss_tau_zu_smag[:,:,:,:]     =     diss_tau_zu_smag[:,:,:,:]
    var_diss_tau_zv_smag[:,:,:,:]     =     diss_tau_zv_smag[:,:,:,:]
    var_diss_tau_zw_smag[:,:,:,:]     =     diss_tau_zw_smag[:,:,:,:]
    var_diss_tot_smag[:,:,:,:]        =        diss_tot_smag[:,:,:,:]
    var_diss_tot_smag_horavg[:,:]     = diss_tot_smag_horavg[:,:]
    #
    var_diss_tau_xu_lbl[:,:,:,:]     =     diss_tau_xu_lbl[:,:,:,:]
    var_diss_tau_xv_lbl[:,:,:,:]     =     diss_tau_xv_lbl[:,:,:,:]
    var_diss_tau_xw_lbl[:,:,:,:]     =     diss_tau_xw_lbl[:,:,:,:]
    var_diss_tau_yu_lbl[:,:,:,:]     =     diss_tau_yu_lbl[:,:,:,:]
    var_diss_tau_yv_lbl[:,:,:,:]     =     diss_tau_yv_lbl[:,:,:,:]
    var_diss_tau_yw_lbl[:,:,:,:]     =     diss_tau_yw_lbl[:,:,:,:]
    var_diss_tau_zu_lbl[:,:,:,:]     =     diss_tau_zu_lbl[:,:,:,:]
    var_diss_tau_zv_lbl[:,:,:,:]     =     diss_tau_zv_lbl[:,:,:,:]
    var_diss_tau_zw_lbl[:,:,:,:]     =     diss_tau_zw_lbl[:,:,:,:]
    var_diss_tot_lbl[:,:,:,:]        =        diss_tot_lbl[:,:,:,:]
    var_diss_tot_lbl_horavg[:,:]     = diss_tot_lbl_horavg[:,:]

    #Close nc-files
    fields.close()
    training.close()
    dissipation.close()


#Extract reconstructed fields if plots and/or table have to be made
if args.make_plots or args.make_table:
    print('Start reading variables needed to make plots and/or table.')
    
    fields = nc.Dataset("reconstructed_fields.nc", "r")
    
    #Extract MLP fluxes
    unres_tau_xu_MLP = np.array(fields['unres_tau_xu_MLP'][:,:,:,:])
    unres_tau_xv_MLP = np.array(fields['unres_tau_xv_MLP'][:,:,:,:])
    unres_tau_xw_MLP = np.array(fields['unres_tau_xw_MLP'][:,:,:,:])
    unres_tau_yu_MLP = np.array(fields['unres_tau_yu_MLP'][:,:,:,:])
    unres_tau_yv_MLP = np.array(fields['unres_tau_yv_MLP'][:,:,:,:])
    unres_tau_yw_MLP = np.array(fields['unres_tau_yw_MLP'][:,:,:,:])
    unres_tau_zu_MLP = np.array(fields['unres_tau_zu_MLP'][:,:,:,:])
    unres_tau_zv_MLP = np.array(fields['unres_tau_zv_MLP'][:,:,:,:])
    unres_tau_zw_MLP = np.array(fields['unres_tau_zw_MLP'][:,:,:,:])
    #
    unres_tau_xu_MLP_horavg = np.array(fields['unres_tau_xu_MLP_horavg'][:,:])
    unres_tau_xv_MLP_horavg = np.array(fields['unres_tau_xv_MLP_horavg'][:,:])
    unres_tau_xw_MLP_horavg = np.array(fields['unres_tau_xw_MLP_horavg'][:,:])
    unres_tau_yu_MLP_horavg = np.array(fields['unres_tau_yu_MLP_horavg'][:,:])
    unres_tau_yv_MLP_horavg = np.array(fields['unres_tau_yv_MLP_horavg'][:,:])
    unres_tau_yw_MLP_horavg = np.array(fields['unres_tau_yw_MLP_horavg'][:,:])
    unres_tau_zu_MLP_horavg = np.array(fields['unres_tau_zu_MLP_horavg'][:,:])
    unres_tau_zv_MLP_horavg = np.array(fields['unres_tau_zv_MLP_horavg'][:,:])
    unres_tau_zw_MLP_horavg = np.array(fields['unres_tau_zw_MLP_horavg'][:,:])
    
    #Extract Smagorinsky fluxes
    unres_tau_xu_smag = np.array(fields['unres_tau_xu_smag'][:,:,:,:])
    unres_tau_xv_smag = np.array(fields['unres_tau_xv_smag'][:,:,:,:])
    unres_tau_xw_smag = np.array(fields['unres_tau_xw_smag'][:,:,:,:])
    unres_tau_yu_smag = np.array(fields['unres_tau_yu_smag'][:,:,:,:])
    unres_tau_yv_smag = np.array(fields['unres_tau_yv_smag'][:,:,:,:])
    unres_tau_yw_smag = np.array(fields['unres_tau_yw_smag'][:,:,:,:])
    unres_tau_zu_smag = np.array(fields['unres_tau_zu_smag'][:,:,:,:])
    unres_tau_zv_smag = np.array(fields['unres_tau_zv_smag'][:,:,:,:])
    unres_tau_zw_smag = np.array(fields['unres_tau_zw_smag'][:,:,:,:])
    #
    unres_tau_xu_smag_horavg = np.array(fields['unres_tau_xu_smag_horavg'][:,:])
    unres_tau_xv_smag_horavg = np.array(fields['unres_tau_xv_smag_horavg'][:,:])
    unres_tau_xw_smag_horavg = np.array(fields['unres_tau_xw_smag_horavg'][:,:])
    unres_tau_yu_smag_horavg = np.array(fields['unres_tau_yu_smag_horavg'][:,:])
    unres_tau_yv_smag_horavg = np.array(fields['unres_tau_yv_smag_horavg'][:,:])
    unres_tau_yw_smag_horavg = np.array(fields['unres_tau_yw_smag_horavg'][:,:])
    unres_tau_zu_smag_horavg = np.array(fields['unres_tau_zu_smag_horavg'][:,:])
    unres_tau_zv_smag_horavg = np.array(fields['unres_tau_zv_smag_horavg'][:,:])
    unres_tau_zw_smag_horavg = np.array(fields['unres_tau_zw_smag_horavg'][:,:])
    
    #Extract training fluxes
    #unres_tau_xu_traceless = np.array(fields['unres_tau_xu_traceless'][:,:,:,:])
    #unres_tau_yv_traceless = np.array(fields['unres_tau_yv_traceless'][:,:,:,:])
    #unres_tau_zw_traceless = np.array(fields['unres_tau_zw_traceless'][:,:,:,:])
    #
    unres_tau_xu = np.array(fields['unres_tau_xu'][:,:,:,:])
    unres_tau_xv = np.array(fields['unres_tau_xv'][:,:,:,:])
    unres_tau_xw = np.array(fields['unres_tau_xw'][:,:,:,:])
    unres_tau_yu = np.array(fields['unres_tau_yu'][:,:,:,:])
    unres_tau_yv = np.array(fields['unres_tau_yv'][:,:,:,:])
    unres_tau_yw = np.array(fields['unres_tau_yw'][:,:,:,:])
    unres_tau_zu = np.array(fields['unres_tau_zu'][:,:,:,:])
    unres_tau_zv = np.array(fields['unres_tau_zv'][:,:,:,:])
    unres_tau_zw = np.array(fields['unres_tau_zw'][:,:,:,:])
    #
    #unres_tau_xu_traceless_horavg = np.array(fields['unres_tau_xu_traceless_horavg'][:,:])
    #unres_tau_yv_traceless_horavg = np.array(fields['unres_tau_yv_traceless_horavg'][:,:])
    #unres_tau_zw_traceless_horavg = np.array(fields['unres_tau_zw_traceless_horavg'][:,:])
    #
    unres_tau_xu_horavg = np.array(fields['unres_tau_xu_horavg'][:,:])
    unres_tau_xv_horavg = np.array(fields['unres_tau_xv_horavg'][:,:])
    unres_tau_xw_horavg = np.array(fields['unres_tau_xw_horavg'][:,:])
    unres_tau_yu_horavg = np.array(fields['unres_tau_yu_horavg'][:,:])
    unres_tau_yv_horavg = np.array(fields['unres_tau_yv_horavg'][:,:])
    unres_tau_yw_horavg = np.array(fields['unres_tau_yw_horavg'][:,:])
    unres_tau_zu_horavg = np.array(fields['unres_tau_zu_horavg'][:,:])
    unres_tau_zv_horavg = np.array(fields['unres_tau_zv_horavg'][:,:])
    unres_tau_zw_horavg = np.array(fields['unres_tau_zw_horavg'][:,:])
    
    #Close netCDF-file
    fields.close()

#Load dissipation if correlation table has to be made or dissiplation plots have to created
if args.plot_dissipation or args.make_table:
    
    #Read nc-file with dissipation values
    dissipation_file = nc.Dataset("dissipation.nc", "r")

    #Extract dissipation fluxes
    diss_tot_MLP  = np.array(dissipation_file['diss_tot_MLP'][:,:,:,:])
    diss_tot_smag = np.array(dissipation_file['diss_tot_smag'][:,:,:,:])
    diss_tot_lbl  = np.array(dissipation_file['diss_tot_lbl'][:,:,:,:])
    diss_tot_MLP_horavg   = np.array(dissipation_file['diss_tot_MLP_horavg'][:,:])
    diss_tot_smag_horavg  = np.array(dissipation_file['diss_tot_smag_horavg'][:,:])
    diss_tot_lbl_horavg   = np.array(dissipation_file['diss_tot_lbl_horavg'][:,:])


#Extract all dissipation fields if specified
if args.plot_dissipation:
    pass

#Write all relevant correlation coefficients to a table
if args.make_table:
    print('start making table')
    mpl.rc('text', usetex=False) #Switch off latex usage for generation table
    heights = np.array(zc, dtype=object)
    heights = np.insert(heights,0,'zall_horavg')
    heights = np.insert(heights,0,'zall')
    heights = np.append(heights,'top_wall')
    components = np.array(
                ['diss_ANN','tau_uu_ANN','tau_vu_ANN','tau_wu_ANN',
                'tau_uv_ANN','tau_vv_ANN','tau_wv_ANN',
                'tau_uw_ANN','tau_vw_ANN','tau_ww_ANN',
                'diss_smag','tau_uu_smag','tau_vu_smag','tau_wu_smag',
                'tau_uv_smag','tau_vv_smag','tau_wv_smag',
                'tau_uw_smag','tau_vw_smag','tau_ww_smag'])

    #Define arrays for storage
    corrcoef_diss = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xu = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yu = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zu = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xv = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yv = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zv = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xw = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yw = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zw = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_diss_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xu_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yu_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zu_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xv_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yv_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zv_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_xw_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_yw_smag = np.zeros((nz+3,),dtype=np.float32)
    corrcoef_zw_smag = np.zeros((nz+3,),dtype=np.float32)

    #Consider all heights over all time steps
    corrcoef_diss[0] = np.round(np.corrcoef(diss_tot_MLP.flatten(), diss_tot_lbl.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xu[0] = np.round(np.corrcoef(unres_tau_xu_MLP.flatten(), unres_tau_xu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yu[0] = np.round(np.corrcoef(unres_tau_yu_MLP.flatten(), unres_tau_yu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zu[0] = np.round(np.corrcoef(unres_tau_zu_MLP.flatten(), unres_tau_zu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xv[0] = np.round(np.corrcoef(unres_tau_xv_MLP.flatten(), unres_tau_xv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yv[0] = np.round(np.corrcoef(unres_tau_yv_MLP.flatten(), unres_tau_yv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zv[0] = np.round(np.corrcoef(unres_tau_zv_MLP.flatten(), unres_tau_zv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xw[0] = np.round(np.corrcoef(unres_tau_xw_MLP.flatten(), unres_tau_xw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yw[0] = np.round(np.corrcoef(unres_tau_yw_MLP.flatten(), unres_tau_yw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zw[0] = np.round(np.corrcoef(unres_tau_zw_MLP.flatten(), unres_tau_zw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_diss_smag[0] = np.round(np.corrcoef(diss_tot_smag.flatten(), diss_tot_lbl.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xu_smag[0] = np.round(np.corrcoef(unres_tau_xu_smag.flatten(), unres_tau_xu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yu_smag[0] = np.round(np.corrcoef(unres_tau_yu_smag.flatten(), unres_tau_yu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zu_smag[0] = np.round(np.corrcoef(unres_tau_zu_smag.flatten(), unres_tau_zu.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xv_smag[0] = np.round(np.corrcoef(unres_tau_xv_smag.flatten(), unres_tau_xv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yv_smag[0] = np.round(np.corrcoef(unres_tau_yv_smag.flatten(), unres_tau_yv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zv_smag[0] = np.round(np.corrcoef(unres_tau_zv_smag.flatten(), unres_tau_zv.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xw_smag[0] = np.round(np.corrcoef(unres_tau_xw_smag.flatten(), unres_tau_xw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yw_smag[0] = np.round(np.corrcoef(unres_tau_yw_smag.flatten(), unres_tau_yw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zw_smag[0] = np.round(np.corrcoef(unres_tau_zw_smag.flatten(), unres_tau_zw.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
 
    #Consider all heights, horizontally averaged over all time steps
    corrcoef_diss[1]      = np.round(np.corrcoef(diss_tot_MLP_horavg.flatten(), diss_tot_lbl_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xu[1]      = np.round(np.corrcoef(unres_tau_xu_MLP_horavg.flatten(), unres_tau_xu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yu[1]      = np.round(np.corrcoef(unres_tau_yu_MLP_horavg.flatten(), unres_tau_yu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zu[1]      = np.round(np.corrcoef(unres_tau_zu_MLP_horavg.flatten(), unres_tau_zu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xv[1]      = np.round(np.corrcoef(unres_tau_xv_MLP_horavg.flatten(), unres_tau_xv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yv[1]      = np.round(np.corrcoef(unres_tau_yv_MLP_horavg.flatten(), unres_tau_yv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zv[1]      = np.round(np.corrcoef(unres_tau_zv_MLP_horavg.flatten(), unres_tau_zv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xw[1]      = np.round(np.corrcoef(unres_tau_xw_MLP_horavg.flatten(), unres_tau_xw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yw[1]      = np.round(np.corrcoef(unres_tau_yw_MLP_horavg.flatten(), unres_tau_yw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zw[1]      = np.round(np.corrcoef(unres_tau_zw_MLP_horavg.flatten(), unres_tau_zw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_diss_smag[1] = np.round(np.corrcoef(diss_tot_smag_horavg.flatten(), diss_tot_lbl_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xu_smag[1] = np.round(np.corrcoef(unres_tau_xu_smag_horavg.flatten(), unres_tau_xu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yu_smag[1] = np.round(np.corrcoef(unres_tau_yu_smag_horavg.flatten(), unres_tau_yu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zu_smag[1] = np.round(np.corrcoef(unres_tau_zu_smag_horavg.flatten(), unres_tau_zu_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xv_smag[1] = np.round(np.corrcoef(unres_tau_xv_smag_horavg.flatten(), unres_tau_xv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yv_smag[1] = np.round(np.corrcoef(unres_tau_yv_smag_horavg.flatten(), unres_tau_yv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zv_smag[1] = np.round(np.corrcoef(unres_tau_zv_smag_horavg.flatten(), unres_tau_zv_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_xw_smag[1] = np.round(np.corrcoef(unres_tau_xw_smag_horavg.flatten(), unres_tau_xw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_yw_smag[1] = np.round(np.corrcoef(unres_tau_yw_smag_horavg.flatten(), unres_tau_yw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
    corrcoef_zw_smag[1] = np.round(np.corrcoef(unres_tau_zw_smag_horavg.flatten(), unres_tau_zw_horavg.flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix

    #Consider each individual height
    for k in range(nz+1): #+1 needed to calculate corr_coefs at top wall for appropriate components
        if k == nz: #Ensure only arrays with additional cell for top wall are accessed, put the others to NaN
            corrcoef_diss[k+2] = np.nan
            corrcoef_xu[k+2] = np.nan
            corrcoef_yu[k+2] = np.nan
            corrcoef_zu[k+2] = np.round(np.corrcoef(unres_tau_zu_MLP[:,k,:,:].flatten(), unres_tau_zu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xv[k+2] = np.nan
            corrcoef_yv[k+2] = np.nan
            corrcoef_zv[k+2] = np.round(np.corrcoef(unres_tau_zv_MLP[:,k,:,:].flatten(), unres_tau_zv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xw[k+2] = np.nan
            corrcoef_yw[k+2] = np.nan
            corrcoef_zw[k+2] = np.nan
            corrcoef_diss_smag[k+2] = np.nan
            corrcoef_xu_smag[k+2] = np.nan
            corrcoef_yu_smag[k+2] = np.nan
            corrcoef_zu_smag[k+2] = np.round(np.corrcoef(unres_tau_zu_smag[:,k,:,:].flatten(), unres_tau_zu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xv_smag[k+2] = np.nan
            corrcoef_yv_smag[k+2] = np.nan
            corrcoef_zv_smag[k+2] = np.round(np.corrcoef(unres_tau_zv_smag[:,k,:,:].flatten(), unres_tau_zv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xw_smag[k+2] = np.nan
            corrcoef_yw_smag[k+2] = np.nan
            corrcoef_zw_smag[k+2] = np.nan

        else:
            corrcoef_diss[k+2] = np.round(np.corrcoef(diss_tot_MLP[:,k,:,:].flatten(), diss_tot_lbl[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xu[k+2] = np.round(np.corrcoef(unres_tau_xu_MLP[:,k,:,:].flatten(), unres_tau_xu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yu[k+2] = np.round(np.corrcoef(unres_tau_yu_MLP[:,k,:,:].flatten(), unres_tau_yu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zu[k+2] = np.round(np.corrcoef(unres_tau_zu_MLP[:,k,:,:].flatten(), unres_tau_zu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xv[k+2] = np.round(np.corrcoef(unres_tau_xv_MLP[:,k,:,:].flatten(), unres_tau_xv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yv[k+2] = np.round(np.corrcoef(unres_tau_yv_MLP[:,k,:,:].flatten(), unres_tau_yv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zv[k+2] = np.round(np.corrcoef(unres_tau_zv_MLP[:,k,:,:].flatten(), unres_tau_zv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xw[k+2] = np.round(np.corrcoef(unres_tau_xw_MLP[:,k,:,:].flatten(), unres_tau_xw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yw[k+2] = np.round(np.corrcoef(unres_tau_yw_MLP[:,k,:,:].flatten(), unres_tau_yw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zw[k+2] = np.round(np.corrcoef(unres_tau_zw_MLP[:,k,:,:].flatten(), unres_tau_zw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_diss_smag[k+2] = np.round(np.corrcoef(diss_tot_smag[:,k,:,:].flatten(), diss_tot_lbl[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xu_smag[k+2] = np.round(np.corrcoef(unres_tau_xu_smag[:,k,:,:].flatten(), unres_tau_xu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yu_smag[k+2] = np.round(np.corrcoef(unres_tau_yu_smag[:,k,:,:].flatten(), unres_tau_yu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zu_smag[k+2] = np.round(np.corrcoef(unres_tau_zu_smag[:,k,:,:].flatten(), unres_tau_zu[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xv_smag[k+2] = np.round(np.corrcoef(unres_tau_xv_smag[:,k,:,:].flatten(), unres_tau_xv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yv_smag[k+2] = np.round(np.corrcoef(unres_tau_yv_smag[:,k,:,:].flatten(), unres_tau_yv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zv_smag[k+2] = np.round(np.corrcoef(unres_tau_zv_smag[:,k,:,:].flatten(), unres_tau_zv[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_xw_smag[k+2] = np.round(np.corrcoef(unres_tau_xw_smag[:,k,:,:].flatten(), unres_tau_xw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_yw_smag[k+2] = np.round(np.corrcoef(unres_tau_yw_smag[:,k,:,:].flatten(), unres_tau_yw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix
            corrcoef_zw_smag[k+2] = np.round(np.corrcoef(unres_tau_zw_smag[:,k,:,:].flatten(), unres_tau_zw[:,k,:,:].flatten())[0,1],3) #Calculate, extract, and round off Pearson correlation coefficient from correlation matrix

    #Add correlation coefficients to DataFrame
    corr_coef = np.array(
               [corrcoef_diss,corrcoef_xu,corrcoef_yu,corrcoef_zu,
                corrcoef_xv,corrcoef_yv,corrcoef_zv,
                corrcoef_xw,corrcoef_yw,corrcoef_zw,
                corrcoef_diss_smag,corrcoef_xu_smag,corrcoef_yu_smag,corrcoef_zu_smag,
                corrcoef_xv_smag,corrcoef_yv_smag,corrcoef_zv_smag,
                corrcoef_xw_smag,corrcoef_yw_smag,corrcoef_zw_smag]
               ,dtype=np.float32)
    
    corr_table = pd.DataFrame(np.swapaxes(corr_coef,0,1), index = heights, columns = components)

    #Save table to figure (code taken from StackOverflow)
    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',bbox=[0, 0, 1, 1], header_columns=0,ax=None, row_color_map=None, corr_table = False, **kwargs):

        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height]) #second numpy array found by trial and error: chosen such that the figure exactly fits the table
            fig = plt.figure(figsize=size)
            ax = plt.gca()
            ax.axis('off')

        #mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs) #Uncomment when no conditional formatting should be applied
        #Uncomment five lines below when conditional formatting should be applied
        if corr_table:
            normal = np.minimum(np.maximum(data, 0.),1.) #Scale colors in range 0-1
        else:
            normal = np.minimum(np.maximum(data, -1.),1.) #Scale colors in range -1 to 1
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, cellColours=plt.cm.jet(normal), **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in  mpl_table._cells.items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
            #    cell.set_facecolor(row_colors[k[0]%len(row_colors)]) #Uncomment to make polished table for paper
                #Do conditional formatting at line 1248
                pass

        #Save figure
        if corr_table:
            fig.savefig('corr_table.pdf', bbox_inches='tight')
        else:
            fig.savefig('re_table.pdf', bbox_inches='tight')

    render_mpl_table(corr_table, header_columns=0, col_width=2.0, bbox=[0.02, 0, 1, 1], corr_table = True)
    
    mpl.rc('text', usetex=True) #Switch latex usage for matplotlib back on

#Define function for making horizontal cross-sections
def make_horcross_heights(values, z, y, x, component, is_lbl, utau_ref_moser, is_smag = False, time_step = 0, delta = 1, plot_dissipation=False):
    #NOTE1: the seventh last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the sixth last input of this function is a boolean that specifies whether the labels (True) or the NN predictions are being plotted.
    #NOTE3: the fifth last input is the friction velocity [m/s], used to normalize
    #NOTE4: the fourth last input of this function is a boolean that specifies whether the Smagorinsky fluxes are plotted (True) or not (False)
    #NOTE5: the thrid last input of this function is an integer specifying which test time step stored in the nc-file is plotted.
    #NOTE6: the second last input of this function is an integer specifying the channel half with [in meter] used to rescale the horizontal dimensions (by default 1m, effectively not rescaling). 
    #NOTE7: the last input of this function is a boolean that specifies whether the SGS fluxes (False) or dissipation values are being plotted.

    #Check that component is not both specified as label and Smagorinsky value
    if is_lbl and is_smag:
        raise RuntimeError("Value specified as both label and Smagorinsky value, which is not possible.")

    for k in range(len(z)-1):
        
        if plot_dissipation:
            values_height = (values[time_step,k,:,:] / (utau_ref_moser ** 3.)) * delta
        else:
            values_height = values[time_step,k,:,:] / (utau_ref_moser ** 2.)

        #Make horizontal cross-sections of the values
        plt.figure()
        if plot_dissipation:
            if is_smag:
                plt.pcolormesh(x / delta, y / delta, values_height, vmin=-150.0, vmax=150.0)
            else:
                plt.pcolormesh(x / delta, y / delta, values_height, vmin=-500.0, vmax=500.0)
        else:
            if is_smag:
                plt.pcolormesh(x / delta, y / delta, values_height, vmin=-0.5, vmax=0.5)
            else:
                plt.pcolormesh(x / delta, y / delta, values_height, vmin=-5.0, vmax=5.0)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        if plot_dissipation:
            if is_lbl:
                cbar.set_label(r'$ \epsilon_{DNS} \ u_{\tau}^{-3} \ \delta \ [-]$',rotation=270,fontsize=20,labelpad=30)
            elif is_smag:
                cbar.set_label(r'$ \epsilon_{Smag} \ u_{\tau}^{-3} \ \delta \ [-]$',rotation=270,fontsize=20,labelpad=30)
            else:
                cbar.set_label(r'$ \epsilon_{ANN} \ u_{\tau}^{-3} \ \delta \ [-]$',rotation=270,fontsize=20,labelpad=30)
        else:
            cbar.set_label(r'$ \tau_{wu} \ u_{\tau}^{-2} \ [-]$',rotation=270,fontsize=20,labelpad=30)
        plt.xlabel(r'$ x \ \delta^{-1} \ [-]$',fontsize=20)
        plt.ylabel(r'$ y \ \delta^{-1} \ [-]$',fontsize=20)
        #plt.xticks(fontsize=16, rotation=90)
        plt.xticks(fontsize=16, rotation=0)
        plt.yticks(fontsize=16, rotation=0)
        fig = plt.gcf()
        fig.set_tight_layout(True)
        if plot_dissipation:
            if is_lbl:
                plt.savefig("Diss_horcross_label_" + str((z[k]+z[k+1])/2.) + ".pdf")
            elif is_smag:
                plt.savefig("Diss_horcross_smag_" + str((z[k]+z[k+1])/2.) + ".pdf")
            else:
                plt.savefig("Diss_horcross_" + str((z[k]+z[k+1])/2.) + ".pdf")
        else:
            if is_lbl:
                plt.savefig("Horcross_label_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".pdf")
            elif is_smag:
                plt.savefig("Horcross_smag_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".pdf")
            else:
                plt.savefig("Horcross_tau_" + component + "_" + str((z[k]+z[k+1])/2.) + ".pdf")
        plt.close()

#Define function for making spectra
def make_spectra_heights(ann, smag, dns, z, component, time_step, delta, utau_ref_moser, domainsize_x = 1, domainsize_y = 1, plot_dissipation=False):
    #NOTE1: the seventh last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the sixth last input of this function is an integer specifying which testing time step stored in the nc-file is plotted
    #NOTE3: the fifth last input of this function is an integer specifying the channel half with [in meter] 
    #NOTE4: the fourth last input is the friction velocity [m/s], used to normalize
    #NOTE5: the third and second last inputs indicate the domain size in the stream- and spanwise directions respectively [in meter].
    #NOTE6: the last input of this function is a boolean that specifies whether the SGS fluxes (False) or dissipation values are being plotted.
    for k in range(len(z)):
        
        ann_height  = ann[time_step,k,:,:]  
        smag_height = smag[time_step,k,:,:] 
        dns_height  = dns[time_step,k,:,:]  

        #Calculate spectra
        #ANN
        nxc = ann_height.shape[1]
        nyc = ann_height.shape[0]
        ann_nwave_modes_x = int(nxc * 0.5)
        ann_nwave_modes_y = int(nyc * 0.5)
        ann_n_streamwise = np.arange(1,ann_nwave_modes_x+1)
        ann_n_spanwise = np.arange(1,ann_nwave_modes_y+1)
        ann_k_streamwise = (ann_n_streamwise / domainsize_x) * 2 * np.pi
        ann_k_spanwise = (ann_n_spanwise / domainsize_y) * 2 * np.pi
        fftx_ann = np.fft.rfft(ann_height,axis=1)*(1/nxc)
        ffty_ann = np.fft.rfft(ann_height,axis=0)*(1/nyc)
        Px_ann = fftx_ann[:,1:] * np.conjugate(fftx_ann[:,1:])
        Py_ann = ffty_ann[1:,:] * np.conjugate(ffty_ann[1:,:])
        if int(nxc % 2) == 0:
            Ex_ann = np.append(2*Px_ann[:,:-1],np.reshape(Px_ann[:,-1],(nyc,1)),axis=1)
        else:
            Ex_ann = 2*Px_ann[:,:]
        
        if int(nyc % 2) == 0:
            Ey_ann = np.append(2*Py_ann[:-1,:],np.reshape(Py_ann[-1,:],(1,nxc)),axis=0)
        else:
            Ey_ann = 2*Py_ann[:,:]

        ann_spec_x = np.nanmean(Ex_ann,axis=0) #Average FT over direction where it was not calculated
        ann_spec_y = np.nanmean(Ey_ann,axis=1)
        #smag
        nxc = smag_height.shape[1]
        nyc = smag_height.shape[0]
        smag_nwave_modes_x = int(nxc * 0.5)
        smag_nwave_modes_y = int(nyc * 0.5)
        smag_n_streamwise = np.arange(1,smag_nwave_modes_x+1)
        smag_n_spanwise = np.arange(1,smag_nwave_modes_y+1)
        smag_k_streamwise = (smag_n_streamwise / domainsize_x) * 2 * np.pi
        smag_k_spanwise = (smag_n_spanwise / domainsize_y) * 2 * np.pi
        fftx_smag = np.fft.rfft(smag_height,axis=1)*(1/nxc)
        ffty_smag = np.fft.rfft(smag_height,axis=0)*(1/nyc)
        Px_smag = fftx_smag[:,1:] * np.conjugate(fftx_smag[:,1:])
        Py_smag = ffty_smag[1:,:] * np.conjugate(ffty_smag[1:,:])
        if int(nxc % 2) == 0:
            Ex_smag = np.append(2*Px_smag[:,:-1],np.reshape(Px_smag[:,-1],(nyc,1)),axis=1)
        else:
            Ex_smag = 2*Px_smag[:,:]
        
        if int(nyc % 2) == 0:
            Ey_smag = np.append(2*Py_smag[:-1,:],np.reshape(Py_smag[-1,:],(1,nxc)),axis=0)
        else:
            Ey_smag = 2*Py_smag[:,:]

        smag_spec_x = np.nanmean(Ex_smag,axis=0) #Average FT over direction where it was not calculated
        smag_spec_y = np.nanmean(Ey_smag,axis=1)
        #DNS
        nxc = dns_height.shape[1]
        nyc = dns_height.shape[0]
        dns_nwave_modes_x = int(nxc * 0.5)
        dns_nwave_modes_y = int(nyc * 0.5)
        dns_n_streamwise = np.arange(1,dns_nwave_modes_x+1)
        dns_n_spanwise = np.arange(1,dns_nwave_modes_y+1)
        dns_k_streamwise = (dns_n_streamwise / domainsize_x) * 2 * np.pi
        dns_k_spanwise = (dns_n_spanwise / domainsize_y) * 2 * np.pi
        fftx_dns = np.fft.rfft(dns_height,axis=1)*(1/nxc)
        ffty_dns = np.fft.rfft(dns_height,axis=0)*(1/nyc)
        Px_dns = fftx_dns[:,1:] * np.conjugate(fftx_dns[:,1:])
        Py_dns = ffty_dns[1:,:] * np.conjugate(ffty_dns[1:,:])
        if int(nxc % 2) == 0:
            Ex_dns = np.append(2*Px_dns[:,:-1],np.reshape(Px_dns[:,-1],(nyc,1)),axis=1)
        else:
            Ex_dns = 2*Px_dns[:,:]
        
        if int(nyc % 2) == 0:
            Ey_dns = np.append(2*Py_dns[:-1,:],np.reshape(Py_dns[-1,:],(1,nxc)),axis=0)
        else:
            Ey_dns = 2*Py_dns[:,:]

        dns_spec_x = np.nanmean(Ex_dns,axis=0) #Average FT over direction where it was not calculated
        dns_spec_y = np.nanmean(Ey_dns,axis=1)

        #Plot spectra
        plt.figure()
        if plot_dissipation:
            plt.plot(ann_k_streamwise  * delta, ann_spec_x / (utau_ref_moser ** 3.),  label = 'ANN')
            plt.plot(smag_k_streamwise  * delta, smag_spec_x / (utau_ref_moser ** 3.), label = 'Smagorinsky')
            plt.plot(dns_k_streamwise * delta, dns_spec_x / (utau_ref_moser ** 3.),  label = 'DNS')
        else:
            plt.plot(ann_k_streamwise  * delta, ann_spec_x / (utau_ref_moser ** 2. * delta),  label = 'ANN')
            plt.plot(smag_k_streamwise  * delta, smag_spec_x / (utau_ref_moser ** 2. * delta), label = 'Smagorinsky')
            plt.plot(dns_k_streamwise * delta, dns_spec_x / (utau_ref_moser ** 2. * delta),  label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        if plot_dissipation:
            plt.ylabel(r'$ E_{\epsilon} \ u_{\tau}^{-3} \ [-]$',fontsize=20)
        else:
            plt.ylabel(r'$ E_{\tau_{wu}} \ u_{\tau}^{-2} \ \delta^{-1} \ [-]$',fontsize=20)
        plt.xlabel(r'$\kappa \delta \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper left')
        fig = plt.gcf()
        fig.set_tight_layout(True)
        if plot_dissipation:
            plt.savefig("Diss_spectrax_" + str(z[k]) + ".pdf")
        else:
            plt.savefig("Spectrax_tau_" + component + "_" + str(z[k]) + ".pdf")
        plt.close()
        #
        plt.figure()
        if plot_dissipation:
            plt.plot(ann_k_spanwise * delta, ann_spec_y / (utau_ref_moser ** 3.),  label = 'ANN')
            plt.plot(smag_k_spanwise * delta, smag_spec_y / (utau_ref_moser ** 3.), label = 'Smagorinsky')
            plt.plot(dns_k_spanwise * delta, dns_spec_y / (utau_ref_moser ** 3.),  label = 'DNS')
        else:
            plt.plot(ann_k_spanwise * delta, ann_spec_y / (utau_ref_moser ** 2. * delta),  label = 'ANN')
            plt.plot(smag_k_spanwise * delta, smag_spec_y / (utau_ref_moser ** 2. * delta), label = 'Smagorinsky')
            plt.plot(dns_k_spanwise * delta, dns_spec_y / (utau_ref_moser ** 2. * delta),  label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        if plot_dissipation:
            plt.ylabel(r'$ E \ u_{\tau}^{-3} \ [-]$',fontsize=20)
        else:
            plt.ylabel(r'$ E \ u_{\tau}^{-2} \ \delta^{-1} \ [-]$',fontsize=20)
        plt.xlabel(r'$\kappa \delta \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper left')
        fig = plt.gcf()
        fig.set_tight_layout(True)
        if plot_dissipation:
            plt.savefig("Diss_spectray_" + str(z[k]) + ".pdf")
        else:
            plt.savefig("Spectray_tau_" + component + "_" + str(z[k]) + ".pdf")
        plt.close()

#Define function for making pdfs
def make_pdfs_heights(values, smag, labels, z, component, time_step, utau_ref_moser, plot_dissipation=False):
    #NOTE1: the fourth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the third last input of this function is an integer specifying which testing time step stored in the nc-file is plotted
    #NOTE3: the second last input is the friction velocity [m/s], used to normalize
    #NOTE4: the last input of this function is a boolean that specifies whether the SGS fluxes (False) or dissipation values are being plotted.
    for k in range(len(z)+1):
        if k == len(z):
            if plot_dissipation:
                values_height = (values[time_step,:,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
                smag_height   = (smag[time_step,:,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
                labels_height = (labels[time_step,:,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
            else:
                values_height = values[time_step,:,:,:].flatten() / (utau_ref_moser ** 2.)
                smag_height   = smag[time_step,:,:,:].flatten() / (utau_ref_moser ** 2.)
                labels_height = labels[time_step,:,:,:].flatten() / (utau_ref_moser ** 2.)
        else:
            if plot_dissipation:
                values_height = (values[time_step,k,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
                smag_height   = (smag[time_step,k,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
                labels_height = (labels[time_step,k,:,:].flatten() / (utau_ref_moser ** 3.)) * delta
            else:
                values_height = values[time_step,k,:,:].flatten() / (utau_ref_moser ** 2.)
                smag_height   = smag[time_step,k,:,:].flatten() / (utau_ref_moser ** 2.)
                labels_height = labels[time_step,k,:,:].flatten() / (utau_ref_moser ** 2.)

        #Determine bins
        num_bins = 100
        min_val = min(values_height.min(), labels_height.min())
        max_val = max(values_height.max(), labels_height.max())
        bin_edges = np.linspace(min_val, max_val, num_bins)

        #Make pdfs of the values and labels
        plt.figure()
        plt.hist(values_height, bins = bin_edges, density = True, histtype = 'step', label = 'ANN')
        plt.hist(smag_height, bins = bin_edges, density = True, histtype = 'step', label = 'Smagorinsky')
        plt.hist(labels_height, bins = bin_edges, density = True, histtype = 'step', label = 'DNS')
        ax = plt.gca()
        ax.set_yscale('log')
        if plot_dissipation:
            ax.set_ylim(bottom=0.00008)
            ax.set_xlim(left=-800, right=800)
        else:
            ax.set_ylim(bottom=0.008)
            ax.set_xlim(left=-10, right=10)
        plt.ylabel(r'$\rm Probability\ density\ [-]$',fontsize=20)
        if plot_dissipation:
            plt.xlabel(r'$ \epsilon \ u_{\tau}^{-3} \ \delta \ [-]$',fontsize=20)
        else:
            plt.xlabel(r'$ \tau_{wu} \ u_{\tau}^{-2} \ [-]$',fontsize=20)
        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=16, rotation=0)
        plt.legend(loc='upper right')
        fig = plt.gcf()
        fig.set_tight_layout(True)
        if k == len(z):
            if plot_dissipation:
                plt.savefig("Diss_PDF.pdf")
            else:
                plt.savefig("PDF_tau_" + component + ".pdf")
        else:
            if plot_dissipation:
                plt.savefig("Diss_PDF_" + str(z[k]) + ".pdf")
            else:
                plt.savefig("PDF_tau_" + component + "_" + str(z[k]) + ".pdf")
        plt.close()

#Define function for making pdfs
def make_vertprof(values, smag, labels, z, component, time_step, delta, utau_ref_moser, plot_dissipation=
        False):
    #NOTE1: the fifth last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the fourth last input of this function is an integer specifying which testing time step stored in the nc-file is plotted.
    #NOTE3: the third last input of this function is an integer specifying the channel half with [in meter]
    #NOTE4: the second input is the friction velocity [m/s], used to normalize
    #NOTE5: the last input of this function is a boolean that specifies whether the SGS fluxes (False) or dissipation values are being plotted.

    #Make vertical profile
    plt.figure()
    if plot_dissipation:
        plt.plot(z / delta, (values[time_step,:] / (utau_ref_moser ** 3.)) * delta, label = 'ANN', marker = 'o', markersize = 2.0)
        plt.plot(z / delta, (smag[time_step,:] / (utau_ref_moser ** 3.)) * delta, label = 'Smagorinsky', marker = 'o', markersize = 2.0)
        plt.plot(z / delta, (labels[time_step,:] / (utau_ref_moser ** 3.)) * delta, label = 'DNS', marker = 'o', markersize = 2.0)
        plt.ylabel(r'$ \epsilon \ u_{\tau}^{-3} \ \delta \ [-]$',fontsize=20)
    else:
        plt.plot(z / delta, values[time_step,:] / (utau_ref_moser ** 2.), label = 'ANN', marker = 'o', markersize = 2.0)
        plt.plot(z / delta, smag[time_step,:] / (utau_ref_moser ** 2.), label = 'Smagorinsky', marker = 'o', markersize = 2.0)
        plt.plot(z / delta, labels[time_step,:] / (utau_ref_moser ** 2.), label = 'DNS', marker = 'o', markersize = 2.0)
        plt.ylabel(r'$ \tau_{wu} \ u_{\tau}^{-2} \ [-]$',fontsize=20)
    plt.xlabel(r'$ z \ \delta^{-1} \ [-]$',fontsize=20)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    plt.legend(loc='upper left')
    fig = plt.gcf()
    fig.set_tight_layout(True)
    if plot_dissipation:
        plt.savefig("Diss_vertprof.pdf")
    else:
        plt.savefig("vertprof_tau_" + component + ".pdf")
    plt.close()

#Define function for making scatterplots
def make_scatterplot_heights(preds, lbls, preds_horavg, lbls_horavg, heights, component, is_smag, time_step, utau_ref_moser, plot_dissipation=False):
    #NOTE1: the fith last input of this function is a string indicating the name of the component being plotted.
    #NOTE2: the fourth last input of this function is a boolean that specifies whether the Smagorinsky fluxes are being plotted (True) or the MLP fluxes (False).
    #NOTE3: the third last input of this function is an integer specifying which testing time step stored in the nc-file is plotted.
    #NOTE4: the second last input is the friction velocity [m/s], used to normalize
    #NOTE5: the last input of this function is a boolean that specifies whether the SGS fluxes (False) or dissipation values are being plotted.
    for k in range(len(heights)+1):
        if k == len(heights):
            if plot_dissipation:
                preds_height = (preds_horavg[time_step,:] / (utau_ref_moser ** 3.)) * delta
                lbls_height  = (lbls_horavg[time_step,:] / (utau_ref_moser ** 3.)) * delta
            else:
                preds_height = preds_horavg[time_step,:] / (utau_ref_moser ** 2.)
                lbls_height  = lbls_horavg[time_step,:] / (utau_ref_moser ** 2.)
        else:
            if plot_dissipation:
                preds_height = (preds[time_step,k,:,:] / (utau_ref_moser ** 3.)) * delta
                lbls_height  = (lbls[time_step,k,:,:] / (utau_ref_moser ** 3.)) * delta
            else:
                preds_height = preds[time_step,k,:,:] / (utau_ref_moser ** 2.)
                lbls_height  = lbls[time_step,k,:,:] / (utau_ref_moser ** 2.)
        preds_height = preds_height.flatten()
        lbls_height  = lbls_height.flatten()
        
        #Make scatterplots of Smagorinsky/MLP fluxes versus labels
        plt.figure()
        plt.scatter(lbls_height, preds_height, s=6, marker='o', alpha=0.2)
        if k == len(heights):
            if plot_dissipation:
                plt.xlim([-200.0, 200.0])
                plt.ylim([-200.0, 200.0])
            else:
                plt.xlim([-2.0, 2.0])
                plt.ylim([-2.0, 2.0])
        else:
            if plot_dissipation:
                plt.xlim([-1500.0, 1500.0])
                plt.ylim([-1500.0, 1500.0])
            else:
                plt.xlim([-15.0, 15.0])
                plt.ylim([-15.0, 15.0])
        axes = plt.gca()
        plt.plot(axes.get_xlim(),axes.get_ylim(),'b--')
        #plt.gca().set_aspect('equal',adjustable='box')
        if plot_dissipation:
            plt.xlabel(r'$ \epsilon_{DNS} \ u_{\tau}^{-3} \delta \ [-]$',fontsize = 20)
            if is_smag:
                plt.ylabel(r'$ \epsilon_{Smag} \ u_{\tau}^{-3} \delta \ [-]$',fontsize = 20)
            else:
                plt.ylabel(r'$ \epsilon_{ANN} \ u_{\tau}^{-3} \delta \ [-]$',fontsize = 20)
        else:
            plt.xlabel(r'$ \tau_{wu,DNS} \ u_{\tau}^{-2} \ [-]$',fontsize = 20)
            if is_smag:
                plt.ylabel(r'$ \tau_{wu,Smag} \ u_{\tau}^{-2} \ [-]$',fontsize = 20)
            else:
                plt.ylabel(r'$ \tau_{wu,ANN} \ u_{\tau}^{-2} \ [-]$',fontsize = 20)
        #plt.title("ρ = " + str(corrcoef),fontsize = 20)
        plt.axhline(c='black')
        plt.axvline(c='black')
        plt.xticks(fontsize = 16, rotation = 90)
        plt.yticks(fontsize = 16, rotation = 0)
        fig = plt.gcf()
        fig.set_tight_layout(True)
        if is_smag:
            if k == len(heights):
                if plot_dissipation:
                    plt.savefig("Diss_scatter_Smagorinsky_horavg.pdf")
                else:
                    plt.savefig("Scatter_Smagorinsky_tau_" + component + "_horavg.pdf")
            else:
                if plot_dissipation:
                    plt.savefig("Diss_scatter_Smagorinsky_" + str(heights[k]) + ".pdf")
                else:
                    plt.savefig("Scatter_Smagorinsky_tau_" + component + "_" + str(heights[k]) + ".pdf")
        else:
            if k == len(heights):
                if plot_dissipation:
                    plt.savefig("Diss_scatter_horavg.pdf")
                else:
                    plt.savefig("Scatter_tau_" + component + "_horavg.pdf")
            else:
                if plot_dissipation:
                    plt.savefig("Diss_scatter_" + str(heights[k]) + ".pdf")
                else:
                    plt.savefig("Scatter_tau_" + component + "_" + str(heights[k]) + ".pdf")

        plt.close()

#Call function multiple times to make all plots for smagorinsky and MLP (currently selected time step: 29, corresponds to first time step (0) in test set)
if args.make_plots:
    print('start making plots')
    
    #Make spectra of labels and MLP predictions
    make_spectra_heights(unres_tau_xu_MLP, unres_tau_xu_smag, unres_tau_xu, zc,       'xu', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_yu_MLP, unres_tau_yu_smag, unres_tau_yu, zc,       'yu', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_zu_MLP, unres_tau_zu_smag, unres_tau_zu, zhc,      'zu', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_xv_MLP, unres_tau_xv_smag, unres_tau_xv, zc,       'xv', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_yv_MLP, unres_tau_yv_smag, unres_tau_yv, zc,       'yv', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_zv_MLP, unres_tau_zv_smag, unres_tau_zv, zhc,      'zv', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_xw_MLP, unres_tau_xw_smag, unres_tau_xw, zhcless,  'xw', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_yw_MLP, unres_tau_yw_smag, unres_tau_yw, zhcless,  'yw', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    make_spectra_heights(unres_tau_zw_MLP, unres_tau_zw_smag, unres_tau_zw, zc,       'zw', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser)
    
    #Plot vertical profiles
    make_vertprof(unres_tau_xu_MLP_horavg, unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,      'xu', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_yu_MLP_horavg, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,      'yu', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_zu_MLP_horavg, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc,     'zu', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_xv_MLP_horavg, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,      'xv', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_yv_MLP_horavg, unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,      'yv', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_zv_MLP_horavg, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc,     'zv', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_xw_MLP_horavg, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhcless, 'xw', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_yw_MLP_horavg, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhcless, 'yw', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_vertprof(unres_tau_zw_MLP_horavg, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,      'zw', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)

    #Make PDFs of labels and MLP predictions
    make_pdfs_heights(unres_tau_xu_MLP, unres_tau_xu_smag, unres_tau_xu, zc,       'xu', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_yu_MLP, unres_tau_yu_smag, unres_tau_yu, zc,       'yu', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_zu_MLP, unres_tau_zu_smag, unres_tau_zu, zhc,      'zu', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_xv_MLP, unres_tau_xv_smag, unres_tau_xv, zc,       'xv', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_yv_MLP, unres_tau_yv_smag, unres_tau_yv, zc,       'yv', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_zv_MLP, unres_tau_zv_smag, unres_tau_zv, zhc,      'zv', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_xw_MLP, unres_tau_xw_smag, unres_tau_xw, zhcless,  'xw', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_yw_MLP, unres_tau_yw_smag, unres_tau_yw, zhcless,  'yw', time_step = 0, utau_ref_moser = utau_ref_moser)
    make_pdfs_heights(unres_tau_zw_MLP, unres_tau_zw_smag, unres_tau_zw, zc,       'zw', time_step = 0, utau_ref_moser = utau_ref_moser)
    
    #Make horizontal cross-sections
    make_horcross_heights(unres_tau_xu, zhc, yhc, xhc,           'xu', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yu, zhc, ygcextra, xgcextra, 'yu', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zu, zgcextra, yhc, xgcextra, 'zu', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xv, zhc, ygcextra, xgcextra, 'xv', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yv, zhc, yhc, xhc,           'yv', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zv, zgcextra, ygcextra, xhc, 'zv', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xw, zgcextra, yhc, xgcextra, 'xw', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yw, zgcextra, ygcextra, xhc, 'yw', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zw, zhc, yhc, xhc,           'zw', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xu_MLP, zhc, yhc, xhc,           'xu', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yu_MLP, zhc, ygcextra, xgcextra, 'yu', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zu_MLP, zgcextra, yhc, xgcextra, 'zu', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xv_MLP, zhc, ygcextra, xgcextra, 'xv', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yv_MLP, zhc, yhc, xhc,           'yv', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zv_MLP, zgcextra, ygcextra, xhc, 'zv', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xw_MLP, zgcextra, yhc, xgcextra, 'xw', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yw_MLP, zgcextra, ygcextra, xhc, 'yw', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zw_MLP, zhc, yhc, xhc,           'zw', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xu_smag, zhc, yhc, xhc,           'xu', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yu_smag, zhc, ygcextra, xgcextra, 'yu', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zu_smag, zgcextra, yhc, xgcextra, 'zu', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xv_smag, zhc, ygcextra, xgcextra, 'xv', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yv_smag, zhc, yhc, xhc,           'yv', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zv_smag, zgcextra, ygcextra, xhc, 'zv', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_xw_smag, zgcextra, yhc, xgcextra, 'xw', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_yw_smag, zgcextra, ygcextra, xhc, 'yw', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    make_horcross_heights(unres_tau_zw_smag, zhc, yhc, xhc,           'zw', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser)
    
    #Make scatterplots
    make_scatterplot_heights(unres_tau_xu_MLP, unres_tau_xu, unres_tau_xu_MLP_horavg, unres_tau_xu_horavg, zc,  'xu', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yu_MLP, unres_tau_yu, unres_tau_yu_MLP_horavg, unres_tau_yu_horavg, zc,  'yu', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zu_MLP, unres_tau_zu, unres_tau_zu_MLP_horavg, unres_tau_zu_horavg, zhc, 'zu', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_xv_MLP, unres_tau_xv, unres_tau_xv_MLP_horavg, unres_tau_xv_horavg, zc,  'xv', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yv_MLP, unres_tau_yv, unres_tau_yv_MLP_horavg, unres_tau_yv_horavg, zc,  'yv', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zv_MLP, unres_tau_zv, unres_tau_zv_MLP_horavg, unres_tau_zv_horavg, zhc, 'zv', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_xw_MLP, unres_tau_xw, unres_tau_xw_MLP_horavg, unres_tau_xw_horavg, zhcless, 'xw', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yw_MLP, unres_tau_yw, unres_tau_yw_MLP_horavg, unres_tau_yw_horavg, zhcless, 'yw', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zw_MLP, unres_tau_zw, unres_tau_zw_MLP_horavg, unres_tau_zw_horavg, zc,  'zw', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser)
    #
    make_scatterplot_heights(unres_tau_xu_smag, unres_tau_xu, unres_tau_xu_smag_horavg, unres_tau_xu_horavg, zc,  'xu', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yu_smag, unres_tau_yu, unres_tau_yu_smag_horavg, unres_tau_yu_horavg, zc,  'yu', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zu_smag, unres_tau_zu, unres_tau_zu_smag_horavg, unres_tau_zu_horavg, zhc, 'zu', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_xv_smag, unres_tau_xv, unres_tau_xv_smag_horavg, unres_tau_xv_horavg, zc,  'xv', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yv_smag, unres_tau_yv, unres_tau_yv_smag_horavg, unres_tau_yv_horavg, zc,  'yv', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zv_smag, unres_tau_zv, unres_tau_zv_smag_horavg, unres_tau_zv_horavg, zhc, 'zv', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_xw_smag, unres_tau_xw, unres_tau_xw_smag_horavg, unres_tau_xw_horavg, zhcless, 'xw', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_yw_smag, unres_tau_yw, unres_tau_yw_smag_horavg, unres_tau_yw_horavg, zhcless, 'yw', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    make_scatterplot_heights(unres_tau_zw_smag, unres_tau_zw, unres_tau_zw_smag_horavg, unres_tau_zw_horavg, zc,  'zw', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser)
    
    print('Finished making plots')
if args.plot_dissipation:
    print('Start making dissipation plots')
    
    #Make spectra of labels and MLP predictions
    make_spectra_heights(diss_tot_MLP, diss_tot_smag, diss_tot_lbl, zc, 'diss', time_step = 0, delta = delta, domainsize_x = 2 * np.pi, domainsize_y = np.pi, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    
    #Plot vertical profiles
    make_vertprof(diss_tot_MLP_horavg, diss_tot_smag_horavg, diss_tot_lbl_horavg, zc, 'diss', time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    
    #Make PDFs of labels and MLP predictions
    make_pdfs_heights(diss_tot_MLP, diss_tot_smag, diss_tot_lbl, zc, 'diss', time_step = 0, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    
    #Make horizontal cross-sections
    make_horcross_heights(diss_tot_lbl, zhc, yhc, xhc,'diss', True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    make_horcross_heights(diss_tot_MLP, zhc, yhc, xhc,'diss', False, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    make_horcross_heights(diss_tot_smag, zhc, yhc, xhc,'diss', False, is_smag = True, time_step = 0, delta = delta, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    
    #Make scatterplots
    make_scatterplot_heights(diss_tot_MLP, diss_tot_lbl, diss_tot_MLP_horavg, diss_tot_lbl_horavg, zc, 'diss', is_smag = False, time_step = 0, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    #
    make_scatterplot_heights(diss_tot_smag, diss_tot_lbl, diss_tot_smag_horavg, diss_tot_lbl_horavg, zc, 'diss', is_smag = True, time_step = 0, utau_ref_moser = utau_ref_moser, plot_dissipation=True)
    print('Finished making dissipation plots')
