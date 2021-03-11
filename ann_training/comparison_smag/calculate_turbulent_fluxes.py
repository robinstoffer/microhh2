#Script that calculates turbulent fluxes according to Smagorinsky sub-grid model.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc

def calculate_turbulent_fluxes(flowfields_filepath = 'training_data.nc', eddy_diffusivity_filepath = 'eddy_diffusivity.nc', output_filepath = 'smagorinsky_fluxes.nc', zero_w_topbottom = True):
    '''Calculate all nine turbulent transport components of the Reynolds stress tensor for stored flow fields according to the Smagorinsky sub-grid model, and store them in a netCDF-file. The inputs are as follows: \\
        -flowfields_filepath: string specifying the filepath of the file where the flow fields needed for the flux calculations are stored.\\
        -eddy_diffusivity_filepath: string specifying the filepath of the file in which the eddy diffusivities for the flux calculations are stored. \\
        NOTE: the filepaths have to refer to netCDF-files created with func_generate_training.py and calculate_eddy_diffusivity.py. \\
        -output_filepath: string specifying the filepath, which should include a name for the netCDF-file created in this script and a directory where it needs to be placed. \\
        -zero_w_topbottom: boolean flag specifying whether the top and bottom boundary are solid (True) or not (False). If true, the fluxes at the bottom and top boundary are consequently set to zero.'''

    #Check types input
    if not isinstance(flowfields_filepath,str):
        raise TypeError("Specified flow fields filepath should be a string.")

    if not isinstance(eddy_diffusivity_filepath,str):
        raise TypeError("Specified eddy diffusivity filepath should be a string.")
    
    if not isinstance(output_filepath,str):
        raise TypeError("Specified output filepath should be a string.")

    if not isinstance(zero_w_topbottom,bool):
        raise TypeError("Specified zero_w_topbottom flag should be a boolean.")

    #Set time range for which it is required to calculate the Smagorinsky fluxes. This time range should correspond to the flow fields reserved for a priori testing
    tstart=28
    tend=31
    nt=tend-tstart
    
    #Fetch grid information from flowfields_filepath
    a = nc.Dataset(flowfields_filepath, 'r')
    igc            = int(a['igc'][:])
    jgc            = int(a['jgc'][:])
    kgc_center     = int(a['kgc_center'][:])
    kgc_edge       = int(a['kgc_edge'][:])
    ihend          = int(a['ihend'][:])
    iend           = int(a['iend'][:])
    jhend          = int(a['jhend'][:])
    jend           = int(a['jend'][:])
    khend          = int(a['khend'][:])
    kend           = int(a['kend'][:])
    cells_around_centercell = int(a['cells_around_centercell'][:])
    
    #Raise error if kgc_center and kgc_edge are not equal; script not designed for this scenario.
    if kgc_center != kgc_center:
        raise RuntimeError("The number of ghostcells in the vertical are not equal for the centered and staggered variables. This script does not take such a difference into account. Please revise the script if necessary.")

    zhgc = np.array(a['zhgc'][:])
    zgc  = np.array(a['zgc'][:])
    yhgc = np.array(a['yhgc'][:])
    ygc  = np.array(a['ygc'][:])
    xhgc = np.array(a['xhgc'][:])
    xgc  = np.array(a['xgc'][:])

    zhc = np.array(a['zhc'][:])
    zc  = np.array(a['zc'][:])
    yhc = np.array(a['yhc'][:])
    yc  = np.array(a['yc'][:])
    xhc = np.array(a['xhc'][:])
    xc  = np.array(a['xc'][:])

    #Define shapes of output arrays based on stored coordinates
    nz = zc.shape[0]
    ny = yc.shape[0]
    nx = xc.shape[0]

    #Open file with eddy diffusivity
    b = nc.Dataset(eddy_diffusivity_filepath, 'r')

    #Loop over timesteps
    create_file = True
    for t in range(tstart,tend):
        
        #Fetch eddy diffusivity coefficients
        evisc = b["eddy_diffusivity"][t-tstart,:,:,:]
        
        #Open/create netCDF-file for storage
        if create_file:
            turbulentflux_file = nc.Dataset(output_filepath, 'w')
            create_file      = False
            create_variables = True
        else:
            turbulentflux_file = nc.Dataset(output_filepath, 'r+')
            create_variables = False #Don't define variables when netCDF file already exists, because it should already contain those variables.

        #Define arrays for storage
        smag_tau_xu = np.zeros((nz  ,ny  ,nx)  , dtype=float)
        smag_tau_yu = np.zeros((nz  ,ny+1,nx+1), dtype=float)
        smag_tau_zu = np.zeros((nz+1,ny  ,nx+1), dtype=float)
        smag_tau_xv = np.zeros((nz  ,ny+1,nx+1), dtype=float)
        smag_tau_yv = np.zeros((nz  ,ny  ,nx)  , dtype=float)
        smag_tau_zv = np.zeros((nz+1,ny+1,nx)  , dtype=float)
        smag_tau_xw = np.zeros((nz+1,ny  ,nx+1), dtype=float)
        smag_tau_yw = np.zeros((nz+1,ny+1,nx)  , dtype=float)
        smag_tau_zw = np.zeros((nz  ,ny  ,nx)  , dtype=float)
    
        #Fetch flow fields from flowfields file
        u = np.array(a['uc'][:][t,:,:,:])
        v = np.array(a['vc'][:][t,:,:,:])
        w = np.array(a['wc'][:][t,:,:,:])

        #Loop over grid cells to calculate the fluxes
        for k in range(kgc_center,kend):
            dz    = zhgc[k+1]- zhgc[k]
            dzi   = 1./dz
            dzhib = 1./(zgc[k] - zgc[k-1])
            #dzhit = 1./(zgc[k+1] - zgc[k])

            for j in range(jgc,jend):
                dy    = yhgc[j+1]- yhgc[j]
                dyi   = 1./dy
                dyhib = 1./(ygc[j] - ygc[j-1])
                #dyhit = 1./(ygc[j+1] - ygc[j])

                for i in range(igc, iend):
                    dx    = xhgc[i+1]- xhgc[i]
                    dxi   = 1./dx
                    dxhib = 1./(xgc[i] - xgc[i-1])
                    #dxhit = 1./(xgc[i+1] - xgc[i])

                    #Calculate eddy viscosity coefficients on corners
                    eviscnu = 0.25*(evisc[k,j,i-1]   + evisc[k,j,i]   + evisc[k,j+1,i-1] + evisc[k,j+1,i])
                    eviscsu = 0.25*(evisc[k,j-1,i-1] + evisc[k,j-1,i] + evisc[k,j,i-1]   + evisc[k,j,i])
                    evisctu = 0.25*(evisc[k,j,i-1]   + evisc[k,j,i]   + evisc[k+1,j,i-1] + evisc[k+1,j,i])
                    eviscbu = 0.25*(evisc[k-1,j,i-1] + evisc[k-1,j,i] + evisc[k,j,i-1]   + evisc[k,j,i])
                    eviscev = 0.25*(evisc[k,j-1,i]   + evisc[k,j,i]   + evisc[k,j-1,i+1] + evisc[k,j,i+1])
                    eviscwv = 0.25*(evisc[k,j-1,i-1] + evisc[k,j,i-1] + evisc[k,j-1,i]   + evisc[k,j,i])
                    evisctv = 0.25*(evisc[k,j-1,i]   + evisc[k,j,i]   + evisc[k+1,j-1,i] + evisc[k+1,j,i])
                    eviscbv = 0.25*(evisc[k-1,j-1,i] + evisc[k-1,j,i] + evisc[k,j-1,i]   + evisc[k,j,i]) 
                    eviscew = 0.25*(evisc[k-1,j,i]   + evisc[k,j,i]   + evisc[k-1,j,i+1] + evisc[k,j,i+1])
                    eviscww = 0.25*(evisc[k-1,j,i-1] + evisc[k,j,i-1] + evisc[k-1,j,i]   + evisc[k,j,i])
                    eviscnw = 0.25*(evisc[k-1,j,i]   + evisc[k,j,i]   + evisc[k-1,j+1,i] + evisc[k,j+1,i])
                    eviscsw = 0.25*(evisc[k-1,j-1,i] + evisc[k,j-1,i] + evisc[k-1,j,i]   + evisc[k,j,i])

                    #Calculate turbulent fluxes accoring to Smagorinsky-Lilly model. NOTE: take into account that the Smagorinsky fluxes do not contain ghost cells
                    smag_tau_xu[k-kgc_center,j-jgc,i-igc] = -2. * evisc[k,j,i]  * (u[k,j,i+1] - u[k,j,i]) * dxi
                    smag_tau_xv[k-kgc_center,j-jgc,i-igc] = -1. * eviscsu * ((u[k,j,i] - u[k,j-1,i]) * dyhib + (v[k,j,i] - v[k,j,i-1]) * dxhib)
                    smag_tau_xw[k-kgc_center,j-jgc,i-igc] = -1. * eviscbu * ((u[k,j,i] - u[k-1,j,i]) * dzhib + (w[k,j,i] - w[k,j,i-1]) * dxhib)
                    smag_tau_yu[k-kgc_center,j-jgc,i-igc] = -1. * eviscwv * ((v[k,j,i] - v[k,j,i-1]) * dxhib + (u[k,j,i] - u[k,j-1,i]) * dyhib)
                    smag_tau_yv[k-kgc_center,j-jgc,i-igc] = -2. * evisc[k,j,i] * (v[k,j+1,i] - v[k,j,i]) * dyi
                    smag_tau_yw[k-kgc_center,j-jgc,i-igc] = -1. * eviscbv * ((v[k,j,i] - v[k-1,j,i]) * dzhib + (w[k,j,i] - w[k,j-1,i]) * dyhib)
                    smag_tau_zu[k-kgc_center,j-jgc,i-igc] = -1. * eviscww * ((w[k,j,i] - w[k,j,i-1]) * dxhib + (u[k,j,i] - u[k-1,j,i]) * dzhib)
                    smag_tau_zv[k-kgc_center,j-jgc,i-igc] = -1. * eviscsw * ((w[k,j,i] - w[k,j-1,i]) * dyhib + (v[k,j,i] - v[k-1,j,i]) * dzhib)
                    smag_tau_zw[k-kgc_center,j-jgc,i-igc] = -2. * evisc[k,j,i] * (w[k+1,j,i] - w[k,j,i]) * dzi
                    
        #If there is no flux at the bottom/top boundaries (i.e. when zero_w_topbottom = True), the fluxes located at the bottom and top are set to 0.
        if zero_w_topbottom:

            #Set fluxes at bottom boundary to 0 (the calculations below ensure that the fluxes are also set to 0 at the top boundary).
            smag_tau_zu[0,:,:] = 0.
            smag_tau_zv[0,:,:] = 0.
            smag_tau_xw[0,:,:] = 0.
            smag_tau_yw[0,:,:] = 0.
        
        #Add one top/downstream cell to fluxes in the directions where they are located on the grid edges, assuming they are the same as the one at the bottom/upstream edge of the domain.
        #NOTE: this should work when the horizontal directions have periodic BCs, and the vertical direction has a no-slip BC.
        #z-direction
        smag_tau_zu[-1,:,:] = smag_tau_zu[0,:,:]
        smag_tau_zv[-1,:,:] = smag_tau_zv[0,:,:]
        smag_tau_xw[-1,:,:] = smag_tau_xw[0,:,:]
        smag_tau_yw[-1,:,:] = smag_tau_yw[0,:,:]

        #y-direction
        smag_tau_yu[:,-1,:] = smag_tau_yu[:,0,:]
        smag_tau_xv[:,-1,:] = smag_tau_xv[:,0,:]
        smag_tau_zv[:,-1,:] = smag_tau_zv[:,0,:]
        smag_tau_yw[:,-1,:] = smag_tau_yw[:,0,:]

        #x-direction
        smag_tau_yu[:,:,-1] = smag_tau_yu[:,:,0]
        smag_tau_zu[:,:,-1] = smag_tau_zu[:,:,0]
        smag_tau_xv[:,:,-1] = smag_tau_xv[:,:,0]
        smag_tau_xw[:,:,-1] = smag_tau_xw[:,:,0]

        #Store calculated values in nc-file
        if create_variables:
            
            #Create new dimensions
            turbulentflux_file.createDimension("nt",nt)
            turbulentflux_file.createDimension("zhc",len(zhc))
            turbulentflux_file.createDimension("zc",len(zc))
            turbulentflux_file.createDimension("yhc",len(yhc))
            turbulentflux_file.createDimension("yc",len(yc))
            turbulentflux_file.createDimension("xhc",len(xhc))
            turbulentflux_file.createDimension("xc",len(xc))

            #Create new variables
            varsmag_tau_xu = turbulentflux_file.createVariable("smag_tau_xu","f8",("nt","zc","yc","xc"))
            varsmag_tau_yu = turbulentflux_file.createVariable("smag_tau_yu","f8",("nt","zc","yhc","xhc"))
            varsmag_tau_zu = turbulentflux_file.createVariable("smag_tau_zu","f8",("nt","zhc","yc","xhc"))
            varsmag_tau_xv = turbulentflux_file.createVariable("smag_tau_xv","f8",("nt","zc","yhc","xhc"))
            varsmag_tau_yv = turbulentflux_file.createVariable("smag_tau_yv","f8",("nt","zc","yc","xc"))
            varsmag_tau_zv = turbulentflux_file.createVariable("smag_tau_zv","f8",("nt","zhc","yhc","xc"))
            varsmag_tau_xw = turbulentflux_file.createVariable("smag_tau_xw","f8",("nt","zhc","yc","xhc"))
            varsmag_tau_yw = turbulentflux_file.createVariable("smag_tau_yw","f8",("nt","zhc","yhc","xc"))
            varsmag_tau_zw = turbulentflux_file.createVariable("smag_tau_zw","f8",("nt","zc","yc","xc"))
            varzhc         = turbulentflux_file.createVariable("zhc","f8",("zhc"))
            varzc          = turbulentflux_file.createVariable("zc","f8",("zc"))
            varyhc         = turbulentflux_file.createVariable("yhc","f8",("yhc"))
            varyc          = turbulentflux_file.createVariable("yc","f8",("yc"))
            varxhc         = turbulentflux_file.createVariable("xhc","f8",("xhc"))
            varxc          = turbulentflux_file.createVariable("xc","f8",("xc"))

            #Store coordinate variables
            varzhc[:]      = zhc[:]
            varzc[:]       = zc[:]
            varyhc[:]      = yhc[:]
            varyc[:]       = yc[:]
            varxhc[:]      = xhc[:]
            varxc[:]       = xc[:]

            create_variables = False #Make sure the variables are only created once

        #Store variables
        varsmag_tau_xu[t-tstart,:,:,:] = smag_tau_xu[:,:,:]
        varsmag_tau_yu[t-tstart,:,:,:] = smag_tau_yu[:,:,:]
        varsmag_tau_zu[t-tstart,:,:,:] = smag_tau_zu[:,:,:]
        varsmag_tau_xv[t-tstart,:,:,:] = smag_tau_xv[:,:,:]
        varsmag_tau_yv[t-tstart,:,:,:] = smag_tau_yv[:,:,:]
        varsmag_tau_zv[t-tstart,:,:,:] = smag_tau_zv[:,:,:]
        varsmag_tau_xw[t-tstart,:,:,:] = smag_tau_xw[:,:,:]
        varsmag_tau_yw[t-tstart,:,:,:] = smag_tau_yw[:,:,:]
        varsmag_tau_zw[t-tstart,:,:,:] = smag_tau_zw[:,:,:]

        #Close file
        turbulentflux_file.close()

    #Close files
    a.close()
    b.close()
