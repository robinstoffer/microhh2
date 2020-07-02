#Script that generates the training data for the NN
#Author: Robin Stoffer (robin.stoffer@wur.nl)
#NOTE: Developed for Python 3!
import numpy   as np
import netCDF4 as nc
import scipy.interpolate
from downsampling_training import generate_coarsecoord_centercell, generate_coarsecoord_edgecell
from grid_objects_training import Finegrid, Coarsegrid
import matplotlib as mpl
mpl.use('agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
import matplotlib.pyplot as plt
import warnings

###############################################
#Actual functions to generate the training data
###############################################

def generate_training_data(dim_new_grid, input_directory, output_directory, reynolds_number_tau, size_samples = 5, precision = 'double', fourth_order = False, periodic_bc = (False, True, True), zero_w_topbottom = True, name_output_file = 'training_data.nc', create_file = True, testing = False, settings_filepath = None, grid_filepath = 'grid.0000000'): 
    
    ''' Generates the training data required for a supervised machine learning algorithm, making use of the Finegrid and Coarsegrid objects defined in grid_objects_training.py. NOTE: the script  does not yet extract individual samples out of the flow fields; this is done in sample_training_data_tfrecord.py. \\ 
    For each available time step in the MicroHH output, the steps in this script are as follows: 
        1) Read the wind velocity variables and the pressure from the output files produced by MicroHH in the specified input_directory (which should be a string).
        In this procedure, the wind velocities are normalized using a reference friction velocity that is calculated from the specified friction Reynolds number (via the variable reynolds_number_tau, which should be an integer or float).\\
        2) Downsample these variables to a user-defined coarse grid. \\ 
        The dimensions of this new grid are defined via the dim_new_grid variable, which should be a tuple existing of three integers each specifying the amount of grid cells in the z,y,x-directions respectively. \\ 
        The script implicitly assumes that the grid distances are uniform in all three directions, and that the sizes of the domain are the same as in the original grid produced by MicroHH. \\
        3) Interpolate the wind velocities on the fine grid to the grid boundaries of the coarse grid. \\
        4) Use the interpolated wind velocities calculated in step 3 to calculate all nine total turbulent transport components by integrating them over the corresponding grid boundaries. This includes calculating weights (or to be precise, fractions) that compensate for the relative contributions to the total integral. \\
        5) Interpolate the wind velocities on the coarse grid cells to the grid boundaries of the coarse grid. \\
        6) Calculate all nine resolved turbulent transport components by multiplying the corresponding interpolated wind velocities calculated in step 5, and calculate all nine unresolved turbulent transport components by taking the difference between the total and resolved turbulent transport components. \\
        7) Store all variables in a netCDF-file with user-specified name (the variable name_output_file, which should be a string) in user-specified output directory (via the variable output_directory, which should be a string), which serves as input for the script sample_training_data_tfrecord.py. \\
        
        NOTE: In the steps above the script implicitly assumes that the variables are located on a staggered Arakawa C-grid, where the variables located on the grid edges are always upstream/down of the corresponding grid centers (as is the case in MicroHH). \\

   The input variables for this script are: \\
        -dim_new_grid: a tuple existing of three integers, which specifies for each coordinate directions (z,y,x) the amount of grid cells in the user-defined coarse grid. \\
        -input_directory: a string specifying where the MicroHH output binary files are stored. \\
        -output_directory: a string specifying where the created netCDF-file should be stored. \\
        -reynolds_number_tau: an integer or float specifying a reference friction Reynolds number for the flow being considered. \\
        -size_samples: an integer specifying the size of the samples that are eventually extracted in the sample_training_data_tfrecord.py script. This is used to determine the correct amount of ghost cells. \\
        -precision: should be either 'double' or 'single'; specifies the required floating point precision for the calculations. \\
        -fourth_order: a boolean flag specifying the order of the interpolations. When False, second-order interpolation is used. When True, an error is thrown because fourth-order interpolation has not been implemented yet. \\
        -periodic_bc: a tuple existing of three booleans, which specifies for each coordinate direction (z,y,x) whether a periodic bc should be implemented. \\ 
        NOTE1: for the required ghost cells in the horizontal directions this is the only implemented way to include them. Consequently, setting the booleans equal to False in the horizontal directions will result in errors being thrown. \\
        NOTE2: no periodic bc has been implemented in the vertical direction. Consequently, setting the boolean in the vertical direction equal to True will results in an error being thrown. \\
        -zero_w_topbottom: boolean specifying whether the vertical wind velocities is 0 at the bottom and top levels of the domain or not. Since this is currently the only implemented way to include ghost cells in the vertical direction, setting the boolean equal to False will result in an error being thrown. \\
        -name_output_file: string defining the name of the produced output file. \\
        -create_file: boolean specifying whether a new netCDF file should be created or an existing one should be read. \\
        -testing: boolean specifying whether for testing-purposes user-defined input variables and grids should be used. Furthermore, plots of the turbulent transport components are made to check whether the calculated transports are correct. \\
        -settings_filepath: string specifying the filepath of the *.ini file produced by MicroHH containing the settings of the simulations. \\
        -grid_filepath: string specifying the filepath of the *.grid file produced by MicroHH containing information about the simulation grid. \\
        '''

    #Check types input variables (not all of them, some are already checked in grid_objects_training.py.
    if not isinstance(output_directory,str):
        raise TypeError("Specified output directory should be a string.")
        
    if not (isinstance(settings_filepath,str) or (settings_filepath is None)):
        raise TypeError("Specified settings filepath should be a string or NoneType.")

    if not isinstance(grid_filepath,str):
        raise TypeError("Specified grid filepath should be a string.")

    if not (isinstance(reynolds_number_tau,int) or isinstance(reynolds_number_tau,float)):
        raise TypeError("Specified friction Reynolds number should be either a integer or float.")

    if not isinstance(name_output_file,str):
        raise TypeError("Specified name of output file should be a string.")

    if not isinstance(create_file,bool):
        raise TypeError("Specified create_file flag should be a boolean.")

    if not isinstance(testing,bool):
        raise TypeError("Specified testing flag should be a boolean.")

    if not isinstance(settings_filepath,str):
        raise TypeError("Specified settings filepath should be a string.")

    if not isinstance(grid_filepath,str):
        raise TypeError("Specified grid filepath should be a string.")

    #Check that size_samples is an integer, uneven, and larger than 1. This is necessary for the sampling strategy currently implemented
    if not isinstance(size_samples,int):
        raise TypeError("Specified size of samples should be an integer.")

    if (size_samples % 2) == 0 or (size_samples <= 1):
        raise ValueError("Specified size of samples should be uneven and larger than 1 for the sampling strategy currently implemented.")
    
    #Define number of ghost cells in horizontal directions (igc, jgc) based on size_samples
    cells_around_centercell = size_samples // 2 #Use floor division to calculate number of gridcells that are above and below the center gridcell of a given sample, which depends on the sample size. NOTE: sampling itself is done in sample_training_data_tfrecord.py.
    igc = cells_around_centercell
    jgc = cells_around_centercell
    kgc = cells_around_centercell
    
    #Check that at least one ghost cell is present in all directions
    if not (igc > 0 and jgc > 0 and kgc > 0): 
       warnings.warn("There should be at least one ghost cell present in the coarsegrid object for each direction in order to calculate the coarse transports, which is more than strictly required by the chozen size of the samples. The ghost cells are consequently set to 1.")
       igc = 1
       jgc = 1
       kgc = 1
     
    #Define flag to ensure variables are only created once in netCDF file
    create_variables = True
    
    #Initialize finegrid object
    if testing:
        #coordx = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75]) #NOTE: small rounding errors in 12th decimal, in case in 9th decimal
        #xsize = 10.0
        #coordy = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5])
        #ysize = 20.0
        #coordz = np.array([0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0,2.0,3.0,4.0,5.0,5.1,5.15,5.2,5.4,6.0,9.0,9.5,10.0,11.5,12.3,13.0,13.5,13.7,13.8,13.85,13.9,13.95])
        #zsize = 14.0
        #coordz = np.array([0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9,5.1,5.3,5.5,5.7])
        #zsize = 5.8

        #Define similar coordinates as real case
        nz, ny, nx = 256, 384, 768 #Dimensions fine grid
        #xsize = 6.283185307179586
        xsize = 6.
        #ysize = 3.141592653589793
        ysize = 3.
        zsize = 2.
        dx = xsize / nx
        dy = ysize / ny
        coordxh = np.arange(0, xsize, dx)
        coordyh = np.arange(0, ysize, dy)
        coordx = np.arange(dx/2, xsize, dx)
        coordy = np.arange(dy/2, ysize, dy)

        with nc.Dataset("/projects/1/flowsim/simulation1/robin_moser_stats.nc", "r") as nc_file:
            coordz  = nc_file.variables["z" ][:]
            #coordzh = nc_file.variables["zh"][:]
            #Define for test arrays zh such that it is located exactly in between the grid centers, which is NOT the case for zh from the stats file.
            coordzh = np.zeros(len(coordz) + 1)
            for k in range(1, len(coordz)):
                coordzh[k] = coordz[k-1] + 0.5* (coordz[k] - coordz[k-1])
            coordzh[-1] = zsize
        dz = coordzh[1:] - coordzh[:-1]
        
        finegrid = Finegrid(read_grid_flag = False, precision = precision, fourth_order = fourth_order, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom, normalisation_grid = False)
            
        #Define arbritary values for scalars that the script below needs.
        finegrid['fields']['visc'] = 1e-5
        finegrid['grid']['channel_half_width'] = 1.

    else:
        finegrid = Finegrid(precision = precision, fourth_order = fourth_order, periodic_bc = periodic_bc, zero_w_topbottom = zero_w_topbottom, normalisation_grid = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath) #Read settings and grid from .ini files produced by MicroHH, normalize grid with channel half width
 
    #Define orientation on grid for all the variables (z,y,x). True means variable is located on the sides in that direction, false means it is located in the grid centers.
    bool_edge_gridcell_u = (False, False, True)
    bool_edge_gridcell_v = (False, True, False)
    bool_edge_gridcell_w = (True, False, False)
    bool_edge_gridcell_p = (False, False, False)

    #Define reference friction velocity based on viscosity and channel half-width read from the settings files produced by Microhh, and the user-defined Reynolds number
    mvisc = finegrid['fields']['visc']
    channel_half_width = finegrid['grid']['channel_half_width']
    utau_ref = (reynolds_number_tau * mvisc) / channel_half_width
    
    #Define reference kinematic viscosity based on reference friction velocity and channel_half_width
    mvisc_ref = mvisc / (utau_ref * channel_half_width)

    #Loop over timesteps
    for t in range(finegrid['time']['timesteps']): #Only works correctly in this script when whole simulation is saved with a constant time interval. 
        #NOTE1: does not select the last time step stored ('endtime' in {case}.ini file, this would require timesteps + 1 iterations.
        #NOTE2: when testing, the # of timesteps should be set equal to 1.
    #for t in range(1): #FOR TESTING PURPOSES ONLY!
        ##Read or define fine-resolution DNS data ##
        ############################################

        #Read variables from fine resolution data into finegrid or manually define them when testing
        if testing:
            output_shape  = (nz, ny, nx)
#            start_value   = 0
#            end_value     = start_value + output_shape[0]*output_shape[1]*output_shape[2]
#            output_array  = np.reshape(np.arange(start_value,end_value), output_shape)
            
            #Define output_array that has only a gradient in the z-direction.
            output_array  = np.zeros((nz, ny, nx))
            output_1level = np.ones((nz, ny))
            for i in range(1,nx+1):
                output_array[:,:,i-1] = (i ** 1.01)  * output_1level

            finegrid.create_variables('u', output_array, bool_edge_gridcell_u)
            finegrid.create_variables('v', output_array, bool_edge_gridcell_v)
            finegrid.create_variables('w', output_array, bool_edge_gridcell_w)
            finegrid.create_variables('p', output_array, bool_edge_gridcell_p)
            
        else:
            finegrid.read_binary_variables(input_directory, 'u', t, bool_edge_gridcell_u, normalisation_factor = utau_ref)
            finegrid.read_binary_variables(input_directory, 'v', t, bool_edge_gridcell_v, normalisation_factor = utau_ref)
            finegrid.read_binary_variables(input_directory, 'w', t, bool_edge_gridcell_w, normalisation_factor = utau_ref)
            finegrid.read_binary_variables(input_directory, 'p', t, bool_edge_gridcell_p, normalisation_factor = 1.)

        ##Downsample from fine DNS data to user specified coarse grid ##
        ################################################################

        #Initialize coarsegrid object
        coarsegrid = Coarsegrid(dim_new_grid, finegrid, igc = igc, jgc = jgc, kgc = kgc)

        #Calculate representative velocities for each coarse grid cell
        coarsegrid.downsample('u')
        coarsegrid.downsample('v')
        coarsegrid.downsample('w')
        coarsegrid.downsample('p')

        ##Interpolate from fine grid to side boundaries coarse gridcells ##
        ###################################################################

        def _interpolate_side_cell(variable, coord_variable_ghost, coord_boundary):
 
            #Define variables
            zghost, yghost, xghost = coord_variable_ghost
            zbound, ybound, xbound = coord_boundary
 
            #Interpolate to boundary
            z_int = np.ravel(np.broadcast_to(zbound[:,np.newaxis,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            y_int = np.ravel(np.broadcast_to(ybound[np.newaxis,:,np.newaxis],(len(zbound),len(ybound),len(xbound))))
            x_int = np.ravel(np.broadcast_to(xbound[np.newaxis,np.newaxis,:],(len(zbound),len(ybound),len(xbound))))
 
            interpolator = scipy.interpolate.RegularGridInterpolator((zghost, yghost, xghost), variable, method = 'linear', bounds_error = True) #Make sure error is thrown when interpolation outside of domain is required.
            interpolator_value = interpolator(np.transpose((z_int, y_int ,x_int))) #np.transpose added to reshape the arrays corresponding to the shape that is expected.
            var_int = np.reshape(interpolator_value,(len(zbound),len(ybound),len(xbound)))
 
            return var_int

        #NOTE: because the controle volume for the wind velocity componenta differs due to the staggered grid, the transport terms (total and resolved) need to be calculated on different locations in the coarse grid depending on the wind component considered.
        
        #Controle volume u-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the u control volume
        #xz-boundary
        u_uxzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        v_uxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
 
        #yz-boundary
        u_uyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #u_uyzint = np.zeros((u_uyzint_noghost.shape[0], u_uyzint_noghost.shape[1], u_uyzint_noghost.shape[2]+1))
        #u_uyzint[:,:,1:] = u_uyzint_noghost.copy()
        #u_uyzint[:,:,0] = u_uyzint_noghost[:,:,-1]
 
        #xy-boundary
        u_uxyint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        w_uxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        
        #Controle volume v-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the v control volume
        #xz-boundary
        v_vxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],   finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        #v_vxzint = np.zeros((v_vxzint_noghost.shape[0], v_vxzint_noghost.shape[1]+1, v_vxzint_noghost.shape[2]))
        #v_vxzint[:,1:,:] = v_vxzint_noghost.copy()
        #v_vxzint[:,0,:] = v_vxzint_noghost[:,-1,:]
    
        #yz-boundary
        u_vyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        v_vyzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend],         finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #xy-boundary
        v_vxyint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'],  finegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_vxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend],          finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        
        #Controle volume w-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the w control volume
        #xz-boundary
        v_wxzint = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        w_wxzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'], finegrid['grid']['x']),  (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
 
        #yz-boundary
        u_wyzint = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend])) 
        w_wyzint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'],  finegrid['grid']['x']), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend],         finegrid['grid']['y'][finegrid.jgc:finegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
    
        #xy-boundary
        w_wxyint = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'], finegrid['grid']['y'],  finegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend],    finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        #w_wxyint = np.zeros((w_wxyint_noghost.shape[0]+1, w_wxyint_noghost.shape[1], w_wxyint_noghost.shape[2]))
        #w_wxyint[1:,:,:] = w_wxyint_noghost.copy()
        #w_wxyint[0,:,:] = 0 - w_wxyint_noghost[0,:,:]

        #Interpolate each  wind velocities from fine resolution to coarse resolution in all 3 spatial direction (instead of one as above!)
        #NOTE1: this is not needed to calculate the unresolved transport, and is only required to calculate spectra of the isotropic unresolved transport components (which requires u'!)
        #u-velocity
        utot_box_center = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        utot_box_stag   = _interpolate_side_cell(finegrid['output']['u']['variable'], (finegrid['grid']['z'],  finegrid['grid']['y'],  finegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        #v-velocity
        vtot_box_center = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'],  finegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        vtot_box_stag   = _interpolate_side_cell(finegrid['output']['v']['variable'], (finegrid['grid']['z'],  finegrid['grid']['yh'],  finegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #w-velocity
        wtot_box_center = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'],  finegrid['grid']['x']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wtot_box_stag   = _interpolate_side_cell(finegrid['output']['w']['variable'], (finegrid['grid']['zh'],  finegrid['grid']['y'],  finegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))

        ##Calculate TOTAL transport of momentum over xz-, yz-, and xy-boundary for all three wind velocity components. ##
        #################################################################################################################
        
        #NOTE1: the staggered dimensions are 1 unit longer than the 'normal' dimensions. This is taken into account by iterating over len+1 iterations.
        #NOTE2: As a consequence of NOTE1, at len+1 iteration for any given coordinate, only weights have to be known at the grid centers for other two coordinates: only part of the total transport terms need to be calculated (i.e. the terms located on the grid side boundaries for the coordinate in the len+1 iteration). This is ensured by additonal if-statements that evaluate to False at the len+1 iteration.
        #NOTE3: for each transport component both the total turbulence AND viscous contribution is considered!
        #NOTE4: for the isotropic transport components, additional ghost cells are added. This is needed for the sampling applied in sample_training_data_tfrecord.py

        #Initialize first arrays for total transport components and viscous transports on fine grid
        total_tau_xu_turb = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yu_turb = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_zu_turb = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_xv_turb = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yv_turb = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zv_turb = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_xw_turb = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yw_turb = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zw_turb = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
  
        total_tau_xu_visc = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yu_visc = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_zu_visc = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_xv_visc = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yv_visc = np.zeros((coarsegrid['grid']['ktot'],   coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zv_visc = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_xw_visc = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']+1), dtype=float)
        total_tau_yw_visc = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot']+1, coarsegrid['grid']['itot']), dtype=float)
        total_tau_zw_visc = np.zeros((coarsegrid['grid']['ktot']+1, coarsegrid['grid']['jtot'],   coarsegrid['grid']['itot']), dtype=float)
        
        #NOTE: in the arrays below, no additional ghost cells are added to the isotropic components because they comprise intermediate steps in the calculations.
        fine_tau_xu_visc = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        fine_tau_yu_visc = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        fine_tau_zu_visc = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        fine_tau_xv_visc = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        fine_tau_yv_visc = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        fine_tau_zv_visc = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        fine_tau_xw_visc = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot'],   finegrid['grid']['itot']+1), dtype=float)
        fine_tau_yw_visc = np.zeros((finegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        fine_tau_zw_visc = np.zeros((finegrid['grid']['ktot'],   finegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)

        interp_tau_xu_visc = np.zeros((finegrid['grid']['ktot'],     finegrid['grid']['jtot'],     coarsegrid['grid']['itot']), dtype=float)
        interp_tau_yu_visc = np.zeros((finegrid['grid']['ktot'],     coarsegrid['grid']['jtot']+1, finegrid['grid']['itot']+1), dtype=float)
        interp_tau_zu_visc = np.zeros((coarsegrid['grid']['ktot']+1, finegrid['grid']['jtot'],     finegrid['grid']['itot']+1), dtype=float)
        interp_tau_xv_visc = np.zeros((finegrid['grid']['ktot'],     finegrid['grid']['jtot']+1,   coarsegrid['grid']['itot']+1), dtype=float)
        interp_tau_yv_visc = np.zeros((finegrid['grid']['ktot'],     coarsegrid['grid']['jtot'],   finegrid['grid']['itot']), dtype=float)
        interp_tau_zv_visc = np.zeros((coarsegrid['grid']['ktot']+1, finegrid['grid']['jtot']+1,   finegrid['grid']['itot']), dtype=float)
        interp_tau_xw_visc = np.zeros((finegrid['grid']['ktot']+1,   finegrid['grid']['jtot'],     coarsegrid['grid']['itot']+1), dtype=float)
        interp_tau_yw_visc = np.zeros((finegrid['grid']['ktot']+1,   coarsegrid['grid']['jtot']+1, finegrid['grid']['itot']), dtype=float)
        interp_tau_zw_visc = np.zeros((coarsegrid['grid']['ktot'],   finegrid['grid']['jtot'],     finegrid['grid']['itot']), dtype=float)

        #Check that at least one ghost cell is present in all directions, which is needed for the viscous transport calculations below
        if not (finegrid.igc > 0 and finegrid.jgc > 0 and finegrid.kgc_center > 0): 
            raise RuntimeError("There should be at least one ghost cell present in the finegrid object for each direction in order to calculate the viscous transports.")

        #Calculate viscous transport for all grid cells on fine grid
        fine_tau_xu_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc+1:finegrid.ihend] - finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend-1]) / (finegrid['grid']['xh'][finegrid.igc+1:finegrid.ihend] - finegrid['grid']['xh'][finegrid.igc:finegrid.ihend-1])[None,None,:]) #NOTE: indices chosen such that length is one less in staggered direction of velocity component, as is consistent with the location in the grid of the transport component

        fine_tau_yu_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.ihend] - finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc-1:finegrid.jend,finegrid.igc:finegrid.ihend]) / (finegrid['grid']['y'][finegrid.jgc:finegrid.jend+1] - finegrid['grid']['y'][finegrid.jgc-1:finegrid.jend])[None,:,None]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component
        
        fine_tau_zu_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['u']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend] - finegrid['output']['u']['variable'][finegrid.kgc_center-1:finegrid.kend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.ihend]) / (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend+1] - finegrid['grid']['z'][finegrid.kgc_center-1:finegrid.kend])[:,None,None]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component
                  
        fine_tau_xv_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend+1] - finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc-1:finegrid.iend]) / (finegrid['grid']['x'][finegrid.igc:finegrid.iend+1] - finegrid['grid']['x'][finegrid.igc-1:finegrid.iend])[None,None,:]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component

        fine_tau_yv_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc+1:finegrid.jhend,finegrid.igc:finegrid.iend] - finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend,finegrid.jgc:finegrid.jhend-1,finegrid.igc:finegrid.iend]) / (finegrid['grid']['yh'][finegrid.jgc+1:finegrid.jhend] - finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend-1])[None,:,None]) #NOTE: indices chosen such that length is one less in staggered direction of velocity component, as is consistent with the location in the grid of the transport component

        fine_tau_zv_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['v']['variable'][finegrid.kgc_center:finegrid.kend+1,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend] - finegrid['output']['v']['variable'][finegrid.kgc_center-1:finegrid.kend,finegrid.jgc:finegrid.jhend,finegrid.igc:finegrid.iend]) / (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend+1] - finegrid['grid']['z'][finegrid.kgc_center-1:finegrid.kend])[:,None,None]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component
        
        fine_tau_xw_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend+1] - finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc-1:finegrid.iend]) / (finegrid['grid']['x'][finegrid.igc:finegrid.iend+1] - finegrid['grid']['x'][finegrid.igc-1:finegrid.iend])[None,None,:]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component

        fine_tau_yw_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc:finegrid.jend+1,finegrid.igc:finegrid.iend] - finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend,finegrid.jgc-1:finegrid.jend,finegrid.igc:finegrid.iend]) / (finegrid['grid']['y'][finegrid.jgc:finegrid.jend+1] - finegrid['grid']['y'][finegrid.jgc-1:finegrid.jend])[None,:,None]) #NOTE: indices chosen such that length is one more in staggered direction of transport component, as is consistent with the location in the grid of the transport component
        
        fine_tau_zw_visc[:,:,:] = -mvisc_ref * ((finegrid['output']['w']['variable'][finegrid.kgc_edge+1:finegrid.khend,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend] - finegrid['output']['w']['variable'][finegrid.kgc_edge:finegrid.khend-1,finegrid.jgc:finegrid.jend,finegrid.igc:finegrid.iend]) / (finegrid['grid']['zh'][finegrid.kgc_edge+1:finegrid.khend] - finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend-1])[:,None,None]) #NOTE: indices chosen such that length is one less in staggered direction of velocity component, as is consistent with the location in the grid of the transport component

        #Interpolate calculated viscous transports on fine grid to the edges of the control volume on the coarse grid
        interp_tau_xu_visc[:,:,:] = _interpolate_side_cell(fine_tau_xu_visc, (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        
        interp_tau_yu_visc[:,:,:] = _interpolate_side_cell(fine_tau_yu_visc, (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))
        
        interp_tau_zu_visc[:,:,:] = _interpolate_side_cell(fine_tau_zu_visc, (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]))

        interp_tau_xv_visc[:,:,:] = _interpolate_side_cell(fine_tau_xv_visc, (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))

        interp_tau_yv_visc[:,:,:] = _interpolate_side_cell(fine_tau_yv_visc, (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]), (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))

        interp_tau_zv_visc[:,:,:] = _interpolate_side_cell(fine_tau_zv_visc, (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))
        
        interp_tau_xw_visc[:,:,:] = _interpolate_side_cell(fine_tau_xw_visc, (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['xh'][finegrid.igc:finegrid.ihend]), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))

        interp_tau_yw_visc[:,:,:] = _interpolate_side_cell(fine_tau_yw_visc, (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]), (finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))

        interp_tau_zw_visc[:,:,:] = _interpolate_side_cell(fine_tau_zw_visc, (finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], finegrid['grid']['y'][finegrid.jgc:finegrid.jend], finegrid['grid']['x'][finegrid.igc:finegrid.iend]))

        print('Started loop over indices to calculate total transports')
        #Loop over indices coarse grid to calculate integrals for total turbulent AND viscous transport components
        for izc in range(len(coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend])+1):
            
            zcor_c_middle_edge = coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend][izc]
            weights_z_edge, points_indices_z_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], cor_c_middle = zcor_c_middle_edge, dist_corc = coarsegrid['grid']['zhdist'], finegrid = finegrid, periodic_bc = periodic_bc[0], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['zsize'])
            if izc != len(coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend]):
                zcor_c_middle_center = coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend][izc]
                weights_z_center, points_indices_z_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], cor_c_middle = zcor_c_middle_center, dist_corc = coarsegrid['grid']['zdist'], finegrid = finegrid)
 
            for iyc in range(len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])+1):
                
                ycor_c_middle_edge = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend][iyc]
                weights_y_edge, points_indices_y_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['y'][finegrid.jgc:finegrid.jend], cor_c_middle = ycor_c_middle_edge, dist_corc = coarsegrid['grid']['yhdist'], finegrid = finegrid, periodic_bc = periodic_bc[1], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['ysize'])
                if iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]):
                    ycor_c_middle_center = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend][iyc]
                    weights_y_center, points_indices_y_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_c_middle = ycor_c_middle_center, dist_corc = coarsegrid['grid']['ydist'], finegrid = finegrid)

                for ixc in range(len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])+1):
                    
                    xcor_c_middle_edge = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend][ixc]
                    weights_x_edge, points_indices_x_edge = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['x'][finegrid.igc:finegrid.iend], cor_c_middle = xcor_c_middle_edge, dist_corc = coarsegrid['grid']['xhdist'], finegrid = finegrid, periodic_bc = periodic_bc[2], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['xsize'])
                    if ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]):
                        xcor_c_middle_center = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend][ixc]
                        weights_x_center, points_indices_x_center = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_c_middle = xcor_c_middle_center, dist_corc = coarsegrid['grid']['xdist'], finegrid = finegrid)

                    ##Apply 1-dimensional weights calculated above to calculate the total transport terms. This is done by: 1) choosing the correct interpolated velocities calculated before and multiplying those, 2) calculating the corresponding 2-dimensional weight arrays that take the relative contributions to the total integral into account, 3) summing over the multiplied velocities compensated by the 2-dimensional weight arrays, and 4) add the contribution to the total transport from viscous forces. ##

                    #x,y,z: center coarse grid cell
                    #NOTE: +1 added to indices to ensure ghost cells are properly added in the code below.
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend])) and (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])) and (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the z-, y- and x-coordinates.

                        #Contribution turbulence
                        weights_y_center_z_center = weights_y_center[np.newaxis,:] * weights_z_center[:,np.newaxis]
                        total_tau_xu_turb[izc,iyc,ixc+1] = np.sum(weights_y_center_z_center * u_uyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_center] ** 2) 
                        
                        #Contribution viscous forces
                        total_tau_xu_visc[izc,iyc,ixc+1] = np.sum(weights_y_center_z_center * interp_tau_xu_visc[:,:,ixc][points_indices_z_center,:][:,points_indices_y_center])

                        #Contribution turbulence
                        weights_x_center_z_center = weights_x_center[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_yv_turb[izc,iyc+1,ixc] = np.sum(weights_x_center_z_center * v_vxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_center] ** 2) 
                        
                        #Contribution viscous forces
                        total_tau_yv_visc[izc,iyc+1,ixc] = np.sum(weights_x_center_z_center * interp_tau_yv_visc[:,iyc,:][points_indices_z_center,:][:,points_indices_x_center])

                        #Contribution turbulence
                        weights_x_center_y_center = weights_x_center[np.newaxis,:]*weights_y_center[:,np.newaxis]
                        total_tau_zw_turb[izc+1,iyc,ixc] = np.sum(weights_x_center_y_center * w_wxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_center] ** 2)
                        
                        #Contribution viscous forces
                        total_tau_zw_visc[izc+1,iyc,ixc] = np.sum(weights_x_center_y_center * interp_tau_zw_visc[izc,:,:][points_indices_y_center,:][:,points_indices_x_center])

                        
                    #x,y: edge coarse grid cell; z: center coarse grid cell
                    if (izc != len(coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend])): #Make sure this not evaluated for the len+1 iteration in the z-coordinates.
                        
                        #Contribution turbulence
                        weights_y_edge_z_center   = weights_y_edge[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_xv_turb[izc,iyc,ixc] = np.sum(weights_y_edge_z_center * u_vyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_edge] * v_vyzint[:,:,ixc][points_indices_z_center,:][:,points_indices_y_edge])

                        #Contribution viscous forces
                        total_tau_xv_visc[izc,iyc,ixc] = np.sum(weights_y_edge_z_center * interp_tau_xv_visc[:,:,ixc][points_indices_z_center,:][:,points_indices_y_edge])
                    
                        #Contribution turbulence
                        weights_x_edge_z_center   = weights_x_edge[np.newaxis,:]*weights_z_center[:,np.newaxis]
                        total_tau_yu_turb[izc,iyc,ixc] = np.sum(weights_x_edge_z_center * v_uxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_edge] * u_uxzint[:,iyc,:][points_indices_z_center,:][:,points_indices_x_edge])

                        #Contribution viscous forces 
                        total_tau_yu_visc[izc,iyc,ixc] = np.sum(weights_x_edge_z_center * interp_tau_yu_visc[:,iyc,:][points_indices_z_center,:][:,points_indices_x_edge])
                    
                    #x,z: edge coarse grid cell; y:center coarse grid cell
                    if (iyc != len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])): #Make sure this not evaluated for the len+1 iteration in the y-coordinates.
                        
                        #Contribution turbulence
                        weights_y_center_z_edge   = weights_y_center[np.newaxis,:]*weights_z_edge[:,np.newaxis]
                        total_tau_xw_turb[izc,iyc,ixc] = np.sum(weights_y_center_z_edge * u_wyzint[:,:,ixc][points_indices_z_edge,:][:,points_indices_y_center] * w_wyzint[:,:,ixc][points_indices_z_edge,:][:,points_indices_y_center])

                        #Contribution viscous forces
                        total_tau_xw_visc[izc,iyc,ixc] = np.sum(weights_y_center_z_edge * interp_tau_xw_visc[:,:,ixc][points_indices_z_edge,:][:,points_indices_y_center])
                    
                        #Contribution turbulence
                        weights_x_edge_y_center   = weights_x_edge[np.newaxis,:]*weights_y_center[:,np.newaxis]
                        total_tau_zu_turb[izc,iyc,ixc] = np.sum(weights_x_edge_y_center * w_uxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_edge] * u_uxyint[izc,:,:][points_indices_y_center,:][:,points_indices_x_edge])

                        #Contribution viscous forces
                        total_tau_zu_visc[izc,iyc,ixc] = np.sum(weights_x_edge_y_center * interp_tau_zu_visc[izc,:,:][points_indices_y_center,:][:,points_indices_x_edge])

                    #y,z: edge coarse grid cell; x:center coarse grid cell
                    if (ixc != len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])): #Make sure this not evaluated for the len+1 iteration in the x-coordinates.
                        
                        #Contribution turbulence
                        weights_x_center_z_edge   = weights_x_center[np.newaxis,:]*weights_z_edge[:,np.newaxis]
                        total_tau_yw_turb[izc,iyc,ixc] = np.sum(weights_x_center_z_edge * v_wxzint[:,iyc,:][points_indices_z_edge,:][:,points_indices_x_center] * w_wxzint[:,iyc,:][points_indices_z_edge,:][:,points_indices_x_center])
                        
                        #Contribution viscous forces
                        total_tau_yw_visc[izc,iyc,ixc] = np.sum(weights_x_center_z_edge * interp_tau_yw_visc[:,iyc,:][points_indices_z_edge,:][:,points_indices_x_center])
     
                        #Contribution turbulence
                        weights_x_center_y_edge   = weights_x_center[np.newaxis,:]*weights_y_edge[:,np.newaxis]
                        total_tau_zv_turb[izc,iyc,ixc] = np.sum(weights_x_center_y_edge * w_vxyint[izc,:,:][points_indices_y_edge,:][:,points_indices_x_center] * v_vxyint[izc,:,:][points_indices_y_edge,:][:,points_indices_x_center])  

                        #Contribution viscous forces
                        total_tau_zv_visc[izc,iyc,ixc] = np.sum(weights_x_center_y_edge * interp_tau_zv_visc[izc,:,:][points_indices_y_edge,:][:,points_indices_x_center])


        print('Finished loop over indices to calculate total transports')

        ##Add ghostcell to isotropic transport components
        #NOTE1: the ghostcell is added at the beginning of the array, such that the orientation of the transport components is consistent with the resolved transport components calculated below. To allow for this, in the code above 1 is added to the indices of the isotropic components.

        #Vertical direction
        if not (periodic_bc[0] or zero_w_topbottom):
            raise RuntimeError("There is no BC defined in the vertical direction to implement the ghost cell needed for the sampling.")

        if periodic_bc[0] and zero_w_topbottom:
            raise RuntimeError("There are two different BCs defined in the vertical direction. Please select only one of them.")
        

        if periodic_bc[0]:
            total_tau_zw_turb[0,:,:] = total_tau_zw_turb[-1,:,:]
            total_tau_zw_visc[0,:,:] = total_tau_zw_visc[-1,:,:]

        elif zero_w_topbottom:
            total_tau_zw_turb[0,:,:] = -total_tau_zw_turb[1,:,:] #Assume linear flux profile close to surface where U=0
            total_tau_zw_visc[0,:,:] = -total_tau_zw_visc[1,:,:]

        #y-direction
        if not periodic_bc[1]:
            raise RuntimeError("There is no BC defined in the y-direction to implement the ghost cell needed for the sampling.")
        total_tau_yv_turb[:,0,:] = total_tau_yv_turb[:,-1,:]
        total_tau_yv_visc[:,0,:] = total_tau_yv_visc[:,-1,:]

        #x-direction
        if not periodic_bc[2]:
            raise RuntimeError("There is no BC defined in the x-direction to implement the ghost cell needed for the sampling.")
        total_tau_xu_turb[:,:,0] = total_tau_xu_turb[:,:,-1]
        total_tau_xu_visc[:,:,0] = total_tau_xu_visc[:,:,-1]


        ##Interpolate wind velocities on user-defined coarse grid to the corresponding grid boundaries ##
        #################################################################################################
        
        #NOTE: because the controle volume for the wind velocity component differs due to the staggered grid, the transport terms (total and resolved) need to be calculated on different location in the coarse grid depending on the wind component considered.
        
        #Control volume u-momentum
        #NOTE1: all transport terms need to be calculated on the upstream boundaries of the u control volume
        #NOTE2: on purpose for interpolated velocities used for the isotropic components, one additonal ghost cell is included such that the coarse transport labels have this ghost cells included. This is needed for the sampling strategy applied in sample_training_data_tfrecord.py.
        #xz-boundary
        uc_uxzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        vc_uxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #yz-boundary
        uc_uyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['x'][coarsegrid.igc-1:coarsegrid.iend]))
        #uc_uyzint = np.zeros((uc_uyzint_noghost.shape[0], uc_uyzint_noghost.shape[1], uc_uyzint_noghost.shape[2]+1))
        #uc_uyzint[:,:,1:] = uc_uyzint_noghost.copy()
        #uc_uyzint[:,:,0] = uc_uyzint_noghost[:,:,-1]
 
        #xy-boundary
        uc_uxyint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        wc_uxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        
        #Control volume v-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the v control volume
        #xz-boundary
        vc_vxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc-1:coarsegrid.jend],   coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #vc_vxzint = np.zeros((vc_vxzint_noghost.shape[0], vc_vxzint_noghost.shape[1]+1, vc_vxzint_noghost.shape[2]))
        #vc_vxzint[:,1:,:] = vc_vxzint_noghost.copy()
        #vc_vxzint[:,0,:] = vc_vxzint_noghost[:,-1,:]
    
        #yz-boundary
        uc_vyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
        vc_vyzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
 
        #xy-boundary
        vc_vxyint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'],  coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_vxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],   coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],          coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        
        #Control volume w-momentum
        #NOTE: all transport terms need to be calculated on the upstream boundaries of the w control volume
        #xz-boundary
        vc_wxzint = _interpolate_side_cell(coarsegrid['output']['v']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['yh'], coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        wc_wxzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'],  coarsegrid['grid']['y'], coarsegrid['grid']['x']),  (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],         coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend],  coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
 
        #yz-boundary
        uc_wyzint = _interpolate_side_cell(coarsegrid['output']['u']['variable'], (coarsegrid['grid']['z'],  coarsegrid['grid']['y'],  coarsegrid['grid']['xh']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],         coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend])) 
        wc_wyzint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'],  coarsegrid['grid']['y'],  coarsegrid['grid']['x']), (coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend],         coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],          coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]))
    
        #xy-boundary
        #NOTE: At the bottom boundary 1 ghostcell needs to be implemented, which aligns the coordinates such that w_wxyint is located below the center of the control volume. Make use of Dirichlet BC that w = 0.
        wc_wxyint = _interpolate_side_cell(coarsegrid['output']['w']['variable'], (coarsegrid['grid']['zh'], coarsegrid['grid']['y'],  coarsegrid['grid']['x']),  (coarsegrid['grid']['z'][coarsegrid.kgc-1:coarsegrid.kend], coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend],    coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]))
        #wc_wxyint = np.zeros((wc_wxyint_noghost.shape[0]+1, wc_wxyint_noghost.shape[1], wc_wxyint_noghost.shape[2]))
        #wc_wxyint[1:,:,:] = wc_wxyint_noghost.copy()
        #wc_wxyint[0,:,:] = 0 - wc_wxyint_noghost[0,:,:]
        
        ##Calculate resolved and unresolved transport user specified coarse grid ##
        ###########################################################################
        #NOTE1: In the resolved transport terms, the resolved contribution of the viscous forces is considered as well. \
        #This implies that the unresolved contribution of the viscous forces should also be considered (given that the total transport contains as well the total contribution of the viscous forces).

        #NOTE2: For the direction in which the gradient is calculated (in the viscous terms), a grid cell is on purpose added or removed to acquire the correct shape for the resulting array. 
        #This is done by replacing (i,j)(h)end (expacted based on the shape of the arrays themselves) with its (un)staggered counterpart, except for the vertical direction whether 1 is either added or substracted from the last index (this is because in the vertical direction the staggered dimension contains no ghost cells and the unstaggered dimension contains just one ghost cell on each boundary).

        #NOTE3: For the resolved isotropic viscous transport components, one additional ghost cell is added. This is needed for the sampling in sample_training_data_tfrecord.py.

        #xz-boundary

        #Define lengths for broadcasting operations
        len_zhc = len(coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend])
        len_zc  = len(coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend])
        len_yhc = len(coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend])
        len_yc  = len(coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend])
        len_xhc = len(coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend])
        len_xc  = len(coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend])

        #Contribution turbulence
        res_tau_yu_turb = vc_uxzint * uc_uxzint
        
        #Contribution viscous forces
        res_tau_yu_visc = - (mvisc_ref * ((coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc - 1:coarsegrid.jhend - 1,coarsegrid.igc:coarsegrid.ihend]) / np.broadcast_to((coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc:coarsegrid.jhend,np.newaxis] - coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc - 1:coarsegrid.jhend - 1,np.newaxis]),(len_zc,len_yhc,len_xhc))))

        #Contribution turbulence
        res_tau_yv_turb = vc_vxzint ** 2 
        
        #Contribution viscous forces
        res_tau_yv_visc = - (mvisc_ref * ((coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend + 1,coarsegrid.igc:coarsegrid.iend] - coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc-1:coarsegrid.jend,coarsegrid.igc:coarsegrid.iend]) / np.broadcast_to((coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc:coarsegrid.jend + 1,np.newaxis] - coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc-1:coarsegrid.jend,np.newaxis]),(len_zc,len_yc+1,len_xc))))

        #Contribution turbulence
        res_tau_yw_turb = vc_wxzint * wc_wxzint 
        
        #Contribution viscous forces
        res_tau_yw_visc = - (mvisc_ref * ((coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.iend] - coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc - 1:coarsegrid.jhend - 1,coarsegrid.igc:coarsegrid.iend]) / np.broadcast_to((coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc:coarsegrid.jhend,np.newaxis] - coarsegrid['grid']['y'][np.newaxis,coarsegrid.jgc - 1:coarsegrid.jhend - 1,np.newaxis]),(len_zhc,len_yhc,len_xc))))

        #Unresolved transport due to turbulence
        unres_tau_yu_turb = total_tau_yu_turb - res_tau_yu_turb
        unres_tau_yv_turb = total_tau_yv_turb - res_tau_yv_turb
        unres_tau_yw_turb = total_tau_yw_turb - res_tau_yw_turb

        #Unresolved transport due to viscous forces
        unres_tau_yu_visc = total_tau_yu_visc - res_tau_yu_visc
        unres_tau_yv_visc = total_tau_yv_visc - res_tau_yv_visc
        unres_tau_yw_visc = total_tau_yw_visc - res_tau_yw_visc

        #Total unresolved transport
        unres_tau_yu_tot = unres_tau_yu_turb + unres_tau_yu_visc
        unres_tau_yv_tot = unres_tau_yv_turb + unres_tau_yv_visc
        unres_tau_yw_tot = unres_tau_yw_turb + unres_tau_yw_visc

        #yz-boundary        
        res_tau_xu_turb = uc_uyzint ** 2
        
        #Contribution viscous forces
        res_tau_xu_visc = - (mvisc_ref * ((coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.iend + 1] - coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc-1:coarsegrid.iend]) / np.broadcast_to((coarsegrid['grid']['xh'][np.newaxis,np.newaxis,coarsegrid.igc:coarsegrid.iend + 1] - coarsegrid['grid']['xh'][np.newaxis,np.newaxis,coarsegrid.igc-1:coarsegrid.iend]),(len_zc,len_yc,len_xc+1))))

        #Contribution turbulence
        res_tau_xv_turb = uc_vyzint * vc_vyzint
        
        #Contribution viscous forces
        res_tau_xv_visc = - (mvisc_ref * ((coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc - 1:coarsegrid.ihend - 1]) / np.broadcast_to((coarsegrid['grid']['x'][np.newaxis,np.newaxis,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['grid']['x'][np.newaxis,np.newaxis,coarsegrid.igc - 1:coarsegrid.ihend - 1]),(len_zc,len_yhc,len_xhc))))

        #Contribution turbulence
        res_tau_xw_turb = uc_wyzint * wc_wyzint
        
        #Contribution viscous forces
        res_tau_xw_visc = - (mvisc_ref * ((coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc - 1:coarsegrid.ihend - 1]) / np.broadcast_to((coarsegrid['grid']['x'][np.newaxis,np.newaxis,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['grid']['x'][np.newaxis,np.newaxis,coarsegrid.igc - 1:coarsegrid.ihend - 1]),(len_zhc,len_yc,len_xhc))))

        #Unresolved transport due to turbulence
        unres_tau_xu_turb = total_tau_xu_turb - res_tau_xu_turb
        unres_tau_xv_turb = total_tau_xv_turb - res_tau_xv_turb
        unres_tau_xw_turb = total_tau_xw_turb - res_tau_xw_turb

        #Unresolved transport due to viscous forces
        unres_tau_xu_visc = total_tau_xu_visc - res_tau_xu_visc
        unres_tau_xv_visc = total_tau_xv_visc - res_tau_xv_visc
        unres_tau_xw_visc = total_tau_xw_visc - res_tau_xw_visc

        #Total unresolved transport
        unres_tau_xu_tot = unres_tau_xu_turb + unres_tau_xu_visc
        unres_tau_xv_tot = unres_tau_xv_turb + unres_tau_xv_visc
        unres_tau_xw_tot = unres_tau_xw_turb + unres_tau_xw_visc
        
        #xy-boundary
        
        #Contribution turbulence
        res_tau_zu_turb = wc_uxyint * uc_uxyint
        
        #Contribution viscous forces
        res_tau_zu_visc = - (mvisc_ref * ((coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend + 1,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.ihend] - coarsegrid['output']['u']['variable'][coarsegrid.kgc - 1:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.ihend]) / np.broadcast_to((coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend + 1,np.newaxis,np.newaxis] - coarsegrid['grid']['z'][coarsegrid.kgc - 1:coarsegrid.kend,np.newaxis,np.newaxis]),(len_zhc,len_yc,len_xhc))))

        #Contribution turbulence
        res_tau_zv_turb = wc_vxyint * vc_vxyint
        
        #Contribution viscous forces
        res_tau_zv_visc = - (mvisc_ref * ((coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend + 1,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.iend] - coarsegrid['output']['v']['variable'][coarsegrid.kgc - 1:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.iend]) / np.broadcast_to((coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend + 1,np.newaxis,np.newaxis] - coarsegrid['grid']['z'][coarsegrid.kgc - 1:coarsegrid.kend,np.newaxis,np.newaxis]),(len_zhc,len_yhc,len_xc))))

        #Contribution turbulence
        res_tau_zw_turb = wc_wxyint ** 2
        
        #Contribution viscous forces
        res_tau_zw_visc = - (mvisc_ref * ((coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.iend] - coarsegrid['output']['w']['variable'][coarsegrid.kgc-1:coarsegrid.khend - 1,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.iend]) / np.broadcast_to((coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend,np.newaxis,np.newaxis] - coarsegrid['grid']['zh'][coarsegrid.kgc-1:coarsegrid.khend - 1,np.newaxis,np.newaxis]),(len_zc+1,len_yc,len_xc))))

        #Unresolved transport due to turbulence
        unres_tau_zu_turb = total_tau_zu_turb - res_tau_zu_turb
        unres_tau_zv_turb = total_tau_zv_turb - res_tau_zv_turb
        unres_tau_zw_turb = total_tau_zw_turb - res_tau_zw_turb
        
        #Unresolved transport due to viscous forces
        unres_tau_zu_visc = total_tau_zu_visc - res_tau_zu_visc
        unres_tau_zv_visc = total_tau_zv_visc - res_tau_zv_visc
        unres_tau_zw_visc = total_tau_zw_visc - res_tau_zw_visc
       
        #Total unresolved transport
        unres_tau_zu_tot = unres_tau_zu_turb + unres_tau_zu_visc
        unres_tau_zv_tot = unres_tau_zv_turb + unres_tau_zv_visc
        unres_tau_zw_tot = unres_tau_zw_turb + unres_tau_zw_visc
        
        #For each  wind velocity, calculate the unresolved part both with and without contribution from the interpolation error
        #NOTE1: this is not needed to calculate the unresolved transport, and is only required to calculate spectra of the isotropic unresolved transport components (which requires u'!)
        #Only box filter
        unres_u_box = utot_box_stag - coarsegrid['output']['u']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.ihend]
        unres_v_box = vtot_box_stag - coarsegrid['output']['v']['variable'][coarsegrid.kgc:coarsegrid.kend,coarsegrid.jgc:coarsegrid.jhend,coarsegrid.igc:coarsegrid.iend]
        unres_w_box = wtot_box_stag - coarsegrid['output']['w']['variable'][coarsegrid.kgc:coarsegrid.khend,coarsegrid.jgc:coarsegrid.jend,coarsegrid.igc:coarsegrid.iend]
        #Box filter + interpolation error
        unres_u_boxint = utot_box_center - uc_uyzint[:,:,1:]
        unres_v_boxint = vtot_box_center - vc_vxzint[:,1:,:]
        unres_w_boxint = wtot_box_center - wc_wxyint[1:,:,:]


        ##Store flow fields coarse grid and total/resolved/unresolved transports in netCDF-file ##
        ##########################################################################################
 
        #Create/open netCDF file
        if create_file:
            a = nc.Dataset(output_directory + name_output_file, 'w')
            create_file = False
        else:
            a = nc.Dataset(output_directory + name_output_file, 'r+')
 
        if create_variables:
            ##Extract time variable from u-file (should be identical to the one from v-,w-,or p-file)
            #time = np.array(f.variables['time'])
 
            #Create new dimensions
            a.createDimension("time",None)
            a.createDimension("xhgc",coarsegrid['grid']['itot']+2*coarsegrid.igc+1)
            a.createDimension("xgc",coarsegrid['grid']['itot']+2*coarsegrid.igc)
            a.createDimension("yhgc",coarsegrid['grid']['jtot']+2*coarsegrid.jgc+1)
            a.createDimension("ygc",coarsegrid['grid']['jtot']+2*coarsegrid.jgc)
            a.createDimension("zhgc",coarsegrid['grid']['ktot']+2*coarsegrid.kgc+1)
            a.createDimension("zgc",coarsegrid['grid']['ktot']+2*coarsegrid.kgc)
            a.createDimension("xhc",coarsegrid['grid']['itot']+1)
            a.createDimension("xc",coarsegrid['grid']['itot'])
            a.createDimension("xgcextra",coarsegrid['grid']['itot']+1)
            a.createDimension("yhc",coarsegrid['grid']['jtot']+1)
            a.createDimension("yc",coarsegrid['grid']['jtot'])
            a.createDimension("ygcextra",coarsegrid['grid']['jtot']+1)
            a.createDimension("zhc",coarsegrid['grid']['ktot']+1)
            a.createDimension("zc",coarsegrid['grid']['ktot'])
            a.createDimension("zgcextra",coarsegrid['grid']['ktot']+1)
 
            #Create coordinate variables and store values
            var_xhgc        = a.createVariable("xhgc","f8",("xhgc",))
            var_xgc         = a.createVariable("xgc","f8",("xgc",))
            var_yhgc        = a.createVariable("yhgc","f8",("yhgc",))
            var_ygc         = a.createVariable("ygc","f8",("ygc",))
            var_zhgc        = a.createVariable("zhgc","f8",("zhgc",))
            var_zgc         = a.createVariable("zgc","f8",("zgc",))
            var_xhc         = a.createVariable("xhc","f8",("xhc",))
            var_xc          = a.createVariable("xc","f8",("xc",))
            var_xgcextra    = a.createVariable("xgcextra","f8",("xgcextra",))
            var_yhc         = a.createVariable("yhc","f8",("yhc",))
            var_yc          = a.createVariable("yc","f8",("yc",))
            var_ygcextra    = a.createVariable("ygcextra","f8",("ygcextra",))
            var_zhc         = a.createVariable("zhc","f8",("zhc",))
            var_zc          = a.createVariable("zc","f8",("zc",))
            var_zgcextra    = a.createVariable("zgcextra","f8",("zgcextra",))
            var_igc         = a.createVariable("igc","i",())
            var_jgc         = a.createVariable("jgc","i",())
            var_kgc_center  = a.createVariable("kgc_center","i",())
            var_kgc_edge    = a.createVariable("kgc_edge","i",())
            var_iend        = a.createVariable("iend","i",())
            var_ihend       = a.createVariable("ihend","i",())
            var_jend        = a.createVariable("jend","i",())
            var_jhend       = a.createVariable("jhend","i",())
            var_kend        = a.createVariable("kend","i",())
            var_khend       = a.createVariable("khend","i",())
            #
            var_size_samples            = a.createVariable("size_samples","i",())
            var_cells_around_centercell = a.createVariable("cells_around_centercell","i",())
            var_reynolds_number_tau     = a.createVariable("reynolds_number_tau","f8",())
            var_mvisc                   = a.createVariable("mvisc","f8",())
            var_mvisc_ref               = a.createVariable("mvisc_ref","f8",())
            var_channel_half_width      = a.createVariable("channel_half_width","f8",())
            var_utau_ref                = a.createVariable("utau_ref","f8",())
            
            #var_dist_midchannel = a.createVariable("dist_midchannel","f8",("zc",))
 
            var_xhgc[:]       = coarsegrid['grid']['xh']
            var_xgc[:]        = coarsegrid['grid']['x']
            var_yhgc[:]       = coarsegrid['grid']['yh']
            var_ygc[:]        = coarsegrid['grid']['y']
            var_zhgc[:]       = coarsegrid['grid']['zh']
            var_zgc[:]        = coarsegrid['grid']['z']
            var_xhc[:]        = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]
            var_xc[:]         = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]
            var_xgcextra[:]   = coarsegrid['grid']['x'][coarsegrid.igc-1:coarsegrid.iend]
            var_yhc[:]        = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend]
            var_yc[:]         = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]
            var_ygcextra[:]   = coarsegrid['grid']['y'][coarsegrid.jgc-1:coarsegrid.jend]
            var_zhc[:]        = coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend]
            var_zc[:]         = coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend]
            var_zgcextra[:]   = coarsegrid['grid']['z'][coarsegrid.kgc-1:coarsegrid.kend]
            var_igc[:]        = coarsegrid.igc
            var_jgc[:]        = coarsegrid.jgc
            var_kgc_center[:] = coarsegrid.kgc
            var_kgc_edge[:]   = coarsegrid.kgc
            var_iend[:]       = coarsegrid.iend
            var_ihend[:]      = coarsegrid.ihend
            var_jend[:]       = coarsegrid.jend
            var_jhend[:]      = coarsegrid.jhend
            var_kend[:]       = coarsegrid.kend
            var_khend[:]      = coarsegrid.khend
            #
            var_size_samples[:]            = size_samples
            var_cells_around_centercell[:] = cells_around_centercell
            var_reynolds_number_tau[:]     = reynolds_number_tau
            var_mvisc[:]                   = mvisc
            var_mvisc_ref[:]               = mvisc_ref
            var_channel_half_width[:]      = channel_half_width
            var_utau_ref[:]                = utau_ref
            
            #var_dist_midchannel[:] = dist_midchannel[:]
 
            #Create variables for coarse fields
            var_uc = a.createVariable("uc","f8",("time","zgc","ygc","xhgc"))
            var_vc = a.createVariable("vc","f8",("time","zgc","yhgc","xgc"))
            var_wc = a.createVariable("wc","f8",("time","zhgc","ygc","xgc"))
            var_pc = a.createVariable("pc","f8",("time","zgc","ygc","xgc"))
 
            var_res_u_boxint      = a.createVariable("res_u_boxint","f8",("time","zc","yc","xc"))
            var_res_v_boxint      = a.createVariable("res_v_boxint","f8",("time","zc","yc","xc"))
            var_res_w_boxint      = a.createVariable("res_w_boxint","f8",("time","zc","yc","xc"))
            
            var_unres_u_box       = a.createVariable("unres_u_box","f8",("time","zc","yc","xhc"))
            var_unres_v_box       = a.createVariable("unres_v_box","f8",("time","zc","yhc","xc"))
            var_unres_w_box       = a.createVariable("unres_w_box","f8",("time","zhc","yc","xc"))
            
            var_unres_u_boxint    = a.createVariable("unres_u_boxint","f8",("time","zc","yc","xc"))
            var_unres_v_boxint    = a.createVariable("unres_v_boxint","f8",("time","zc","yc","xc"))
            var_unres_w_boxint    = a.createVariable("unres_w_boxint","f8",("time","zc","yc","xc"))
            
            var_total_tau_xu_turb = a.createVariable("total_tau_xu_turb","f8",("time","zc","yc","xgcextra"))
            var_total_tau_xu_visc = a.createVariable("total_tau_xu_visc","f8",("time","zc","yc","xgcextra"))
            var_res_tau_xu_turb   = a.createVariable("res_tau_xu_turb","f8",("time","zc","yc","xgcextra"))
            var_res_tau_xu_visc   = a.createVariable("res_tau_xu_visc","f8",("time","zc","yc","xgcextra"))
            var_unres_tau_xu_tot  = a.createVariable("unres_tau_xu_tot","f8",("time","zc","yc","xgcextra"))
            var_unres_tau_xu_turb = a.createVariable("unres_tau_xu_turb","f8",("time","zc","yc","xgcextra"))
            var_unres_tau_xu_visc = a.createVariable("unres_tau_xu_visc","f8",("time","zc","yc","xgcextra"))

            var_total_tau_xv_turb = a.createVariable("total_tau_xv_turb","f8",("time","zc","yhc","xhc"))
            var_total_tau_xv_visc = a.createVariable("total_tau_xv_visc","f8",("time","zc","yhc","xhc"))
            var_res_tau_xv_turb   = a.createVariable("res_tau_xv_turb","f8",("time","zc","yhc","xhc"))
            var_res_tau_xv_visc   = a.createVariable("res_tau_xv_visc","f8",("time","zc","yhc","xhc"))
            var_unres_tau_xv_tot  = a.createVariable("unres_tau_xv_tot","f8",("time","zc","yhc","xhc"))
            var_unres_tau_xv_turb = a.createVariable("unres_tau_xv_turb","f8",("time","zc","yhc","xhc"))
            var_unres_tau_xv_visc = a.createVariable("unres_tau_xv_visc","f8",("time","zc","yhc","xhc"))

            var_total_tau_xw_turb = a.createVariable("total_tau_xw_turb","f8",("time","zhc","yc","xhc"))
            var_total_tau_xw_visc = a.createVariable("total_tau_xw_visc","f8",("time","zhc","yc","xhc"))
            var_res_tau_xw_turb   = a.createVariable("res_tau_xw_turb","f8",("time","zhc","yc","xhc"))
            var_res_tau_xw_visc   = a.createVariable("res_tau_xw_visc","f8",("time","zhc","yc","xhc"))
            var_unres_tau_xw_tot  = a.createVariable("unres_tau_xw_tot","f8",("time","zhc","yc","xhc"))
            var_unres_tau_xw_turb = a.createVariable("unres_tau_xw_turb","f8",("time","zhc","yc","xhc"))
            var_unres_tau_xw_visc = a.createVariable("unres_tau_xw_visc","f8",("time","zhc","yc","xhc"))
 
            var_total_tau_yu_turb = a.createVariable("total_tau_yu_turb","f8",("time","zc","yhc","xhc"))
            var_total_tau_yu_visc = a.createVariable("total_tau_yu_visc","f8",("time","zc","yhc","xhc"))
            var_res_tau_yu_turb   = a.createVariable("res_tau_yu_turb","f8",("time","zc","yhc","xhc"))
            var_res_tau_yu_visc   = a.createVariable("res_tau_yu_visc","f8",("time","zc","yhc","xhc"))
            var_unres_tau_yu_tot  = a.createVariable("unres_tau_yu_tot","f8",("time","zc","yhc","xhc"))
            var_unres_tau_yu_turb = a.createVariable("unres_tau_yu_turb","f8",("time","zc","yhc","xhc"))
            var_unres_tau_yu_visc = a.createVariable("unres_tau_yu_visc","f8",("time","zc","yhc","xhc"))
 
            var_total_tau_yv_turb = a.createVariable("total_tau_yv_turb","f8",("time","zc","ygcextra","xc"))
            var_total_tau_yv_visc = a.createVariable("total_tau_yv_visc","f8",("time","zc","ygcextra","xc"))
            var_res_tau_yv_turb   = a.createVariable("res_tau_yv_turb","f8",("time","zc","ygcextra","xc"))
            var_res_tau_yv_visc   = a.createVariable("res_tau_yv_visc","f8",("time","zc","ygcextra","xc"))
            var_unres_tau_yv_tot  = a.createVariable("unres_tau_yv_tot","f8",("time","zc","ygcextra","xc"))
            var_unres_tau_yv_turb = a.createVariable("unres_tau_yv_turb","f8",("time","zc","ygcextra","xc"))
            var_unres_tau_yv_visc = a.createVariable("unres_tau_yv_visc","f8",("time","zc","ygcextra","xc"))

            var_total_tau_yw_turb = a.createVariable("total_tau_yw_turb","f8",("time","zhc","yhc","xc"))
            var_total_tau_yw_visc = a.createVariable("total_tau_yw_visc","f8",("time","zhc","yhc","xc"))
            var_res_tau_yw_turb   = a.createVariable("res_tau_yw_turb","f8",("time","zhc","yhc","xc"))
            var_res_tau_yw_visc   = a.createVariable("res_tau_yw_visc","f8",("time","zhc","yhc","xc"))
            var_unres_tau_yw_tot  = a.createVariable("unres_tau_yw_tot","f8",("time","zhc","yhc","xc"))
            var_unres_tau_yw_turb = a.createVariable("unres_tau_yw_turb","f8",("time","zhc","yhc","xc"))
            var_unres_tau_yw_visc = a.createVariable("unres_tau_yw_visc","f8",("time","zhc","yhc","xc"))
 
            var_total_tau_zu_turb = a.createVariable("total_tau_zu_turb","f8",("time","zhc","yc","xhc"))
            var_total_tau_zu_visc = a.createVariable("total_tau_zu_visc","f8",("time","zhc","yc","xhc"))
            var_res_tau_zu_turb   = a.createVariable("res_tau_zu_turb","f8",("time","zhc","yc","xhc"))
            var_res_tau_zu_visc   = a.createVariable("res_tau_zu_visc","f8",("time","zhc","yc","xhc"))
            var_unres_tau_zu_tot  = a.createVariable("unres_tau_zu_tot","f8",("time","zhc","yc","xhc"))
            var_unres_tau_zu_turb = a.createVariable("unres_tau_zu_turb","f8",("time","zhc","yc","xhc"))
            var_unres_tau_zu_visc = a.createVariable("unres_tau_zu_visc","f8",("time","zhc","yc","xhc"))
 
            var_total_tau_zv_turb = a.createVariable("total_tau_zv_turb","f8",("time","zhc","yhc","xc"))
            var_total_tau_zv_visc = a.createVariable("total_tau_zv_visc","f8",("time","zhc","yhc","xc"))
            var_res_tau_zv_turb   = a.createVariable("res_tau_zv_turb","f8",("time","zhc","yhc","xc"))
            var_res_tau_zv_visc   = a.createVariable("res_tau_zv_visc","f8",("time","zhc","yhc","xc"))
            var_unres_tau_zv_tot  = a.createVariable("unres_tau_zv_tot","f8",("time","zhc","yhc","xc"))
            var_unres_tau_zv_turb = a.createVariable("unres_tau_zv_turb","f8",("time","zhc","yhc","xc"))
            var_unres_tau_zv_visc = a.createVariable("unres_tau_zv_visc","f8",("time","zhc","yhc","xc"))

            var_total_tau_zw_turb = a.createVariable("total_tau_zw_turb","f8",("time","zgcextra","yc","xc"))
            var_total_tau_zw_visc = a.createVariable("total_tau_zw_visc","f8",("time","zgcextra","yc","xc"))
            var_res_tau_zw_turb   = a.createVariable("res_tau_zw_turb","f8",("time","zgcextra","yc","xc"))
            var_res_tau_zw_visc   = a.createVariable("res_tau_zw_visc","f8",("time","zgcextra","yc","xc"))
            var_unres_tau_zw_tot  = a.createVariable("unres_tau_zw_tot","f8",("time","zgcextra","yc","xc"))
            var_unres_tau_zw_turb = a.createVariable("unres_tau_zw_turb","f8",("time","zgcextra","yc","xc"))
            var_unres_tau_zw_visc = a.createVariable("unres_tau_zw_visc","f8",("time","zgcextra","yc","xc"))
 
        create_variables = False #Make sure variables are only created once.
 
        #Store values coarse fields
        var_uc[t,:,:,:] = coarsegrid['output']['u']['variable']
        var_vc[t,:,:,:] = coarsegrid['output']['v']['variable']
        var_wc[t,:,:,:] = coarsegrid['output']['w']['variable']
        var_pc[t,:,:,:] = coarsegrid['output']['p']['variable']

        var_res_u_boxint[t,:,:,:]      = uc_uyzint[:,:,1:]
        var_res_v_boxint[t,:,:,:]      = vc_vxzint[:,1:,:]
        var_res_w_boxint[t,:,:,:]      = wc_wxyint[1:,:,:]
        
        var_unres_u_box[t,:,:,:]       = unres_u_box[:,:,:]
        var_unres_v_box[t,:,:,:]       = unres_v_box[:,:,:]
        var_unres_w_box[t,:,:,:]       = unres_w_box[:,:,:]
        
        var_unres_u_boxint[t,:,:,:]    = unres_u_boxint[:,:,:]
        var_unres_v_boxint[t,:,:,:]    = unres_v_boxint[:,:,:]
        var_unres_w_boxint[t,:,:,:]    = unres_w_boxint[:,:,:]
 
        var_total_tau_xu_turb[t,:,:,:] = total_tau_xu_turb[:,:,:]
        var_total_tau_xu_visc[t,:,:,:] = total_tau_xu_visc[:,:,:]
        var_res_tau_xu_turb[t,:,:,:]   = res_tau_xu_turb[:,:,:]
        var_res_tau_xu_visc[t,:,:,:]   = res_tau_xu_visc[:,:,:]
        var_unres_tau_xu_tot[t,:,:,:]  = unres_tau_xu_tot[:,:,:]
        var_unres_tau_xu_turb[t,:,:,:] = unres_tau_xu_turb[:,:,:]
        var_unres_tau_xu_visc[t,:,:,:] = unres_tau_xu_visc[:,:,:]
 
        var_total_tau_xv_turb[t,:,:,:] = total_tau_xv_turb[:,:,:]
        var_total_tau_xv_visc[t,:,:,:] = total_tau_xv_visc[:,:,:]
        var_res_tau_xv_turb[t,:,:,:]   = res_tau_xv_turb[:,:,:]
        var_res_tau_xv_visc[t,:,:,:]   = res_tau_xv_visc[:,:,:]
        var_unres_tau_xv_tot[t,:,:,:]  = unres_tau_xv_tot[:,:,:]
        var_unres_tau_xv_turb[t,:,:,:] = unres_tau_xv_turb[:,:,:]
        var_unres_tau_xv_visc[t,:,:,:] = unres_tau_xv_visc[:,:,:]
       
        var_total_tau_xw_turb[t,:,:,:] = total_tau_xw_turb[:,:,:]
        var_total_tau_xw_visc[t,:,:,:] = total_tau_xw_visc[:,:,:]
        var_res_tau_xw_turb[t,:,:,:]   = res_tau_xw_turb[:,:,:]
        var_res_tau_xw_visc[t,:,:,:]   = res_tau_xw_visc[:,:,:]
        var_unres_tau_xw_tot[t,:,:,:]  = unres_tau_xw_tot[:,:,:]
        var_unres_tau_xw_turb[t,:,:,:] = unres_tau_xw_turb[:,:,:]
        var_unres_tau_xw_visc[t,:,:,:] = unres_tau_xw_visc[:,:,:]

        var_total_tau_yu_turb[t,:,:,:] = total_tau_yu_turb[:,:,:]
        var_total_tau_yu_visc[t,:,:,:] = total_tau_yu_visc[:,:,:]
        var_res_tau_yu_turb[t,:,:,:]   = res_tau_yu_turb[:,:,:]
        var_res_tau_yu_visc[t,:,:,:]   = res_tau_yu_visc[:,:,:]
        var_unres_tau_yu_tot[t,:,:,:]  = unres_tau_yu_tot[:,:,:]
        var_unres_tau_yu_turb[t,:,:,:] = unres_tau_yu_turb[:,:,:]
        var_unres_tau_yu_visc[t,:,:,:] = unres_tau_yu_visc[:,:,:]

        var_total_tau_yv_turb[t,:,:,:] = total_tau_yv_turb[:,:,:]
        var_total_tau_yv_visc[t,:,:,:] = total_tau_yv_visc[:,:,:]
        var_res_tau_yv_turb[t,:,:,:]   = res_tau_yv_turb[:,:,:]
        var_res_tau_yv_visc[t,:,:,:]   = res_tau_yv_visc[:,:,:]
        var_unres_tau_yv_tot[t,:,:,:]  = unres_tau_yv_tot[:,:,:]
        var_unres_tau_yv_turb[t,:,:,:] = unres_tau_yv_turb[:,:,:]
        var_unres_tau_yv_visc[t,:,:,:] = unres_tau_yv_visc[:,:,:]

        var_total_tau_yw_turb[t,:,:,:] = total_tau_yw_turb[:,:,:]
        var_total_tau_yw_visc[t,:,:,:] = total_tau_yw_visc[:,:,:]
        var_res_tau_yw_turb[t,:,:,:]   = res_tau_yw_turb[:,:,:]
        var_res_tau_yw_visc[t,:,:,:]   = res_tau_yw_visc[:,:,:]
        var_unres_tau_yw_tot[t,:,:,:]  = unres_tau_yw_tot[:,:,:]
        var_unres_tau_yw_turb[t,:,:,:] = unres_tau_yw_turb[:,:,:]
        var_unres_tau_yw_visc[t,:,:,:] = unres_tau_yw_visc[:,:,:]

        var_total_tau_zu_turb[t,:,:,:] = total_tau_zu_turb[:,:,:]
        var_total_tau_zu_visc[t,:,:,:] = total_tau_zu_visc[:,:,:]
        var_res_tau_zu_turb[t,:,:,:]   = res_tau_zu_turb[:,:,:]
        var_res_tau_zu_visc[t,:,:,:]   = res_tau_zu_visc[:,:,:]
        var_unres_tau_zu_tot[t,:,:,:]  = unres_tau_zu_tot[:,:,:]
        var_unres_tau_zu_turb[t,:,:,:] = unres_tau_zu_turb[:,:,:]
        var_unres_tau_zu_visc[t,:,:,:] = unres_tau_zu_visc[:,:,:]

        var_total_tau_zv_turb[t,:,:,:] = total_tau_zv_turb[:,:,:]
        var_total_tau_zv_visc[t,:,:,:] = total_tau_zv_visc[:,:,:]
        var_res_tau_zv_turb[t,:,:,:]   = res_tau_zv_turb[:,:,:]
        var_res_tau_zv_visc[t,:,:,:]   = res_tau_zv_visc[:,:,:]
        var_unres_tau_zv_tot[t,:,:,:]  = unres_tau_zv_tot[:,:,:]
        var_unres_tau_zv_turb[t,:,:,:] = unres_tau_zv_turb[:,:,:]
        var_unres_tau_zv_visc[t,:,:,:] = unres_tau_zv_visc[:,:,:]

        var_total_tau_zw_turb[t,:,:,:] = total_tau_zw_turb[:,:,:]
        var_total_tau_zw_visc[t,:,:,:] = total_tau_zw_visc[:,:,:]
        var_res_tau_zw_turb[t,:,:,:]   = res_tau_zw_turb[:,:,:]
        var_res_tau_zw_visc[t,:,:,:]   = res_tau_zw_visc[:,:,:]
        var_unres_tau_zw_tot[t,:,:,:]  = unres_tau_zw_tot[:,:,:]
        var_unres_tau_zw_turb[t,:,:,:] = unres_tau_zw_turb[:,:,:]
        var_unres_tau_zw_visc[t,:,:,:] = unres_tau_zw_visc[:,:,:]

        #Close file
        a.close()
