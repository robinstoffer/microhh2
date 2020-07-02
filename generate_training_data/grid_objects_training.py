#Script containing Finegrid and Coarsegrid classes for generation training data NN
#Author: Robin Stoffer (robin.stoffer@wur.nl)

#Developed for Python 3!
import numpy as np
import downsampling_training
import boxfilter
import microhh_tools_robinst as tools
import os

class Finegrid:
    """ Returns a single object that can read from MicroHH output binary files the grid and output variables. For testing purposes, the grid and output variables can also be explicitly defined. Furthermore, the object contains methods to implement ghost cells and calculate the volume integral. The inputs for this script are:
        
        -read_grid flag: whether the grid is read from an existing simulation (True) or explicitly defined for testing purposes (False).
         When read_grid flag is True, specify explicitly (using keywords) grid_filepath and settings_filepath.
            -If no grid filepath is provided, grid.0000000 from the current directory is read.
            -If no settings filepath is provided, a .ini file should be present in the current directory to read.
         When read_grid flag is False, specifiy explicitly (using keywords) each spatial coordinate (coordz,coordy,coordx) as a 1d numpy array, e.g. Finegrid(read_grid_flag = False, coordx = np.array([1,2,3]),xsize = 4, coordy = np.array([0.5,1.5,3.5,6]), ysize = 7, coordz = np.array([0.1,0.3,1]), zsize = 1.2).
         Furthermore, for each spatial dimension the length should be indicated (zsize,ysize,xsize), which has to be larger than the last spatial coordinate in each dimension. 
         Each coordinate should 1) refer to the center of the grid cell, 2) be larger than 0, 3) be larger than the previous coordinate in the same direction. Furthermore, in the x- and y-direction the specified should be uniform (i.e. constant grid distance).

        -precision: The precision indicates the floating point precision of the fine resolution data, which is either 'single' or 'double'.
       
        -fourth_order flag: indicates the order of the needed spatial interpolation in later processing steps, making sure sufficient ghost cells are added.
         If true, ghost cells are implemented for fourth_order interpolation. If false (default), ghost cells are added for second order spatial interpolation.

        -periodic_bc: should be a tuple indicating for each spatial direction (z, y, x) whether periodic boundary conditions are assumed (True when present, False when not present).
        
        -zero_w_topbottom: boolean specifying whether the vertical wind velocity is 0 at the bottom and top levels of the domain or not. This is used to define ghost cells in the vertical direction.
        
        -normalisation_grid: boolean indicating if the specified height of the simulation should be used to normalize the coordinate axes (True), which is for the a turbulent channel flow equal to the half-channel width."""
        

    def __init__(self, read_grid_flag = True , precision = 'double', fourth_order = False, periodic_bc = (False, True, True), zero_w_topbottom = True, normalisation_grid = False, **kwargs):
        
        # Test whether types of input variables are correct.
        if not isinstance(read_grid_flag, bool):
            raise TypeError('The fourth_order flag should be a boolean (True/False).')
        if not isinstance(fourth_order, bool):
            raise TypeError('The fourth_order flag should be a boolean (True/False).')
        if not isinstance(zero_w_topbottom, bool):
            raise TypeError('The zero_w_topbottom flag should be a boolean (True/False).')
        if not isinstance(normalisation_grid, bool):
            raise TypeError('The normalisation_grid flag should be a boolean (True/False).')

            
        #Set correct precision and order of spatial interpolation with corresponding ghost cells
        self.prec  = tools._process_precision(precision)
        self.__determine_sgn_digits() #Determine significant decimals digits corresponding to self.prec
        self.fourth_order = fourth_order
        self.zero_w_topbottom = zero_w_topbottom
        
        if self.fourth_order:
            #self.igc = 3
            #self.jgc = 3
            #self.kgc_center = 3
            #self.kgc_edge = 0
            raise RuntimeError('Fourth order interpolation is currently not yet implemented in this script, only second order interpolation. If the user wants to implement fourth order interpolation, the code that adds the ghost cells in the vertical direction needs to be changed (where no periodic boundary conditions can be assumed).')
        else:
            self.igc = 1
            self.jgc = 1
            self.kgc_center = 1 #Although in the vertical direction no periodic_bc is usually assumed, you can make use of the boundary bc that w = 0 (no-slip BC).
            self.kgc_edge = 0

        #Define empty dictionary to store grid and variables
        self.var = {}
        self.var['output'] = {}
        self.var['boxfilter'] = {}
        self.var['grid'] = {}
        self.var['time'] = {}
        self.var['fields'] = {}
        
        #Check whether specified period_bc satisfies needed format and subsequently store it in object
        if not isinstance(periodic_bc, tuple):
            raise TypeError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
        if not len(periodic_bc) == 3:
            raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not any(isinstance(flag, bool) for flag in periodic_bc):
            raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
        if periodic_bc[0]:
            raise RuntimeError("In the vertical direction the periodic BC's cannot be imposed in this script.  Add additional functionality to this script if needed.")
        
        self.periodic_bc = periodic_bc

        #Read or define the grid depending on read_grid_flag
        if read_grid_flag:
            self.read_grid_flag = True
            self.settings_filepath = kwargs.get('settings_filepath', None)
            self.grid_filepath = kwargs.get('grid_filepath', 'grid.0000000')
            self.__read_settings(settings_filepath = self.settings_filepath, normalisation_grid = normalisation_grid)
            self.__read_binary_grid(grid_filepath = self.grid_filepath, normalisation_grid = normalisation_grid)
            self.define_grid_flag = False

        else:
            self.read_grid_flag = False
            self.define_grid_flag = False
            try:
                def __checks_coord(coord_name, dim, uniform):
                    diff_coord = np.round(kwargs[coord_name][1:] - kwargs[coord_name][:-1], self.sgn_digits)
                    if not(np.all(diff_coord > 0) and np.all(kwargs[coord_name][:] > 0) and \
                    (len(kwargs[coord_name][:].shape) == 1)):
                        raise ValueError("The coordinates should be 1-dimensional, strictly increasing, and consist of positive values only.")
                    
                    if not (len(kwargs[coord_name]) == 1): #Test whether the grid is uniform only when the coordinate exists of more than one grid cell.
                        diff_bottom = np.round(kwargs[coord_name][0] - 0, self.sgn_digits)
                        half_distance = np.round(0.5*diff_coord[0], self.sgn_digits)
                        if uniform and ((np.any(diff_coord != diff_coord[0])) or (diff_bottom != half_distance)):
                            raise ValueError("The coordinates in the x- and y-direction should be uniform.")
                            
                    self.var['grid'][dim] = kwargs[coord_name]

                __checks_coord('coordx', 'x', True)
                __checks_coord('coordy', 'y', True)
                __checks_coord('coordz', 'z', False)

                def __checks_size(size_name, coord_name, uniform):
                    diff_coord = np.round(kwargs[coord_name][1:] - kwargs[coord_name][:-1], self.sgn_digits)
                    if not kwargs[size_name] > kwargs[coord_name][-1]:
                        raise ValueError("The length of the coordinate should be larger than the last coordinate.")
                    
                    if not (len(kwargs[coord_name]) == 1): #Test whether the grid is uniform only when the coordinate exists of more than one grid cell.
                        diff_top = np.round(kwargs[size_name] - kwargs[coord_name][-1], self.sgn_digits)
                        half_distance = np.round(0.5*diff_coord[0], self.sgn_digits)
                        if uniform and (diff_top != half_distance):
                            raise ValueError("The coordinates in the x- and y-direction should be uniform, which should also be taken into account when defining the sizes of the grid.")
 
                    self.var['grid'][size_name] = kwargs[size_name]

                __checks_size('xsize', 'coordx', True)
                __checks_size('ysize', 'coordy', True)
                __checks_size('zsize', 'coordz', False)

            except KeyError:
                print("The needed arguments were not correctly specified. Make sure that coordx, coordy and coordz are assinged to a 1d numpy array.")
                raise
                 
            #For convenience with the script func_generate_training.py, timesteps is set equal to 1.
            self.var['time']['timesteps'] = 1
            
            #Store manually defined grid
            self.__define_grid(self.var['grid']['z'],self.var['grid']['y'],self.var['grid']['x'], self.var['grid']['zsize'], self.var['grid']['ysize'], self.var['grid']['xsize'])

    def __read_settings(self, normalisation_grid, settings_filepath = None):
        """ Read settings from specified file (default: .ini file in current directory). """

        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing settings file.")

        #Read settings of MicroHH simulation from specified settings_filename. 
        #If None, a .ini file should be present in the current directory to read.
        settings = tools.Read_namelist(settings_filepath)

        self.var['grid']['itot'] = int(settings['grid']['itot'])
        self.var['grid']['jtot'] = int(settings['grid']['jtot'])
        self.var['grid']['ktot'] = int(settings['grid']['ktot'])
        self.var['grid']['zsize'] = settings['grid']['zsize']
        self.var['grid']['channel_half_width'] = self.var['grid']['zsize'] * 0.5 #Channel half-width is equal to 0.5 * channel height.

        if normalisation_grid:
            normalisation_factor_grid = self.var['grid']['channel_half_width'] #Use channel_half_width to normalize grid.
            self.var['grid']['xsize'] = settings['grid']['xsize'] / normalisation_factor_grid
            self.var['grid']['ysize'] = settings['grid']['ysize'] / normalisation_factor_grid
            self.var['grid']['zsize'] = settings['grid']['zsize'] / normalisation_factor_grid
        else:
            self.var['grid']['xsize'] = settings['grid']['xsize']
            self.var['grid']['ysize'] = settings['grid']['ysize']
            self.var['grid']['zsize'] = settings['grid']['zsize']


        self.var['fields']['visc'] = settings['fields']['visc']

        self.var['time']['starttime'] = int(settings['time']['starttime'])
        #self.var['time']['starttime'] = 0
        self.var['time']['endtime'] = int(settings['time']['endtime'])
        #self.var['time']['endtime'] = 7200
        self.var['time']['savetime'] = int(settings['time']['savetime'])
        #self.var['time']['savetime'] = 600
        self.var['time']['timesteps'] = int((self.var['time']['endtime'] - self.var['time']['starttime']) // self.var['time']['savetime'])
        #self.var['time']['timesteps'] = 13

    def __read_binary_grid(self, grid_filepath, normalisation_grid):
        """ Read binary grid from specified file (default: grid.0000000). """

        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read the grid from an already existing binary grid file.")

        #Open binary file containing the grid information
        f = open(grid_filepath,'rb')

        #Infer number of bytes from the used precision
        nbytes = 8 if self.prec == 'float64' else 4

        #Define function to read each coordinate
        def __read_coordinate(file_opened, nbytes, number_gridcells, dim):
            self.var['grid'][dim] = np.fromfile(f, dtype = self.prec, count = number_gridcells)

        #Call above defined function for all coordinates in the right order (the order in which they are stored in the file)
        __read_coordinate(f, nbytes, self.var['grid']['itot'], 'x') 
        __read_coordinate(f, nbytes, self.var['grid']['itot'], 'xh')
        __read_coordinate(f, nbytes, self.var['grid']['jtot'], 'y')
        __read_coordinate(f, nbytes, self.var['grid']['jtot'], 'yh')
        __read_coordinate(f, nbytes, self.var['grid']['ktot'], 'z')
        __read_coordinate(f, nbytes, self.var['grid']['ktot'], 'zh')
        f.close()

        #Normalise coordinates with specified normalisation_factor (except when specified as None)
        if normalisation_grid:

            normalisation_factor_grid = self.var['grid']['channel_half_width'] #Use channel_half_width to normalize grid.
            self.var['grid']['x']  = self.var['grid']['x']  / normalisation_factor_grid
            self.var['grid']['xh'] = self.var['grid']['xh'] / normalisation_factor_grid
            self.var['grid']['y']  = self.var['grid']['y']  / normalisation_factor_grid
            self.var['grid']['yh'] = self.var['grid']['yh'] / normalisation_factor_grid
            self.var['grid']['z']  = self.var['grid']['z']  / normalisation_factor_grid
            self.var['grid']['zh'] = self.var['grid']['zh'] / normalisation_factor_grid


        #Add ghostcells to grid
        self.__add_ghostcells_grid()

    def read_binary_variables(self, filepath, variable_name, timestep, bool_edge_gridcell = (False, False, False), normalisation_factor = None):
        """ Read specified variable at a certain time step from binary files of a MicroHH simulation stored in filepath. Furthermore, ghost cells are added consistent with the specified order of spatial interpolation, and the variable is normalized when normalisation_factor is not None."""
        if not self.read_grid_flag:
            raise RuntimeError("Object defined to use grid explicitly set by user (via the define_grid method). Create a new Finegrid object with read_grid_flag = True to read settings from an already existing binary file.")

        #Check that variable_name is a string
        if not isinstance(filepath, str):
            raise TypeError("Specified filepath should be a string.")

        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
            
        #Check that timestep is an integer
        if not isinstance(timestep, int):
            raise TypeError("Specified timestep should be an integer.")
            
        #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object
        if not isinstance(bool_edge_gridcell, tuple):
            raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not len(bool_edge_gridcell) == 3:
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
                
        if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

        #Check wheter timestep is present in data
        if timestep > (self.var['time']['timesteps'] - 1):
            raise ValueError("Specified timestep not contained in data. Note that the timestep starts counting from 0.")

        #Open binary file to read variable at timestep
        time = self.var['time']['starttime'] + timestep*self.var['time']['savetime']
        var_filename = '{0}.{1:07d}'.format(variable_name,time)
        var_filepath = os.path.join(filepath, var_filename)        

        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['variable'] = np.fromfile(var_filepath, dtype = self.prec).reshape((self.var['grid']['ktot'], self.var['grid']['jtot'], self.var['grid']['itot']))
        self.var['output'][variable_name]['orientation'] = bool_edge_gridcell
        
        #Normalise variable with specified normalisation_factor (except when specified as None)
        if normalisation_factor is not None:
            #Test whether normalisation_factor is an integer or float
            if not (isinstance(normalisation_factor, int) or isinstance(normalisation_factor, float)):
                raise TypeError("Specified normalisation_factor should be an integer or float.")
            self.var['output'][variable_name]['variable'] = self.var['output'][variable_name]['variable'] / normalisation_factor


        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name)

    def __define_grid(self, coordz, coordy, coordx, sizez, sizey, sizex):
        """ Manually define grid stored in object. """

        if self.read_grid_flag:
            raise RuntimeError("Object defined to read the grid from binary files (via the read_binary_grid method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")

        #define lengths of specified grid
        self.var['grid']['itot'] = len(coordx)
        self.var['grid']['jtot'] = len(coordy)
        self.var['grid']['ktot'] = len(coordz)

        #create coordinates corresponding to edges grid cell
        self.var['grid']['xh'] = self.__edgegrid_from_centergrid(self.var['grid']['x'], self.var['grid']['itot'], sizex)
        self.var['grid']['yh'] = self.__edgegrid_from_centergrid(self.var['grid']['y'], self.var['grid']['jtot'], sizey)
        self.var['grid']['zh'] = self.__edgegrid_from_centergrid(self.var['grid']['z'], self.var['grid']['ktot'], sizez)
        
        #Add ghostcells to grid
        self.__add_ghostcells_grid()
        
        #Set define_grid_flag to True, such that the create_variables method can be executed
        self.define_grid_flag = True

    def create_variables(self, variable_name, output_field, bool_edge_gridcell = (False, False, False)):
        """ Manually define output variables for a manually defined grid. Furthermore, ghost cells are added consistent with the specified order of spatial interpolation. """

        if self.read_grid_flag:
            raise RuntimeError("Object defined to read variables from binary files (via the read_binary_variables method). Create a new Finegrid object with read_grid_flag = False to manually define the settings of the grid.")
            
        #Check that grid is already defined.
        if not self.define_grid_flag:
            raise RuntimeError("Object does not yet contain the coordinates for the output variable. Define them explicitly first via the __define_grid method")

        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
            
        #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object
        if not isinstance(bool_edge_gridcell, tuple):
            raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
        if not len(bool_edge_gridcell) == 3:
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
                
        if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
            raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

        #Check that shape output field is correct according to the coordinates specified via the define_grid method.
        if not (output_field.shape == (self.var['grid']['ktot'],self.var['grid']['jtot'],self.var['grid']['itot'])):
            raise RuntimeError("The shape corresponding to the specified coordinates (z,y,x) is not the same as the shape of the specified output field. Note that for the staggered grid the top/downstream boundaries do not have to be included, these are automatically added.")

        #Store output_field in object
        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['variable'] = output_field
        
        #Store orientation of output_field in object
        self.var['output'][variable_name]['orientation'] = bool_edge_gridcell

        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name, is_boxfiltered = False)
        
    def __add_ghostcells(self, variable_name, is_boxfiltered = False):
        """ Add ghostcells (defined as grid cells located at the downstream/top boundary and outside the domain) to fine grid for variable_name. 
            For periodic_bc, the samples at the end of the array are repeated at the beginning and vice versa.
            For zero_w_topbottom bc, the samples are extrapolated such that w=0 at the edges of the domain."""
        
        #If boxfiltered variable is considered (is_boxfiltered = True), store in appropriate dictionary
        if is_boxfiltered:
            dict_name = 'boxfilter'
        else:
            dict_name = 'output'

        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var[dict_name].keys():
            raise KeyError("Specified variable_name not defined in object.")

        #Check that ghost cells are not negative
        if self.jgc < 0 or self.igc < 0 or self.kgc_center < 0 or self.kgc_edge < 0:
            raise ValueError("The specified number of ghostcells cannot be negative.")
            
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's, except one ghost cell in the vertical direction with a no-slip BC.
        if self.kgc_edge > 0 and (not self.periodic_bc[0]):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")
        
        if self.kgc_center > 0 and (not self.periodic_bc[0]) and not (self.kgc_center == 1 and self.zero_w_topbottom):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Read variable from object
        s = self.var[dict_name][variable_name]['variable']
        
        #Depending on bool_edge_gridcell, add one additional ghost cell at top/downstream boundaries independent of self.igc/jgc/kgc
        self.kgc = self.kgc_center
        bkgc = 0
        bjgc = 0
        bigc = 0
        if self.var[dict_name][variable_name]['orientation'][0]:
            bkgc = 1
            self.kgc = self.kgc_edge
        if self.var[dict_name][variable_name]['orientation'][1]:
            bjgc = 1
        if self.var[dict_name][variable_name]['orientation'][2]:
            bigc = 1

        self.siend = self.igc + self.var['grid']['itot'] + bigc
        self.sjend = self.jgc + self.var['grid']['jtot'] + bjgc
        self.skend = self.kgc + self.var['grid']['ktot'] + bkgc
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + bigc) > (s.shape[2]) or (self.jgc + bjgc) > (s.shape[1]) or (self.kgc + bkgc) > (s.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Initialize new output array including ghost cells
        sgc = np.zeros((kcells+bkgc, jcells+bjgc, icells+bigc))
        
        #Fill new initialzid array including ghost cells for horizontal directions
        sgc[self.kgc:self.skend-bkgc, self.jgc:self.sjend-bjgc, self.igc:self.siend-bigc] = s[:,:,:].copy()
        sgc[:,:,0:self.igc] = sgc[:,:,self.siend-self.igc-bigc:self.siend-bigc] #Add ghostcell upstream x-direction
        sgc[:,:,self.siend-bigc:self.siend+self.igc] = sgc[:,:,self.igc:self.igc+self.igc+bigc] #Add ghostcell downstream x-direction
        sgc[:,0:self.jgc,:] = sgc[:,self.sjend-self.jgc-bjgc:self.sjend-bjgc,:] #Add ghostcell upstream y-direction
        sgc[:,self.sjend-bjgc:self.sjend+self.jgc,:] = sgc[:,self.jgc:self.jgc+self.jgc+bjgc,:] #Add ghostcell downstream y-direction
        if self.kgc == 1: #Mirror profile over BC
            sgc[0:self.kgc,:,:] = 0 - sgc[self.kgc,:,:] #Add ghostcell bottom z-direction
            sgc[self.skend,:,:] = 0 - sgc[self.skend-1,:,:] #Add ghostcell top z-direction
        if bkgc == 1:
            sgc[self.skend-bkgc] = sgc[self.kgc,:,:] #Add top boundary when variable s is located on the grid_edges in the vertical direction, making sure that is satisfies the same BC.
        
        #Store new fields in object
        self.var[dict_name][variable_name]['variable'] = sgc
        
    def __add_ghostcells_grid(self):
        '''Store new coordinates with ghost cells in object.'''

        #Get original coordinates
        z = self.var['grid']['z']
        zh = self.var['grid']['zh']
        y = self.var['grid']['y']
        yh = self.var['grid']['yh']
        x = self.var['grid']['x']
        xh = self.var['grid']['xh']

        #Check that the grid center and grid edges arrays defined above have the same shape (which is an implicit assumption in the code below).
        if (z.shape != zh.shape) or (y.shape != yh.shape) or (x.shape != xh.shape):
            raise RuntimeError("The shape of the arrays representing the grid centers and edges should be the same, but they are not while this implicitly assumed in the script." )
        
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's, except one ghost cell in the vertical direction with a no-slip BC.
        if self.kgc_edge > 0 and (not self.periodic_bc[0]):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells within this script. Add additional functionality to this script if needed.")
        
        if self.kgc_center > 0 and (not self.periodic_bc[0]) and not (self.kgc_center == 1 and self.zero_w_topbottom):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells within this script. Add additional functionality to this script if needed.")
                
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + 1) > (x.shape[0]) or (self.jgc + 1) > (y.shape[0]) or (max(self.kgc_center, self.kgc_edge) + 1) > (z.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc_center
        khcells = self.var['grid']['ktot'] + 2*self.kgc_edge 
        
        #Initialize new coordinates and store ghost cells in them
        xgc = np.zeros(icells)
        xhgc = np.zeros(icells+1)
        ygc = np.zeros(jcells)
        yhgc = np.zeros(jcells+1)
        zgc = np.zeros(kcells)
        zhgc = np.zeros(khcells+1)
        
        self.iend = self.igc + self.var['grid']['itot']
        self.ihend = self.igc + self.var['grid']['itot'] + 1
        self.jend = self.jgc + self.var['grid']['jtot']
        self.jhend = self.jgc + self.var['grid']['jtot'] + 1
        self.kend = self.kgc_center + self.var['grid']['ktot']
        self.khend = self.kgc_edge + self.var['grid']['ktot'] + 1
        
        xgc  = self.__add_ghostcells_cor(xgc , 0,  self.igc, self.iend , x,  self.var['grid']['xsize'])
        xhgc = self.__add_ghostcells_cor(xhgc, 1,  self.igc, self.ihend, xh, self.var['grid']['xsize'])
        ygc  = self.__add_ghostcells_cor(ygc , 0,  self.jgc, self.jend , y,  self.var['grid']['ysize'])
        yhgc = self.__add_ghostcells_cor(yhgc, 1,  self.jgc, self.jhend, yh, self.var['grid']['ysize'])
        zgc  = self.__add_ghostcells_cor_ver(zgc , 0,  self.kgc_center, self.kend , z,  self.var['grid']['zsize'])
        zhgc = self.__add_ghostcells_cor_ver(zhgc, 1,  self.kgc_edge, self.khend, zh, self.var['grid']['zsize'])

        self.var['grid']['z'] = zgc
        self.var['grid']['zh'] = zhgc
        self.var['grid']['y'] = ygc
        self.var['grid']['yh'] = yhgc
        self.var['grid']['x'] = xgc
        self.var['grid']['xh'] = xhgc
        
    def __add_ghostcells_cor(self, corgc, bgc, gc, endindex, cor, size):
        """ Add ghostcells to coordinates in horizontal directions, making use of the periodic BC's. """
        corgc[gc:endindex-bgc] = cor[:]
        if gc != 0:
            corgc[0:gc] = 0 - (size - corgc[endindex-bgc-gc:endindex-bgc])
        if (gc != 0) or (bgc != 0): 
            corgc[endindex-bgc:endindex+gc] = size + corgc[gc:gc+gc+bgc]
        return corgc
    
    def __add_ghostcells_cor_ver(self, corgc, bgc, gc, endindex, cor, size):
        """ Add 0 or 1 ghostcell to coordinates in vertical direction, making use of the zero_w_topbottom BC. """
        corgc[gc:endindex-bgc] = cor[:]
        if not (gc == 0 or gc == 1):
            raise RuntimeError("Number of ghost cells to be added in vertical direction not consistent with specified BC's.")
        if gc == 1:
            corgc[0] = 0 - cor[0]
            corgc[endindex] = size + (size - corgc[endindex-1])
        if bgc == 1: 
            corgc[endindex-bgc] = size
        return corgc
    
    def volume_integral(self, variable_name):
        """ Return volume integral for variable specified by variable name. (Defined for testing purposes.)"""
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")
            
        #Read in variable values and axes, remove ghostcells, calculate grid distances for integration
        s = self.var['output'][variable_name]['variable']
        
        if self.var['output'][variable_name]['orientation'][0]:
            s = s[self.kgc_edge:self.khend,:,:].copy()
            z_alt = self.var['grid']['z'][self.kgc_center:self.kend]
            z_diff = z_alt[1:] - z_alt[:-1]
            z_diff = np.insert(z_diff, 0, z_alt[0])
            z_diff = np.append(z_diff, self.var['grid']['zsize'] - z_alt[-1])
            
        else:
            s = s[self.kgc_center:self.kend,:,:].copy()
            z_alt = self.var['grid']['zh'][self.kgc_edge:self.khend]
            z_diff = z_alt[1:] - z_alt[:-1]
            
        if self.var['output'][variable_name]['orientation'][1]:
            s = s[:,self.jgc:self.jhend,:].copy()
            y_alt = self.var['grid']['y'][self.jgc:self.jend]
            y_diff = y_alt[1:] - y_alt[:-1]
            y_diff = np.insert(y_diff, 0, y_alt[0])
            y_diff = np.append(y_diff, self.var['grid']['ysize'] - y_alt[-1])
            
        else:
            s = s[:,self.jgc:self.jend,:].copy()
            y_alt = self.var['grid']['yh'][self.jgc:self.jhend]
            y_diff = y_alt[1:] - y_alt[:-1]
        
        if self.var['output'][variable_name]['orientation'][2]:
            s = s[:,:,self.igc:self.ihend].copy()
            x_alt = self.var['grid']['x'][self.igc:self.iend]
            x_diff = x_alt[1:] - x_alt[:-1]
            x_diff = np.insert(x_diff, 0, x_alt[0])
            x_diff = np.append(x_diff, self.var['grid']['xsize'] - x_alt[-1])
            
        else:
            s = s[:,:,self.igc:self.iend].copy()
            x_alt = self.var['grid']['xh'][self.igc:self.ihend]
            x_diff = x_alt[1:] - x_alt[:-1]
            
        #Integrate over variable by multiplying with the corresponding grid distances and subsequent summing.
        s_intx = np.tensordot(s, z_diff, axes = (0,0))
        s_intyx = np.tensordot(s_intx, y_diff, axes = (0,0))
        s_intzyx = np.tensordot(s_intyx, x_diff, axes = (0,0))
        
        return s_intzyx
    
    def __edgegrid_from_centergrid(self, coord_center, len_coord, size_coord):
        """ Define coordinates corresponding to the grid walls from coordinates corresponding to the centers of the grid. """

        dcoord = coord_center[1:] - coord_center[:-1]
        coord_edge = np.empty(len_coord)
        coord_edge[0] = 0
        for i in range(1,len_coord):
            coord_edge[i] = coord_center[i-1] + 0.5 * dcoord[i-1]
        return coord_edge
    
    def __determine_sgn_digits(self):
        '''Determine significant decimal digits corresponding to specified machine precision (double or single).'''
        if self.prec == 'float64':
            self.sgn_digits = 15
        elif self.prec == 'float32':
            self.sgn_digits = 6
        else:
            raise RuntimeError('Script not configred for precision specified in finegrid object.')

    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')
    
    def boxfilter(self, filter_widths):
        
        #Check that filter_widths is a tuple of length 3 (z,y,x) with positive floats only.
        if not isinstance(filter_widths, tuple):
            raise TypeError("Filter_widths should be a tuple with length 3 (z, y, x), and consist only of floats.")
        
        if not (len(filter_widths) == 3):
            raise TypeError("Filter_widths should be a tuple with length 3 (z, y, x), and consist only of floats.")

        if not any(isinstance(filter_width, float) for filter_width in filter_widths):
            raise TypeError("Filter_widths should be a tuple with length 3 (z, y, x), and consist only of floats.")
        
        if any((filter_width <= 0.) for filter_width in filter_widths):
            raise ValueError("Filter width must be larger than 0.")

        self.var['boxfilter']['u'] = {}
        self.var['boxfilter']['v'] = {}
        self.var['boxfilter']['w'] = {}
        self.var['boxfilter']['uu'] = {}
        self.var['boxfilter']['vu'] = {}
        self.var['boxfilter']['wu'] = {}
        self.var['boxfilter']['uv'] = {}
        self.var['boxfilter']['vv'] = {}
        self.var['boxfilter']['wv'] = {}
        self.var['boxfilter']['uw'] = {}
        self.var['boxfilter']['vw'] = {}
        self.var['boxfilter']['ww'] = {}
        #
        self.var['boxfilter']['u']['orientation'] = (False, False, False)
        self.var['boxfilter']['v']['orientation'] = (False, False, False)
        self.var['boxfilter']['w']['orientation'] = (False, False, False)
        self.var['boxfilter']['uu']['orientation'] = (False, False, False)
        self.var['boxfilter']['vu']['orientation'] = (False, False, False)
        self.var['boxfilter']['wu']['orientation'] = (False, False, False)
        self.var['boxfilter']['uv']['orientation'] = (False, False, False)
        self.var['boxfilter']['vv']['orientation'] = (False, False, False)
        self.var['boxfilter']['wv']['orientation'] = (False, False, False)
        self.var['boxfilter']['uw']['orientation'] = (False, False, False)
        self.var['boxfilter']['vw']['orientation'] = (False, False, False)
        self.var['boxfilter']['ww']['orientation'] = (False, False, False)
        #
        self = boxfilter.boxfilter(filter_widths, finegrid = self, periodic_bc = self.periodic_bc, zero_w_topbottom = self.zero_w_topbottom)

        ##Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        #self.__add_ghostcells('u', is_boxfiltered = True)
        #self.__add_ghostcells('v', is_boxfiltered = True)
        #self.__add_ghostcells('w', is_boxfiltered = True)

class Coarsegrid:
    """ Returns a single object that defines a coarse grid based on a corresponding finegrid_object (instance of Finegrid class) and the dimensions of the new grid. \\
        These dimensions of the coarse grid should be specified as a tuple (z,y,x) (dim_new_grid). Optionally, the number of ghost cells in all directions (igc,jgc, kgc) can be specified as integers. \\ 
        By default, the corresponding amount of ghostcells in the finegrid_object are used. Besides that, a method is provided to downsample variables already present in the finegrid_ojbect. \\ 
        NOTE1: if the finegrid does not contain the variable to be downsampled at the time when the coarsegrid object is initialized, no downsampling is possible. \\
        NOTE2: in the vertical staggered dimensions (zh, yh, xh) of the grid, similar to the finegrid_object one more ghost cell is added than specified by (igc,jgc, kgc). This is always a copy of the first value, which makes sure that the top value satisfies the BCs (i.e. U = 0 or periodic BC)."""

    def __init__(self, dim_new_grid, finegrid_object, **kwargs):
        
        #Check whether dim_new_grid is a tuple of length 3 with only positive integers
        if not isinstance(dim_new_grid, tuple):
            raise TypeError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
        
        if not len(dim_new_grid) == 3:
            raise ValueError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
            
        if not any( (dim > 0 and isinstance(dim, int)) for dim in dim_new_grid):
            raise ValueError("Dim_new_grid should be a tuple with length 3 (z, y, x), and consist only of positive integers.")
        
        nz,ny,nx = dim_new_grid

        #Store finegrid_object in Coarsegrid object
        self.finegrid = finegrid_object

        #Store relevant settings from finegrid_object and dim_new_grid into coarsegrid_object.
        #NOTE: for the ghostcells in the horizontal direction by default the corresponding amount of ghostcells in the finegrid is used.
        self.var = {}
        self.var['grid'] = {}
        self.var['output'] = {} #After the coarsegrid object is created, output can be stored here. This is done in the script func_generate_training.py.
        try:
            self.igc = kwargs['igc']
            if not isinstance(self.igc, int):
                raise TypeError('Specified ghost cells should be an integer.')
        except KeyError:
            self.igc = self.finegrid.igc
        try:
            self.jgc = kwargs['jgc']
            if not isinstance(self.jgc, int):
                raise TypeError('Specified ghost cells should be an integer.')
        except KeyError:
           self.jgc = self.finegrid.jgc  
        try:
            self.kgc = kwargs['kgc']
            if not isinstance(self.kgc, int):
                raise TypeError('Specified ghost cells should be an integer.')
        except KeyError:
            self.kgc = self.finegrid.kgc_center  

        self.periodic_bc = self.finegrid.periodic_bc
        self.zero_w_topbottom = self.finegrid.zero_w_topbottom
        self.sgn_digits = self.finegrid.sgn_digits

        #Boundary condition: the resolution of the coarse grid should be coarser than the fine resolution
        if not (nz <= finegrid_object['grid']['ktot'] and ny <= finegrid_object['grid']['jtot'] and nx <= finegrid_object['grid']['itot']):
            raise ValueError("The resolution of the coarse grid should not be finer than the fine resolution contained in the Finegrid object.")        

        self.var['grid']['ktot'] = nz
        self.var['grid']['jtot'] = ny
        self.var['grid']['itot'] = nx

        try:
            self.var['grid']['xsize'] = finegrid_object['grid']['xsize']
            self.var['grid']['ysize'] = finegrid_object['grid']['ysize']
            self.var['grid']['zsize'] = finegrid_object['grid']['zsize']

        except KeyError:
            raise KeyError("At least one of the needed settings was not contained in the finegrid_object. Check that the relevant methods of the finegrid object have been called to initialize the grid.")
    
        self.var['grid']['x'] , self.var['grid']['xdist']  = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = True) 
        self.var['grid']['xh'], self.var['grid']['xhdist'] = self.__define_coarsegrid(self.var['grid']['itot'], self.var['grid']['xsize'], center_cell = False)
        self.var['grid']['y'] , self.var['grid']['ydist']  = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = True)
        self.var['grid']['yh'], self.var['grid']['yhdist'] = self.__define_coarsegrid(self.var['grid']['jtot'], self.var['grid']['ysize'], center_cell = False)
        self.var['grid']['z'] , self.var['grid']['zdist']  = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = True)
        self.var['grid']['zh'], self.var['grid']['zhdist'] = self.__define_coarsegrid(self.var['grid']['ktot'], self.var['grid']['zsize'], center_cell = False)
        
        #Add ghostcells to grid
        self.__add_ghostcells_grid()
        
    #Calculate new grid using the information stored above, assuming the grid is uniform
    def __define_coarsegrid(self, number_gridcells, gridsize, center_cell):
        """ Manually define the coarse grid. Note that the size of the domain is determined by the finegrid object used to initialize the coarsegrid object. """        

        dist_coarsecoord = gridsize / number_gridcells
        if center_cell:
            coarsecoord, step = np.linspace(0.5*dist_coarsecoord, gridsize-0.5*dist_coarsecoord, number_gridcells, endpoint=True, retstep=True)
        else:
            coarsecoord, step = np.linspace(0, gridsize, number_gridcells, endpoint=False, retstep=True)

        return coarsecoord, step
    
    def downsample(self, variable_name):
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.finegrid['output'].keys():
            raise KeyError("Specified variable_name not defined in finegrid object on which this coarsegrid object is based. Not possible to do downsampling.")

        self.var['output'][variable_name] = {}
        self.var['output'][variable_name]['orientation'] = self.finegrid['output'][variable_name]['orientation']
        variable_downsampled = downsampling_training.downsample(finegrid = self.finegrid, coarsegrid = self, variable_name = variable_name, bool_edge_gridcell = self.var['output'][variable_name]['orientation'], periodic_bc = self.periodic_bc, zero_w_topbottom = self.zero_w_topbottom)
        self.var['output'][variable_name]['variable'] = variable_downsampled.copy()
        #Add ghostcells depending on the order used for the spatial interpolation, add 1 additonal cell on downstream/top boundaries
        self.__add_ghostcells(variable_name)
            
    def __add_ghostcells(self, variable_name):
        """ Add ghostcells (defined as grid cells located at the downstream/top boundary and outside the domain) to coarse grid for variable_name. 
            The samples at the end of the array are repeated at the beginning and vice versa. Furthermore, the ghostcells are added to the coordinates as well."""
            
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")

        #Check that ghost cells are not negative
        #if self.jgc < 0 or self.igc < 0 or self.kgc_center < 0 or self.kgc_edge < 0:
        if self.jgc < 0 or self.igc < 0 or self.kgc < 0 :
            raise ValueError("The specified number of ghostcells cannot be negative.")
            
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's, except one ghost cell in the vertical direction with a no-slip BC.
        #if self.kgc_edge > 0 and (not self.periodic_bc[0]):
        #    raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")
        
        #if self.kgc_center > 0 and (not self.periodic_bc[0]) and not (self.kgc_center == 1 and self.zero_w_topbottom):
        if self.kgc > 0 and not (self.periodic_bc[0] or self.zero_w_topbottom):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")
            
        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Read variable from object
        s = self.var['output'][variable_name]['variable']
        
        #Depending on bool_edge_gridcell, add one additional ghost cell at top/downstream boundaries independent of self.igc/jgc/kgc
        #self.kgc = self.kgc_center
        bkgc = 0
        bjgc = 0
        bigc = 0
        if self.var['output'][variable_name]['orientation'][0]:
            bkgc = 1
            #self.kgc = self.kgc_edge
            #s = s[:-1,:,:].copy() NOTE: This line should NOT be commented out in case of periodic BC
        if self.var['output'][variable_name]['orientation'][1]:
            bjgc = 1
            s = s[:,:-1,:].copy()
        if self.var['output'][variable_name]['orientation'][2]:
            bigc = 1
            s = s[:,:,:-1].copy()

        self.siend = self.igc + self.var['grid']['itot'] + bigc
        self.sjend = self.jgc + self.var['grid']['jtot'] + bjgc
        self.skend = self.kgc + self.var['grid']['ktot'] + bkgc
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Check that there are not more ghost cells than grid cells. 
        if (self.igc + bigc) > (s.shape[2]) or (self.jgc + bjgc) > (s.shape[1]) or (self.kgc + bkgc) > (s.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Initialize new output array including ghost cells
        sgc = np.zeros((kcells+bkgc, jcells+bjgc, icells+bigc))
        
        #Fill new initialzid array including ghost cells for horizontal directions
        #sgc[self.kgc:self.skend-bkgc, self.jgc:self.sjend-bjgc, self.igc:self.siend-bigc] = s[:,:,:].copy() NOTE: This line should NOT be commented out in case of periodic BC in vertical direction
        sgc[self.kgc:self.skend, self.jgc:self.sjend-bjgc, self.igc:self.siend-bigc] = s[:,:,:].copy() #compared to line above, -bkgc removed: top ghost cell already implemented (using no-slip BC rather than periodic BC) in the downsampling procedure
        sgc[:,:,0:self.igc] = sgc[:,:,self.siend-self.igc-bigc:self.siend-bigc] #Add ghostcell upstream x-direction
        sgc[:,:,self.siend-bigc:self.siend+self.igc] = sgc[:,:,self.igc:self.igc+self.igc+bigc] #Add ghostcell downstream x-direction
        sgc[:,0:self.jgc,:] = sgc[:,self.sjend-self.jgc-bjgc:self.sjend-bjgc,:] #Add ghostcell upstream y-direction
        sgc[:,self.sjend-bjgc:self.sjend+self.jgc,:] = sgc[:,self.jgc:self.jgc+self.jgc+bjgc,:] #Add ghostcell downstream y-direction
        #Fill new initialized array including ghost cells for vertical direction
        #NOTE1: assuming that U = 0 at the bottom, the code below mirrors the profile over the bottom/top BC.
        sgc[0:self.kgc,:,:] = 0 - np.flip(sgc[self.kgc+bkgc:2*self.kgc+bkgc,:,:], axis = 0) #Add ghostcells bottom z-direction
        sgc[self.skend:self.skend+self.kgc,:,:] = 0 - np.flip(sgc[self.skend-bkgc-self.kgc:self.skend-bkgc,:,:], axis = 0) #Add ghostcell top z-direction
#        if bkgc == 1:
#            sgc[self.skend-bkgc] = sgc[self.kgc,:,:] #Add top boundary when variable s is located on the grid_edges in the vertical direction
        #NOTE2: The two lines above should NOT be commented out in case of periodic BC. In case of periodic BC, the ghost cells can implemented in a similar way as in the horizontal directions.
        #NOTE3: The lines above are taking into account that in the downsampling procedure the top cell in the staggered vertical direction has already been added.

        #Store new fields in object
        self.var['output'][variable_name]['variable'] = sgc
        
    def __add_ghostcells_grid(self):
        '''Store new coordinates with ghost cells in object.'''

        #Get original coordinates
        z = self.var['grid']['z']
        zh = self.var['grid']['zh']
        y = self.var['grid']['y']
        yh = self.var['grid']['yh']
        x = self.var['grid']['x']
        xh = self.var['grid']['xh']

        #Check that the grid center and grid edges arrays defined above have the same shape (which is an implicit assumption in the code below).
        if (z.shape != zh.shape) or (y.shape != yh.shape) or (x.shape != xh.shape):
            raise RuntimeError("The shape of the arrays representing the grid centers and edges should be the same, but they are not while this implicitly assumed in the script." )
        
        #Check for each coordinate that no ghostcells are specified when no periodic bc are assumed. No ghost cells are yet implemented for other types of BC's, except one ghost cell in the vertical direction with a no-slip BC.
        #if self.kgc_edge > 0 and (not self.periodic_bc[0]):
        #    raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")
        
        #if self.kgc_center > 0 and (not self.periodic_bc[0]) and not (self.kgc_center == 1 and self.zero_w_topbottom):
        if self.kgc > 0 and not (self.periodic_bc[0] or self.zero_w_topbottom):
            raise ValueError("The current settings for the periodic_bc and zero_w_topbottom flags do not allow to implement the specified number of ghost cells in the vertical direction. Add additional functionality to this script if needed.")

        if self.jgc > 0 and not self.periodic_bc[1]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        if self.igc > 0 and not self.periodic_bc[2]:
            raise ValueError("Ghost cells need to be implemented while no periodic boundary conditions have been assumed. This has not been implemented yet.")

        #Check that there are not more ghost cells than grid cells. 
        #if (self.igc + 1) > (x.shape[0]) or (self.jgc + 1) > (y.shape[0]) or (max(self.kgc_center, self.kgc_edge) + 1) > (z.shape[0]):
        if (self.igc + 1) > (x.shape[0]) or (self.jgc + 1) > (y.shape[0]) or (self.kgc + 1) > (z.shape[0]):
            raise ValueError("The needed number of ghostcells is larger than the number of grid cells present in the object.") 
        
        #Determine size new output array including ghost cells
        icells = self.var['grid']['itot'] + 2*self.igc
        jcells = self.var['grid']['jtot'] + 2*self.jgc
      #  kcells = self.var['grid']['ktot'] + 2*kgc
        kcells = self.var['grid']['ktot'] + 2*self.kgc
        khcells = self.var['grid']['ktot'] + 2*self.kgc
        
        #Initialize new coordinates and store ghost cells in them
        xgc = np.zeros(icells)
        xhgc = np.zeros(icells+1)
        ygc = np.zeros(jcells)
        yhgc = np.zeros(jcells+1)
        zgc = np.zeros(kcells)
        zhgc = np.zeros(khcells+1)
        
        self.iend = self.igc + self.var['grid']['itot']
        self.ihend = self.igc + self.var['grid']['itot'] + 1
        self.jend = self.jgc + self.var['grid']['jtot']
        self.jhend = self.jgc + self.var['grid']['jtot'] + 1
        self.kend = self.kgc + self.var['grid']['ktot']
        self.khend = self.kgc + self.var['grid']['ktot'] + 1
        
        xgc  = self.__add_ghostcells_cor(xgc , 0,  self.igc, self.iend , x,  self.var['grid']['xsize'])
        xhgc = self.__add_ghostcells_cor(xhgc, 1,  self.igc, self.ihend, xh, self.var['grid']['xsize'])
        ygc  = self.__add_ghostcells_cor(ygc , 0,  self.jgc, self.jend , y,  self.var['grid']['ysize'])
        yhgc = self.__add_ghostcells_cor(yhgc, 1,  self.jgc, self.jhend, yh, self.var['grid']['ysize'])
        zgc  = self.__add_ghostcells_cor_ver(zgc , 0,  self.kgc, self.kend , z,  self.var['grid']['zsize'])
        zhgc = self.__add_ghostcells_cor_ver(zhgc, 1,  self.kgc, self.khend, zh, self.var['grid']['zsize'])

        self.var['grid']['z'] = zgc
        self.var['grid']['zh'] = zhgc
        self.var['grid']['y'] = ygc
        self.var['grid']['yh'] = yhgc
        self.var['grid']['x'] = xgc
        self.var['grid']['xh'] = xhgc
        
    def __add_ghostcells_cor(self, corgc, bgc, gc, endindex, cor, size):
        """ Add ghostcells to coordinates. """
        corgc[gc:endindex-bgc] = cor[:]
        if gc != 0:
            corgc[0:gc] = 0 - (size - corgc[endindex-bgc-gc:endindex-bgc])
        if (gc != 0) or (bgc != 0): 
            corgc[endindex-bgc:endindex+gc] = size + corgc[gc:gc+gc+bgc]
        return corgc
    
    def __add_ghostcells_cor_ver(self, corgc, bgc, gc, endindex, cor, size):
        """ Add 0 or 1 ghostcell to coordinates corresponding to the vertical direction, making use of the no-slip BC. """
        corgc[gc:endindex-bgc] = cor[:]
        if not (gc >= 0):
            raise RuntimeError("Number of ghost cells to be added in vertical direction not consistent with specified BC's.")
        if gc > 0 :
            corgc[0:gc] = 0 - np.flip(cor[0+bgc:gc+bgc])
            corgc[endindex:endindex+gc] = size + (size - np.flip(corgc[endindex-bgc-gc:endindex-bgc]))
        if bgc == 1: 
            corgc[endindex-bgc] = size
        return corgc
    
    def volume_integral(self, variable_name):
        """ Return volume integral for variable specified by variable name. (Defined for testing purposes.)"""
        
        #Check that variable_name is a string
        if not isinstance(variable_name, str):
            raise TypeError("Specified variable_name should be a string.")
        
        #Check whether variable specified via variable_name is defined in object
        if not variable_name in self.var['output'].keys():
            raise KeyError("Specified variable_name not defined in object.")
            
        #Read in variable values and axes, remove ghostcells, calculate grid distances for integration
        #NOTE: because in the next section np.trapz only integrates over the range specified by the coordinates, it is needed that all coordinates have the same range. This is done below by copying values of the variable.
        s = self.var['output'][variable_name]['variable']
        
        if self.var['output'][variable_name]['orientation'][0]:
            s = s[self.kgc:self.khend,:,:].copy()
            z_alt = self.var['grid']['z'][self.kgc:self.kend]
            z_diff = z_alt[1:] - z_alt[:-1]
            z_diff = np.insert(z_diff, 0, z_alt[0])
            z_diff = np.append(z_diff, self.var['grid']['zsize'] - z_alt[-1])
            
        else:
            s = s[self.kgc:self.kend,:,:].copy()
            z_alt = self.var['grid']['zh'][self.kgc:self.khend]
            z_diff = z_alt[1:] - z_alt[:-1]
            
        if self.var['output'][variable_name]['orientation'][1]:
            s = s[:,self.jgc:self.jhend,:].copy()
            y_alt = self.var['grid']['y'][self.jgc:self.jend]
            y_diff = y_alt[1:] - y_alt[:-1]
            y_diff = np.insert(y_diff, 0, y_alt[0])
            y_diff = np.append(y_diff, self.var['grid']['ysize'] - y_alt[-1])
            
        else:
            s = s[:,self.jgc:self.jend,:].copy()
            y_alt = self.var['grid']['yh'][self.jgc:self.jhend]
            y_diff = y_alt[1:] - y_alt[:-1]
        
        if self.var['output'][variable_name]['orientation'][2]:
            s = s[:,:,self.igc:self.ihend].copy()
            x_alt = self.var['grid']['x'][self.igc:self.iend]
            x_diff = x_alt[1:] - x_alt[:-1]
            x_diff = np.insert(x_diff, 0, x_alt[0])
            x_diff = np.append(x_diff, self.var['grid']['xsize'] - x_alt[-1])
            
        else:
            s = s[:,:,self.igc:self.iend].copy()
            x_alt = self.var['grid']['xh'][self.igc:self.ihend]
            x_diff = x_alt[1:] - x_alt[:-1]
            
        #Integrate over variable by multiplying with the corresponding grid distances and subsequent summing.
        s_intx = np.tensordot(s, z_diff, axes = (0,0))
        s_intyx = np.tensordot(s_intx, y_diff, axes = (0,0))
        s_intzyx = np.tensordot(s_intyx, x_diff, axes = (0,0))
        
        return s_intzyx
    
    def __edgegrid_from_centergrid(self, coord_center, len_coord, size_coord):
        """ Define coordinates corresponding to the grid walls from coordinates corresponding to the centers of the grid. """

        dcoord = coord_center[1:] - coord_center[:-1]
        coord_edge = np.empty(len_coord)
        coord_edge[0] = 0
        for i in range(1,len_coord):
            coord_edge[i] = coord_center[i-1] + 0.5 * dcoord[i-1]
        return coord_edge
 
    def __getitem__(self, name):
        if name in self.var.keys():
            return self.var[name]
        else:
            raise RuntimeError('Can\'t find variable \"{}\" in object.')
