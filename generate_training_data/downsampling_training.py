#Script to downsample finegrid data to coarsegrid, which is needed for the generation of the training data for the NN.
#Author: Robin Stoffer (robin.stoffer@wur.nl)

#Developed for Python 3!
import numpy as np
import warnings

def generate_coarsecoord_centercell(cor_edges, cor_c_middle, dist_corc, finegrid):
    """ Function that downsamples a variable in finegrid, located at the center of the grid cells, to a coarse grid cell centered around cor_c_middle and with size dist_corc. Since the control volume of such a variable is defined by the edges of the grid cells, the cor_edges are used to select the correct cells of the fine grid and to calculate the corresponding weights (or to be more precise, fractions)."""

    dist_corc = np.round(dist_corc, finegrid.sgn_digits)
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    
    #Determine machine precision specified by finegrid with corresponding significant decimal digits
    sgn_digits = finegrid.sgn_digits
    cor_c_bottom = np.round(cor_c_bottom, sgn_digits)
    cor_c_top = np.round(cor_c_top, sgn_digits)
    cor_edges = np.round(cor_edges, sgn_digits)
    
    #Find points of fine grid located just outside the coarse grid cell considered in iteration
    cor_bottom = np.round(cor_edges[cor_edges <= cor_c_bottom].max(), sgn_digits)
    #cor_top = cor_c_top if iteration == (len_coordinate - 1) else cor_edges[cor_edges >= cor_c_top].min()
    cor_top = np.round(cor_edges[cor_edges >= cor_c_top].min(), sgn_digits)    

    #Select all points inside and just outside coarse grid cell
    points_indices_cor = np.logical_and(cor_bottom <= cor_edges , cor_edges < cor_top) #NOTE: cor_points includes the bottom boundary (cor_bottom), but not the top boundary (cor_top).
    cor_points = cor_edges[points_indices_cor]
    
    #Calculate weights for cor_points. 
    #NOTE: only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell.
    #NOTE: since cor_edges is one point longer than cor_center, the lengths of weights and points_indices_cor should be 1 shorter. 
    points_indices_cor = points_indices_cor[:-1].copy()
    len_weights = len(np.where(points_indices_cor)[0])
    weights = np.zeros(len_weights)
    if len(cor_points) == 1:
        weights = np.array([1])
        warnings.warn("Note that at certain coordinate interval(s) the defined fine grid has either the same or even a lower resolution than the defined coarse grid. The script applies in these instances a nearest neighbour procedure to fill in the values of the coarse grid.", RuntimeWarning)


    else:
        for i in range(len_weights):
            if i == 0:
                weights[0] = (cor_points[1] - cor_c_bottom)/(cor_c_top - cor_c_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_c_top - cor_points[i])/(cor_c_top - cor_c_bottom)
            else:
                weights[i] = (cor_points[i+1] - cor_points[i])/(cor_c_top - cor_c_bottom)

#    #Select two additional points just outside coarse grid cell, which are needed for interpolation (in the total transport calculation) but not for calculating representative velocities.
#    #Consequently, the weights are set to 0.
#    points_indices_cor = np.insert(points_indices_cor, 0, points_indices_cor[0] - 1)
#    points_indices_cor = np.append(points_indices_cor, point_indices_cor[-1] + 1)
#    weights = np.insert(weights, 0, 0)
#    weights = np.append(weights, 0)

    return weights, points_indices_cor

def generate_coarsecoord_edgecell(cor_center, cor_c_middle, dist_corc, finegrid, periodic_bc = True, zero_w_topbottom = True, size = 0): #For 'size' a default value is used that should not affect results as long as vert_flag = False.
    """ Function that downsamples a variable in finegrid, located at the edge of the grid cells, to a coarse grid cell centered around cor_c_middle and with size dist_corc. Since the control volume of such a variable is defined by the centers of the grid cells, the cor_center are used to select the correct cells of the fine grid and to calculate the corresponding weights (or to be more precise, fractions). NOTE: to deal with the edges of the domain, it needs to be specified whether periodic bc and zero_w_topbottom are present (True) or not (False). Furthermore, the size of the domain needs to be specified as well. """
    
    dist_corc = np.round(dist_corc, finegrid.sgn_digits)
    cor_c_bottom = cor_c_middle - 0.5*dist_corc
    cor_c_top = cor_c_middle + 0.5*dist_corc
    
    #Determine machine precision specified by finegrid with corresponding significant decimal digits
    sgn_digits = finegrid.sgn_digits
    cor_c_bottom = np.round(cor_c_bottom, sgn_digits)
    cor_c_top = np.round(cor_c_top, sgn_digits)
    cor_center = np.round(cor_center, sgn_digits)

    #Find points of fine grid located just outside the coarse grid cell considered in iteration, except when the top and bottom boundary in the vertical direction are considered (since in that direction no ghost cells are added).

    #Define function that adds an additional point to points_indices_cor, which is needed because cor_center is 1 point smaller than cor_edges
    def _add_points(cor_center, cor_c_top, points_indices_cor):
        if cor_c_top > cor_center[-1]: #Select top/downstream boundary when coarse grid top/downstream boundary exceeds the boundaries of the fine grid cell centers
            points_indices_cor = np.append(points_indices_cor, True)
        else:
            points_indices_cor = np.append(points_indices_cor, False)
        return points_indices_cor
    
    #Define function that selects correct points for coarse grid cells where periodic bc are not used for the downsampling from the fine grid.
    def _select_points(cor_center, cor_c_bottom, cor_c_top, sgn_digits, cor_bottom_defined = False, cor_top_defined = False):

        if not cor_bottom_defined and np.any(cor_center <= cor_c_bottom):
            cor_bottom = np.round(cor_center[cor_center <= cor_c_bottom].max(), sgn_digits)
        else:
            cor_bottom = cor_c_bottom
            
        if (not cor_top_defined) and np.any(cor_center >= cor_c_top):
            cor_top = np.round(cor_center[cor_center >= cor_c_top].min(), sgn_digits)
        else:
            cor_top = cor_c_top
            
        #Select all points inside and just outside coarse grid cell
        points_indices_cor = np.logical_and(cor_bottom<cor_center , cor_center<=cor_top) #NOTE: points_indices_cor includes the top boundary (cor_top), but not the bottom boundary (cor_bottom). 
        points_indices_cor = _add_points(cor_center, cor_c_top, points_indices_cor)
        
        return points_indices_cor
    
    #Select points of fine grid depending on coarse grid configuration and periodic bc.
    if periodic_bc and (cor_c_bottom < 0):
        #Select two different regions of coarse grid
        cor_c_bottom1 = 0.0
        cor_bottom1 = cor_c_bottom1
        cor_c_top1 = cor_c_top
        cor_top1 = np.round(cor_center[cor_center >= cor_c_top1].min(), sgn_digits)
        
        cor_c_bottom2 = np.round(cor_c_bottom + size, sgn_digits)
        cor_bottom2 = np.round(cor_center[cor_center <= cor_c_bottom2].max(), sgn_digits)
        cor_c_top2 = np.round(size, sgn_digits)
        cor_top2 = cor_c_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor1 = _add_points(cor_center, cor_c_top1, points_indices_cor1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor2 = _add_points(cor_center, cor_c_top2, points_indices_cor2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0] - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        two_boxes = True
        dirichlet_bc = False
    
    elif cor_c_bottom < 0:
        cor_c_bottom = 0.0
        points_indices_cor = _select_points(cor_center, cor_c_bottom, cor_c_top, sgn_digits, cor_bottom_defined = True)
        two_boxes = False
        dirichlet_bc = True
        
    elif periodic_bc and (cor_c_top > size):
        #Select two different regions of coarse grid
        cor_c_bottom1 = 0.0
        cor_bottom1 = cor_c_bottom1
        cor_c_top1 = np.round(cor_c_top - size, sgn_digits)
        cor_top1 = np.round(cor_center[cor_center >= cor_c_top1].min(), sgn_digits)
        
        cor_c_bottom2 = cor_c_bottom
        cor_bottom2 = np.round(cor_center[cor_center <= cor_c_bottom2].max(), sgn_digits)
        cor_c_top2 = np.round(size, sgn_digits)
        cor_top2 = cor_c_top2
        
        points_indices_cor1 = np.logical_and(cor_bottom1 < cor_center, cor_center <= cor_top1)
        points_indices_cor1 = _add_points(cor_center, cor_c_top1, points_indices_cor1)
        points_indices_cor2 = np.logical_and(cor_bottom2 < cor_center, cor_center <= cor_top2)
        points_indices_cor2 = _add_points(cor_center, cor_c_top2, points_indices_cor2)
        points_indices_cor = np.logical_or(points_indices_cor1, points_indices_cor2)
        index_bottom_coarse_cell = np.where(points_indices_cor2)[0][0]  - len(np.where(np.logical_and(np.logical_not(points_indices_cor1), np.logical_not(points_indices_cor2)))[0]) #Compensate for the indices that are not selected, cor_points has fewer indices than cor_center!
        index_top_coarse_cell = np.where(points_indices_cor1)[0][-1]
        two_boxes = True
        dirichlet_bc = False
                
    elif cor_c_top > size:
        cor_c_top = size
        points_indices_cor = _select_points(cor_center, cor_c_bottom, cor_c_top, sgn_digits, cor_top_defined = True)
        two_boxes = False
        dirichlet_bc = True
        
    else:
        points_indices_cor = _select_points(cor_center, cor_c_bottom, cor_c_top, sgn_digits)
        two_boxes = False
        dirichlet_bc = False

#    #Find points of fine grid located just outside the coarse grid cell considered in iteration
#    cor_bottom = cor_center[cor_center <= cor_c_bottom].max()
#    cor_top = cor_center[cor_center >= cor_c_top].min()
    
    #Select all cells that are within coarse grid cell
    cor_points = cor_center[points_indices_cor[:-1]] #NOTE: cor_center is one shorter than cor_edges, meaning last index point in array (i.e. [-1]) included earlier should be removed for correct selection cor_center.
    
    #Calculate weights for cor_points. 
    #NOTE: only the top and bottom fine grid cell may be PARTLY present in the corresponding coarse grid cell
    #NOTE: since cor_center is one point shorter than cor_edges, the lengths of weights and points_indices_cor should be 1 longer. The additional point in points_indices_cor is already added via the _add_points function.
    len_weights = len(np.where(points_indices_cor)[0])
    weights = np.zeros(len_weights)
    if zero_w_topbottom and dirichlet_bc:
        pass #Keep the weights at zero
    
    elif len_weights == 1:
        weights = np.array([1])
        warnings.warn("Note that at certain coordinate interval(s) the defined fine grid has either the same or even a lower resolution than the defined coarse grid. The script applies in these instances a nearest neighbour procedure to fill in the values of the coarse grid.", RuntimeWarning)
    
    elif two_boxes: #Deal with the cases where the selected fine grid cells consist of two separate regions (which occurs when periodic bc are imposed and the center of the coarse grid cell is located on the bottom/top of the domain).
        for i in range(len_weights):
            if i == 0 and i == index_top_coarse_cell:
                weights[i] = (cor_c_top1 - cor_c_bottom1)/(cor_c_top - cor_c_bottom)
            elif i == 0:
                weights[i] = (cor_points[i] - cor_c_bottom1)/(cor_c_top - cor_c_bottom)
            elif i == index_top_coarse_cell:
                weights[i] = (cor_c_top1 - cor_points[i-1])/(cor_c_top - cor_c_bottom)
            elif i == (len_weights - 1) and i == index_bottom_coarse_cell:
                weights[i] = (cor_c_top2 - cor_c_bottom2)/(cor_c_top - cor_c_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_c_top2 - cor_points[i-1])/(cor_c_top - cor_c_bottom)
            elif i == index_bottom_coarse_cell:
                weights[i] = (cor_points[i] - cor_c_bottom2)/(cor_c_top - cor_c_bottom)
            else:
                weights[i] = (cor_points[i] - cor_points[i-1])/(cor_c_top - cor_c_bottom)

    else:
        for i in range(len_weights):
            if i == 0:
                weights[i] = (cor_points[i] - cor_c_bottom)/(cor_c_top - cor_c_bottom)
            elif i == (len_weights - 1):
                weights[i] = (cor_c_top - cor_points[i-1])/(cor_c_top - cor_c_bottom)
            else:
                weights[i] = (cor_points[i] - cor_points[i-1])/(cor_c_top - cor_c_bottom)

#    #Select one additional point just upstream/down of coarse grid cell, which is needed for interpolation (in the total transport calculation) but not for calculating representative velocities.
#    #Consequently, the weight of this point set to 0. Note: interpolation not required at bottom boundary, since boundaries of coarse and fine grid exactly coincide there. Since no ghostcells are added in the vertical direction, this would be even impossible (i.e. giving an error message).
#    if not (vert_flag and (cor_c_bottom == 0)):
#        points_indices_cor = np.insert(points_indices_cor, 0, points_indices_cor[0] - 1)
#        weights = np.insert(weights, 0, 0)
#   # if not (vert_flag and (cor_c_top == size)):
#   #     points_indices_cor = np.append(points_indices_cor, point_indices_cor[-1] + 1)
#   #     weights = np.append(weights, 0)
    
    return weights, points_indices_cor

def downsample(finegrid, coarsegrid, variable_name, bool_edge_gridcell = (False, False, False), periodic_bc = (False, True, True), zero_w_topbottom = True):
    """Function to generate coarse grid with variables from a fine grid. Returns the specified variable on the coarse grid, together with the corresponding weights and coarse coordinates. The inputs are as follows:
    
    -variable_name: string that specifies the variable to calculate on the coarse grid.
    
    -bool_edge_gridcell: a tuple with a boolean for each spatial direction (z, y, x), indicating whether they should be aligned at the center of the grid cells (False) or the edges (True).
    
    -periodic_bc: a tuple with a boolean for each spatial direction (z, y, x), indicating  whether periodic boundary conditions are assumed (True when present, False when not present).
    
    -zero_w_topbottom: boolean specifying whether the vertical wind velocity is 0 at the bottom and top levels of the domain or not."""

    #Read in the right coarse coordinates determined by bool_edge_gridcell.
    #z-direction
    if bool_edge_gridcell[0]:
        zcor_c     = coarsegrid['grid']['zh'][coarsegrid.kgc:coarsegrid.khend]
        dist_zc    = coarsegrid['grid']['zhdist']
    else:
        zcor_c     = coarsegrid['grid']['z'][coarsegrid.kgc:coarsegrid.kend]
        dist_zc    = coarsegrid['grid']['zdist']

    #y-direction
    if bool_edge_gridcell[1]:
        ycor_c     = coarsegrid['grid']['yh'][coarsegrid.jgc:coarsegrid.jhend]
        dist_yc    = coarsegrid['grid']['yhdist']
    else:
        ycor_c     = coarsegrid['grid']['y'][coarsegrid.jgc:coarsegrid.jend]
        dist_yc    = coarsegrid['grid']['ydist']

    #x-direction
    if bool_edge_gridcell[2]:
        xcor_c     = coarsegrid['grid']['xh'][coarsegrid.igc:coarsegrid.ihend]
        dist_xc    = coarsegrid['grid']['xhdist']
    else:
        xcor_c     = coarsegrid['grid']['x'][coarsegrid.igc:coarsegrid.iend]
        dist_xc    = coarsegrid['grid']['xdist']
        
    #Check that variable_name is a string.
    if not isinstance(variable_name, str):
        raise TypeError("Specified variable_name should be a string.")

    #Check whether variable to be downsampled is contained in finegrid object associated with coarsegrid object.
    if variable_name not in finegrid['output'].keys(): 
        raise KeyError("Specified variable_name not defined in finegrid object on which this coarsegrid object is based.")

    #Check whether specified period_bc satisfies needed format and subsequently store it in object.
    if not isinstance(periodic_bc, tuple):
        raise TypeError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(periodic_bc) == 3:
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in periodic_bc):
        raise ValueError("Periodic_bc should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    #Check whether specified bool_edge_gridcell satisfies needed format and subsequently store it in object.
    if not isinstance(bool_edge_gridcell, tuple):
        raise TypeError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
        
    if not len(bool_edge_gridcell) == 3:
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")
            
    if not any(isinstance(flag, bool) for flag in bool_edge_gridcell):
        raise ValueError("Bool_edge_gridcell should be a tuple with length 3 (z, y, x), and consist only of booleans.")

    #Check that zero_w_topbottom is a boolean.
    if not isinstance(zero_w_topbottom, bool):
        raise TypeError('The zero_w_topbottom flag should be a boolean (True/False).')

    var_c = np.zeros((len(zcor_c), len(ycor_c), len(xcor_c)), dtype=float)
    #weights_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))
    #points_indices_c = np.zeros(len(zcor_c), len(ycor_c), len(xcor_c), dtype=(object, object, object))

    #Add needed ghostcells to finegrid object for the downsampling and calculation total transport
    #finegrid = add_ghostcells_finegrid(finegrid, coarsegrid, variable_name, bool_edge_gridcell)

    #Loop over coordinates for downsampling
    izc = 0
    for zcor_c_middle in zcor_c:
    #for izc in range(coarsegrid['grid']['ktot'])
        if bool_edge_gridcell[0]:
            weights_z, points_indices_z = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['z'][finegrid.kgc_center:finegrid.kend], cor_c_middle = zcor_c_middle, dist_corc = dist_zc, finegrid = finegrid, periodic_bc = periodic_bc[0], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['zsize'])
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc_edge:finegrid.khend, :, :]
        else:
            weights_z, points_indices_z = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['zh'][finegrid.kgc_edge:finegrid.khend], cor_c_middle = zcor_c_middle, dist_corc = dist_zc, finegrid = finegrid)
            var_finez = finegrid['output'][variable_name]['variable'][finegrid.kgc_center:finegrid.kend, :, :]

        var_finez = var_finez[points_indices_z,:,:]

        iyc = 0
	
        for ycor_c_middle in ycor_c:
            if bool_edge_gridcell[1]:
                weights_y, points_indices_y = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['y'][finegrid.jgc:finegrid.jend], cor_c_middle = ycor_c_middle, dist_corc = dist_yc, finegrid = finegrid, periodic_bc = periodic_bc[1], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['ysize'])
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jhend,:]
            else:
                weights_y, points_indices_y = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['yh'][finegrid.jgc:finegrid.jhend], cor_c_middle = ycor_c_middle, dist_corc = dist_yc, finegrid = finegrid)
                var_finezy = var_finez[:, finegrid.jgc:finegrid.jend,:]

            var_finezy = var_finezy[:,points_indices_y,:]

            ixc = 0
				
            for xcor_c_middle in xcor_c:
                if bool_edge_gridcell[2]:
                    weights_x, points_indices_x = generate_coarsecoord_edgecell(cor_center = finegrid['grid']['x'][finegrid.igc:finegrid.iend], cor_c_middle = xcor_c_middle, dist_corc = dist_xc, finegrid = finegrid, periodic_bc = periodic_bc[2], zero_w_topbottom = zero_w_topbottom, size = finegrid['grid']['xsize'])
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.ihend]
                else:
                    weights_x, points_indices_x = generate_coarsecoord_centercell(cor_edges = finegrid['grid']['xh'][finegrid.igc:finegrid.ihend], cor_c_middle = xcor_c_middle, dist_corc = dist_xc, finegrid = finegrid)
                    var_finezyx = var_finezy[:, :, finegrid.igc:finegrid.iend]
                    
                var_finezyx = var_finezyx[:,:,points_indices_x]

                #Calculate downsampled variable on coarse grid using the selected points in var_finezyx and the fractions defined in the weights variables
                weights =  weights_x[np.newaxis,np.newaxis,:]*weights_y[np.newaxis,:,np.newaxis]*weights_z[:,np.newaxis,np.newaxis]
                var_c[izc,iyc,ixc] = np.sum(np.multiply(weights, var_finezyx))
 
                #weights_c[izc,iyc,ixc] = (weights_z,weights_y,weights_x)
                #points_indices_c[izc,iyc,ixc] = (points_indices_z,points_indices_y,points_indices_x)

                ixc += 1
            iyc += 1
        izc += 1
    
#    #Store downsampled variable in coarsegrid object
#    coarsegrid['output'][variable_name] = var_c
 
    return var_c
    
