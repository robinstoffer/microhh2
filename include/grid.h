/*
 * MicroHH
 * Copyright (c) 2011-2017 Chiel van Heerwaarden
 * Copyright (c) 2011-2017 Thijs Heus
 * Copyright (c) 2014-2017 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GRID
#define GRID

#ifdef USEMPI
#include <mpi.h>
#endif

#include <string>
#include <vector>
#include "defines.h"
#include "transpose.h"

class Master;
class Input;
class Data_block;

template<typename TF>
struct Grid_data
{
    int itot; ///< Total number of grid cells in the x-direction.
    int jtot; ///< Total number of grid cells in the y-direction.
    int ktot; ///< Total number of grid cells in the z-direction.
    int ntot; ///< Total number of grid cells.

    int imax; ///< Number of grid cells in the x-direction for one process.
    int jmax; ///< Number of grid cells in the y-direction for one process.
    int kmax; ///< Number of grid cells in the z-direction for one process.
    int nmax; ///< Number of grid cells for one process.

    int iblock; ///< Number of grid cells in the x-direction for calculation blocks in transposes.
    int jblock; ///< Number of grid cells in the y-direction for calculation blocks in transposes.
    int kblock; ///< Number of grid cells in the z-direction for calculation blocks in transposes.

    int igc; ///< Number of ghost cells in the x-direction.
    int jgc; ///< Number of ghost cells in the y-direction.
    int kgc; ///< Number of ghost cells in the z-direction.

    int icells;  ///< Number of grid cells in the x-direction including ghost cells for one process.
    int jcells;  ///< Number of grid cells in the y-direction including ghost cells for one process.
    int ijcells; ///< Number of grid cells in the xy-plane including ghost cells for one process.
    int kcells;  ///< Number of grid cells in the z-direction including ghost cells for one process.
    int ncells;  ///< Total number of grid cells for one process including ghost cells.
    int istart;  ///< Index of the first grid point in the x-direction.
    int jstart;  ///< Index of the first grid point in the y-direction.
    int kstart;  ///< Index of the first grid point in the z-direction.
    int iend;    ///< Index of the last gridpoint+1 in the x-direction.
    int jend;    ///< Index of the last gridpoint+1 in the y-direction.
    int kend;    ///< Index of the last gridpoint+1 in the z-direction.

    TF xsize; ///< Size of the domain in the x-direction.
    TF ysize; ///< Size of the domain in the y-direction.
    TF zsize; ///< Size of the domain in the z-direction.

    TF dx;     ///< Distance between the center of two grid cell in the x-direction.
    TF dy;     ///< Distance between the center of two grid cell in the y-direction.
    TF dxi;    ///< Reciprocal of dx.
    TF dyi;    ///< Reciprocal of dy.

    std::vector<TF> dz;    ///< Distance between the center of two grid cell in the z-direction.
    std::vector<TF> dzh;   ///< Distance between the two grid cell faces in the z-direction.
    std::vector<TF> dzi;   ///< Reciprocal of dz.
    std::vector<TF> dzhi;  ///< Reciprocal of dzh.
    std::vector<TF> dzi4;  ///< Fourth order gradient of the distance between cell centers to be used in 4th-order schemes.
    std::vector<TF> dzhi4; ///< Fourth order gradient of the distance between cell faces to be used in 4th-order schemes.

    TF dzhi4bot;
    TF dzhi4top;

    std::vector<TF> x;  ///< Grid coordinate of cell center in x-direction.
    std::vector<TF> y;  ///< Grid coordinate of cell center in y-direction.
    std::vector<TF> z;  ///< Grid coordinate of cell center in z-direction.
    std::vector<TF> xh; ///< Grid coordinate of cell faces in x-direction.
    std::vector<TF> yh; ///< Grid coordinate of cell faces in x-direction.
    std::vector<TF> zh; ///< Grid coordinate of cell faces in x-direction.

    int ithread_block; ///< Number of grid cells in the x-direction for GPU thread block.
    int jthread_block; ///< Number of grid cells in the y-direction for GPU thread block.

    TF* z_g;
    TF* zh_g;
    TF* dz_g;
    TF* dzh_g;
    TF* dzi_g;
    TF* dzhi_g;
    TF* dzi4_g;
    TF* dzhi4_g;
};

/**
 * Class for the grid settings and operators.
 * This class contains the grid properties, such as dimensions and resolution.
 * The public funcions in this class contain operations that are called from many routines
 * in order to interpolate, transpose and save data. The MPI operations that work over multiple
 * processes on the entire grid are contained in this class.
 */
template<typename TF>
class Grid
{
    public:
        Grid(Master&, Input&); ///< Constructor of the grid class.
        ~Grid();               ///< Destructor of the grid class.

        void init();              ///< Initialization of the grid arrays.
        void create(Data_block&); ///< Creation of the grid data.
        void save();              ///< Saves grid data to file.
        void load();              ///< Loads grid data to file.

        TF utrans; ///< Galilean transformation velocity in x-direction.
        TF vtrans; ///< Galilean transformation velocity in y-direction.

        std::string swspatialorder; ///< Default spatial order of the operators to be used on this grid.

        const Grid_data<TF>& get_grid_data();

        void set_minimum_ghost_cells(int, int, int);

        // MPI functions
        void init_mpi(); ///< Creates the MPI data types used in grid operations.
        void exit_mpi(); ///< Destructs the MPI data types used in grid operations.

        // void get_max (double*);      ///< Gets the maximum of a number over all processes.
        // void get_max (int*);         ///< Gets the maximum of a number over all processes.
        // void get_sum (double*);      ///< Gets the sum of a number over all processes.
        // void get_prof(double*, int); ///< Averages a vertical profile over all processes.
        // void calc_mean(double*, const double*, int);

        // IO functions
        int save_field3d(TF*, TF*, TF*, char*, const TF); ///< Saves a full 3d field.
        int load_field3d(TF*, TF*, TF*, char*, const TF); ///< Loads a full 3d field.

        int save_xz_slice(TF*, TF*, char*, int);           ///< Saves a xz-slice from a 3d field.
        int save_yz_slice(TF*, TF*, char*, int);           ///< Saves a yz-slice from a 3d field.
        int save_xy_slice(TF*, TF*, char*, int kslice=-1); ///< Saves a xy-slice from a 3d field.
        int load_xy_slice(TF*, TF*, char*, int kslice=-1); ///< Loads a xy-slice.

        // interpolation functions
        void interpolate_2nd(TF*, const TF*, const int[3], const int[3]); ///< Second order interpolation
        //void interpolate_4th(double*, double*, const int[3], const int[3]); ///< Fourth order interpolation

        // GPU functions
        void prepare_device();                          ///< Load the arrays onto the GPU
        void clear_device();                            ///< Deallocate the arrays onto the GPU

    private:
        Master& master; ///< Reference to master class.
        Transpose<TF> transpose;

        bool mpitypes;  ///< Boolean to check whether MPI datatypes are created.

        void calculate(); ///< Computation of dimensions, faces and ghost cells.
        void check_ghost_cells(); ///< Check whether slice thickness is at least equal to number of ghost cells.

        void save_grid(); ///< Save the grid properties.
        void load_grid(); ///< Load the grid properties.

        Grid_data<TF> gd;

        #ifdef USEMPI
        // MPI Datatypes
        MPI_Datatype subi;       ///< MPI datatype containing a subset of the entire x-axis.
        MPI_Datatype subj;       ///< MPI datatype containing a subset of the entire y-axis.
        MPI_Datatype subarray;   ///< MPI datatype containing the dimensions of the total array that is contained in one process.
        MPI_Datatype subxzslice; ///< MPI datatype containing only one xz-slice.
        MPI_Datatype subyzslice; ///< MPI datatype containing only one yz-slice.
        MPI_Datatype subxyslice; ///< MPI datatype containing only one xy-slice.
        #endif
};
#endif
