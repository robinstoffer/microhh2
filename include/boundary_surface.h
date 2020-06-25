/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
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

#ifndef BOUNDARY_SURFACE_H
#define BOUNDARY_SURFACE_H

#include "boundary.h"
#include "stats.h"

template<typename> class Diff;

template<typename TF>
class Boundary_surface : public Boundary<TF>
{
    public:
        Boundary_surface(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Boundary_surface();

        void init(Input&, Thermo<TF>&);
        void create(Input&, Netcdf_handle&, Stats<TF>&);
        void set_values();

        void exec_stats(Stats<TF>&);
        void exec_cross(int);

        using Boundary<TF>::ustar;
        using Boundary<TF>::obuk;
        using Boundary<TF>::nobuk;
        using Boundary<TF>::z0m;
        using Boundary<TF>::z0h;

        using Boundary<TF>::ustar_g;
        using Boundary<TF>::obuk_g;
        using Boundary<TF>::nobuk_g;

        #ifdef USECUDA
        // GPU functions and variables
        void prepare_device();
        void clear_device();
        void forward_device();  // TMP BVS
        void backward_device(); // TMP BVS
        #endif

    protected:
        void process_input(Input&, Thermo<TF>&); // Process and check the surface input
        void init_surface(); // Allocate and initialize the surface arrays
        void init_solver(); // Prepare the lookup table's for the surface layer solver
        void set_ustar(); // Set fixed ustar

    private:
        using Boundary<TF>::master;
        using Boundary<TF>::grid;
        using Boundary<TF>::fields;
        using Boundary<TF>::boundary_cyclic;
        using Boundary<TF>::swboundary;
        using Boundary<TF>::field3d_io;

        using Boundary<TF>::process_bcs;

        using Boundary<TF>::mbcbot;
        using Boundary<TF>::ubot;
        using Boundary<TF>::vbot;

        typedef std::map<std::string, Field3dBc<TF>> BcMap;
        using Boundary<TF>::sbc;

        void update_bcs(Thermo<TF>&);

        TF ustarin;

        std::vector<float> zL_sl;
        std::vector<float> f_sl;

        #ifdef USECUDA
        float* zL_sl_g;
        float* f_sl_g;
        #endif

        Boundary_type thermobc;

    protected:
        // cross sections
        // std::vector<std::string> crosslist;        // List with all crosses from ini file
        // std::vector<std::string> allowedcrossvars; // List with allowed cross variables

        void update_slave_bcs();
};
#endif
