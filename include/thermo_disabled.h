/*
 * MicroHH
 * Copyright (c) 2011-2019 Chiel van Heerwaarden
 * Copyright (c) 2011-2019 Thijs Heus
 * Copyright (c) 2014-2019 Bart van Stratum
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

#ifndef THERMO_DISABLED_H
#define THERMO_DISABLED_H

#include "thermo.h"

class Master;
class Input;
class Netcdf_handle;
template<typename> class Grid;
template<typename> class Stats;
template<typename> class Diff;
template<typename> class Column;
template<typename> class Dump;
template<typename> class Cross;
template<typename> class Field3d;
template<typename> class Thermo;
template<typename> class Timeloop;

template<typename TF>
class Thermo_disabled : public Thermo<TF>
{
    public:
        Thermo_disabled(Master&, Grid<TF>&, Fields<TF>&, Input&);
        virtual ~Thermo_disabled();

        // Interfacing functions to get buoyancy properties from other classes.
        bool check_field_exists(std::string name);

        // Empty functions that are allowed to pass.

        // Interfacing functions to get buoyancy properties from other classes.
        void init() {};
        void create(Input&, Netcdf_handle&, Stats<TF>&, Column<TF>&, Cross<TF>&, Dump<TF>&) {};
        void create_basestate(Input&, Netcdf_handle&) {};
        void load(const int) {};
        void save(const int) {};
        void exec(const double, Stats<TF>&) {};
        void exec_stats(Stats<TF>&) {};
        void exec_column(Column<TF>&) {};
        void exec_dump(Dump<TF>&, unsigned long) {};
        void exec_cross(Cross<TF>&, unsigned long) {};
        void get_mask(Stats<TF>&, std::string) {};
        bool has_mask(std::string) {return false;};
        void get_prog_vars(std::vector<std::string>&) {};
        void update_time_dependent(Timeloop<TF>&) {};
        int get_bl_depth() { throw std::runtime_error("Function get_bl_depth not implemented"); };

        TF get_buoyancy_diffusivity();

        unsigned long get_time_limit(unsigned long, double);

        #ifdef USECUDA
        void prepare_device() {};
        void clear_device() {};
        void forward_device() {};
        void backward_device() {};
        void get_thermo_field_g(Field3d<TF>&, const std::string&, const bool) {};
        void get_buoyancy_surf_g(Field3d<TF>&) {};
        void get_buoyancy_fluxbot_g(Field3d<TF>&) {};
        TF* get_basestate_fld_g(std::string) { throw std::runtime_error("Function get_basestate_fld_g not implemented"); };

        #endif

        // Empty functions that shall throw.
        void get_thermo_field(Field3d<TF>&, const std::string&, const bool, const bool)
        { throw std::runtime_error("Function get_thermo_field not implemented"); }
        void get_radiation_fields(
                Field3d<TF>&, Field3d<TF>&, Field3d<TF>&, Field3d<TF>&, Field3d<TF>&) const
        { throw std::runtime_error("Function get_radiation_fields not implemented"); }
        void get_buoyancy_surf(Field3d<TF>&, bool)
        { throw std::runtime_error("Function get_buoyancy_surf not implemented"); }
        void get_buoyancy_fluxbot(Field3d<TF>&, bool)
        { throw std::runtime_error("Function get_buoyancy_fluxbot not implemented"); }

        void get_T_bot(Field3d<TF>&, bool) { throw std::runtime_error("Function get_T_bot not implemented"); }
        const std::vector<TF>& get_p_vector() const { throw std::runtime_error("Function get_p_vector not implemented"); }
        const std::vector<TF>& get_ph_vector() const { throw std::runtime_error("Function get_ph_vector not implemented"); }
        const std::vector<TF>& get_exner_vector() const { throw std::runtime_error("Function get_exner_vector not implemented"); }
        TF get_db_ref() const { throw std::runtime_error("Function get_db_ref not implemented"); }

    private:
        using Thermo<TF>::swthermo;
};
#endif
