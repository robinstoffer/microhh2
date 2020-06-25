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

#ifndef BUDGET_2_H
#define BUDGET_2_H

#include "budget.h"

template<typename> class Field3d_operators;

template<typename TF>
class Budget_2 : public Budget<TF>
{
    public:
        Budget_2(Master&, Grid<TF>&, Fields<TF>&, Thermo<TF>&, Diff<TF>&, Advec<TF>&, Force<TF>&, Input&);
        ~Budget_2();

        void init();
        void create(Stats<TF>&);
        void exec_stats(Stats<TF>&);

    private:
        using Budget<TF>::master;
        using Budget<TF>::grid;
        using Budget<TF>::fields;
        using Budget<TF>::thermo;
        using Budget<TF>::diff;
        using Budget<TF>::advec;
        using Budget<TF>::force;

        Field3d_operators<TF> field3d_operators;

        std::vector<TF> umodel;
        std::vector<TF> vmodel;
        std::vector<TF> wmodel;

        /*
        void calc_kinetic_energy(double*, double*, const double*, const double*, const double*, const double*, const double*, const double, const double);

        void calc_advection_terms(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*,
                                  const double*, const double*, const double*, const double*, const double*,
                                  double*, double*, const double*, const double*); 

        void calc_advection_terms_scalar(double*, double*, double*, double*,
                                         const double*, const double*, const double*, const double*, const double*);

        void calc_pressure_terms(double*, double*, double*, double*, double*, 
                                 double*, double*, double*, double*,
                                 const double*, const double*, const double*, const double*,
                                 const double*, const double*, const double*, const double*,
                                 const double, const double);

        void calc_pressure_terms_scalar(double*, double*, 
                                        const double*, const double*, const double*, 
                                        const double*, const double*, const double*);

        void calc_diffusion_terms_DNS(double*, double*, double*, double*, double*, double*,
                                      double*, double*, double*, double*, double*, double*, double*,
                                      const double*, const double*, const double*, const double*,
                                      const double*, const double*, const double*,
                                      const double, const double, const double);

        void calc_diffusion_terms_scalar_DNS(double*, double*, double*, double*,
                                             const double*, const double*, const double*, const double*, const double*,
                                             const double, const double, const double, const double);

        void calc_diffusion_terms_LES(double*, double*, double*, double*, double*, double*,
                                      double*, double*, double*, double*, double*, double*,
                                      double*, double*, double*, double*, double*, double*,
                                      double*, double*, double*,
                                      const double*, const double*, const double*, const double*, const double*,
                                      const double*, const double*, const double*, const double*, const double*,
                                      const double, const double);

        void calc_buoyancy_terms(double*, double*, double*, double*,
                                 const double*, const double*, const double*, const double*, 
                                 const double*, const double*, const double*);

        void calc_buoyancy_terms_scalar(double*, const double*, const double*, const double*, const double*);

        void calc_coriolis_terms(double*, double*, double*, double*,
                                 const double*, const double*, const double*, 
                                 const double*, const double*, const double);
                                 */
};
#endif
