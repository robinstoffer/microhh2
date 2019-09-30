// Header file for diff_u.cpp
#include "Network.h"
#include "Grid.h"
#define restrict __restrict
#ifndef DIFF_U_H
#define DIFF_U_H

void select_box(
	const float* restrict const field_var,
	float* restrict const box_var,
	const int k_center,
	const int j_center,
	const int i_center,
	const int boxsize,
	const int skip_firstx,
	const int skip_lastx,
	const int skip_firsty,
	const int skip_lasty,
	const int skip_firstz,
	const int skip_lastz,
	const Grid& grid
);

void diff_U(
	const float* restrict const u,
	const float* restrict const v,
	const float* restrict const w,
	const float* restrict const dzi,
	const float* restrict const dzhi,
	float* restrict const ut,
	float* restrict const vt,
	float* restrict const wt,
	const Grid& grid,
	Network& MLP
);
#endif
