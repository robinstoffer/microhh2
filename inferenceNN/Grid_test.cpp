// Function for grid class, reads grid information from netCDF-file
#include <string>
#include <vector>
#include <iostream>
#include "Grid.h"

Grid::Grid()
{
	//Hard-code values for grid rather than reading them from a netCDF-file (as done in Grid class below)
	m_ktot = 64;
	m_khtot = 65;
	m_jtot = 48;
	m_itot = 96;
	m_kstart = 2;
	m_jstart = 2;
	m_istart = 2;
	m_kend = 66;
	m_jend = 50;
	m_iend = 98;
	m_khend = 67;
	m_dx = 0.06545f;
	m_dy = 0.06545f;

	// Define last grid variables based on the other ones
	m_kcells = m_kstart + m_kend;
	m_khcells = m_kstart + m_khend;
	m_jcells = m_jstart + m_jend;
	m_icells = m_istart + m_iend;
	m_ijcells = m_icells * m_jcells;
	m_dx = m_xsize / m_itot;
	m_dy = m_ysize / m_jtot;
}

Grid::~Grid()
{
}