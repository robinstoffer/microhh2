// Function for grid class, reads grid information from netCDF-file
#include <string>
#include <vector>
#include <iostream>
#include "Grid.h"
extern "C"
{
#include <netcdf.h>
}

// Define status code for netCDF functions
static int retval = 0;

// Return this error code in event of a problem related to netCDF
static constexpr int NC_ERR = 2;

// Define error function for netCDF exceptions
inline static int nc_error_print(int e)
{
	std::cerr << "Error: " << nc_strerror(e);
	exit(NC_ERR);
}

Grid::Grid(std::string grid_filenc)
{
	//Read grid information from file

	//Define IDs for netCDF-file and variables for reading:
	m_ncid_grid = 0;
	size_t ktot = 0;
	size_t khtot = 0;
	size_t jtot = 0;
	size_t itot = 0;
	int varid_zc = 0;
	int varid_zhc = 0;
	int varid_yc = 0;
	int varid_yhc = 0;
	int varid_xc = 0;
	int varid_xhc = 0;
	int varid_kstart = 0;
	int varid_kend = 0;
	int varid_khend = 0;
	int varid_jstart = 0;
	int varid_jend = 0;
	int varid_istart = 0;
	int varid_iend = 0;
	int dimid_ktot = 0;
	int dimid_khtot = 0;
	int dimid_jtot = 0;
	int dimid_itot = 0;

	// Open nc-file  for reading
	if ((retval = nc_open(grid_filenc.c_str(), NC_NOWRITE, &m_ncid_grid)))
	{
		nc_error_print(retval);
	}

	// Get the varids of the variables based on their names
	if ((retval = nc_inq_varid(m_ncid_grid, "zc", &varid_zc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "zhc", &varid_zhc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "yc", &varid_yc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "yhc", &varid_yhc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "xc", &varid_xc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "xhc", &varid_xhc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "kgc_center", &varid_kstart))) //NOTE: kgc_center should be identical to kgc_edge
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "kend", &varid_kend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "khend", &varid_khend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "jgc", &varid_jstart)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "jend", &varid_jend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "igc", &varid_istart)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "iend", &varid_iend)))
	{
		nc_error_print(retval);
	}

	// Read the grid indices from the nc-file
	if ((retval = nc_get_var_int(m_ncid_grid, varid_kstart, &m_kstart)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_kend, &m_kend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_khend, &m_khend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_jstart, &m_jstart)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_jend, &m_jend)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_istart, &m_istart)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_int(m_ncid_grid, varid_iend, &m_iend)))
	{
		nc_error_print(retval);
	}

	//Extract dimension lengths from the nc-file
	if ((retval = nc_inq_dimid(m_ncid_grid, "zc", &dimid_ktot)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_ktot, &ktot)))
	{
		nc_error_print(retval);
	}
	m_ktot = static_cast<int>(ktot); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "zhc", &dimid_khtot)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_khtot, &khtot)))
	{
		nc_error_print(retval);
	}
	m_khtot = static_cast<int>(khtot); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "yc", &dimid_jtot)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_jtot, &jtot)))
	{
		nc_error_print(retval);
	}
	m_jtot = static_cast<int>(jtot); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "xc", &dimid_itot)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_itot, &itot)))
	{
		nc_error_print(retval);
	}
	m_itot = static_cast<int>(itot); //Cast from size_t to int

	//Initialize std::vectors as here lengths are known runtime
	m_zc.resize(m_ktot, 0);  //NOTE: initialize to 0.
	m_zhc.resize(m_khtot, 0);
	m_yc.resize(m_jtot, 0);
	m_yhc.resize(m_jtot, 0);
	m_xc.resize(m_itot, 0);
	m_xhc.resize(m_itot, 0);

	// Read the grid information from the nc-file
	if ((retval = nc_get_var_float(m_ncid_grid, varid_zc, &m_zc[0])))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_float(m_ncid_grid, varid_zhc, &m_zhc[0])))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_float(m_ncid_grid, varid_yc, &m_yc[0])))
	{
		nc_error_print(retval);
	}
	//
	size_t start_yh[2] = {};
	size_t count_yh[2] = {};
	count_yh[0] = jtot;
	count_yh[1] = 1;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_yhc, start_yh, count_yh, &m_yhc[0]))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}
	//
	if ((retval = nc_get_var_float(m_ncid_grid, varid_xc, &m_xc[0])))
	{
		nc_error_print(retval);
	}
	//
	size_t start_xh[2] = {};
	size_t count_xh[2] = {};
	count_xh[0] = itot;
	count_xh[1] = 1;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_xhc, start_xh, count_xh, &m_xhc[0]))) //NOTE: implicitly the ghost cell in the xh-direction is removed.
	{
		nc_error_print(retval);
	}


	// Extract grid sizes from nc-files by selecting the last coordinates in the zh-, yh-, and xh-direction
	size_t start_size[2] = {};
	size_t count_size[2] = {};
	count_size[0] = 1;
	count_size[1] = 1;
	start_size[0] = m_khtot - 1;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_zhc, start_size, count_size, &m_zsize))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}

	start_size[0] = m_jtot; // NOTE: index not -1 because yhc stored in nc-file is one value larger
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_yhc, start_size, count_size, &m_ysize))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}

	start_size[0] = m_itot; // NOTE: index not -1 because xhc stored in nc-file is one value larger
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_xhc, start_size, count_size, &m_xsize))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}
	
	
	// Define last grid variables based on the other ones
	m_kcells = m_kstart + m_kend;
	m_khcells = m_kstart + m_khend;
	m_jcells = m_jstart + m_jend;
	m_icells = m_istart + m_iend;
	m_ijcells = m_icells * m_jcells;
	m_ijtot = m_itot * m_jtot;
	m_dx = m_xsize / m_itot;
	m_dy = m_ysize / m_jtot;
}

Grid::~Grid()
{
	// Close nc-file with grid information if not in test mode
	if ((retval = nc_close(m_ncid_grid)))
	{
		nc_error_print(retval);
	}
}