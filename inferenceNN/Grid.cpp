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
	size_t kcells = 0;
	size_t khcells = 0;
	size_t jcells = 0;
	size_t icells = 0;
	int varid_zgc = 0;
	int varid_zhgc = 0;
	int varid_ygc = 0;
	int varid_yhgc = 0;
	int varid_xgc = 0;
	int varid_xhgc = 0;
	int varid_kstart = 0;
	int varid_kend = 0;
	int varid_khend = 0;
	int varid_jstart = 0;
	int varid_jend = 0;
	int varid_istart = 0;
	int varid_iend = 0;
	int dimid_kcells = 0;
	int dimid_khcells = 0;
	int dimid_jcells = 0;
	int dimid_icells = 0;

	// Open nc-file  for reading
	if ((retval = nc_open(grid_filenc.c_str(), NC_NOWRITE, &m_ncid_grid)))
	{
		nc_error_print(retval);
	}

	// Get the varids of the variables based on their names
	if ((retval = nc_inq_varid(m_ncid_grid, "zgc", &varid_zgc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "zhgc", &varid_zhgc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "ygc", &varid_ygc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "yhgc", &varid_yhgc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "xgc", &varid_xgc)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_varid(m_ncid_grid, "xhgc", &varid_xhgc)))
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
	if ((retval = nc_inq_dimid(m_ncid_grid, "zgc", &dimid_kcells)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_kcells, &kcells)))
	{
		nc_error_print(retval);
	}
	m_kcells = static_cast<int>(kcells); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "zhgc", &dimid_khcells)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_khcells, &khcells)))
	{
		nc_error_print(retval);
	}
	m_khcells = static_cast<int>(khcells); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "ygc", &dimid_jcells)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_jcells, &jcells)))
	{
		nc_error_print(retval);
	}
	m_jcells = static_cast<int>(jcells); //Cast from size_t to int
	if ((retval = nc_inq_dimid(m_ncid_grid, "xgc", &dimid_icells)))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_inq_dimlen(m_ncid_grid, dimid_icells, &icells)))
	{
		nc_error_print(retval);
	}
	m_icells = static_cast<int>(icells); //Cast from size_t to int

	//Initialize std::vectors as here lengths are known runtime
	m_zgc.resize(m_kcells, 0);  //NOTE: initialize to 0.
	m_zhgc.resize(m_khcells, 0);
	m_ygc.resize(m_jcells, 0);
	m_yhgc.resize(m_jcells, 0);
	m_xgc.resize(m_icells, 0);
	m_xhgc.resize(m_icells, 0);

	// Read the grid information from the nc-file
	if ((retval = nc_get_var_float(m_ncid_grid, varid_zgc, &m_zgc[0])))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_float(m_ncid_grid, varid_zhgc, &m_zhgc[0])))
	{
		nc_error_print(retval);
	}
	if ((retval = nc_get_var_float(m_ncid_grid, varid_ygc, &m_ygc[0])))
	{
		nc_error_print(retval);
	}
	//
	size_t start_yh[2] = {};
	size_t count_yh[2] = {};
	count_yh[0] = jcells;
	count_yh[1] = 1;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_yhgc, start_yh, count_yh, &m_yhgc[0]))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}
	//
	if ((retval = nc_get_var_float(m_ncid_grid, varid_xgc, &m_xgc[0])))
	{
		nc_error_print(retval);
	}
	//
	size_t start_xh[2] = {};
	size_t count_xh[2] = {};
	count_xh[0] = icells;
	count_xh[1] = 1;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_xhgc, start_xh, count_xh, &m_xhgc[0]))) //NOTE: implicitly the ghost cell in the xh-direction is removed.
	{
		nc_error_print(retval);
	}


	// Extract grid sizes from nc-files by selecting the last coordinates in the zh-, yh-, and xh-direction
	size_t start_size[2] = {};
	size_t count_size[2] = {};
	count_size[0] = 1;
	count_size[1] = 1;
	start_size[0] = m_khcells - 1 - m_kstart;
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_zhgc, start_size, count_size, &m_zsize)))
	{
		nc_error_print(retval);
	}

	start_size[0] = m_jcells - m_jstart; // NOTE: index not -1 because yhc stored in nc-file is one value larger
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_yhgc, start_size, count_size, &m_ysize))) // NOTE: implicitly the ghost cell in the yh - direction is removed.
	{
		nc_error_print(retval);
	}

	start_size[0] = m_icells - m_istart; // NOTE: index not -1 because xhc stored in nc-file is one value larger
	if ((retval = nc_get_vara_float(m_ncid_grid, varid_xhgc, start_size, count_size, &m_xsize))) // NOTE: implicitly the ghost cell in the xh - direction is removed.
	{
		nc_error_print(retval);
	}
	
	
	// Define last grid variables based on the other ones
	m_ijcells = m_icells * m_jcells;
	m_khtot = m_khcells - 2 * m_kstart;
	m_ktot = m_kcells   - 2 * m_kstart;
	m_jtot = m_jcells - 2 * m_jstart;
	m_itot = m_icells - 2 * m_istart;
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