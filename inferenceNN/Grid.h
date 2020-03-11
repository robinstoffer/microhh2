//Header file for grid class
#include <string>
#include <vector>
#ifndef GRID_H // Implement header guard
#define GRID_H

class Grid
{
	public:
		std::vector<float> m_zgc;
		std::vector<float> m_ygc;
		std::vector<float> m_xgc;
		std::vector<float> m_zhgc;
		std::vector<float> m_yhgc;
		std::vector<float> m_xhgc;
		float m_zsize;
		float m_ysize;
		float m_xsize;
		int m_kstart;
		int m_jstart;
		int m_istart;
		int m_kend;
		int m_jend;
		int m_iend;
		int m_khend;
		int m_kcells;
		int m_khcells;
		int m_jcells;
		int m_icells;
		int m_khtot;
		int m_ktot;
		int m_jtot;
		int m_itot;
		int m_ijcells;
		float m_dx;
		float m_dy;
		//Grid(); // constructor testing
		Grid(std::string grid_filenc); // constructor actual inference

		~Grid(); // destructor

	private:
		int m_ncid_grid;
};
#endif
