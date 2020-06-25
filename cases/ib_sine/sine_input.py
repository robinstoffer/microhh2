import numpy as np
import matplotlib.pylab as pl
import netCDF4 as nc

from microhh_tools import *    # available in MICROHH_DIR/python; copy to this directory

# Class to create a smooth stretched vertical grid
class Grid:
    def __init__(self, kmax, nloc1, nbuf1, dz1, dz2):
        dn         = 1./kmax
        n          = np.linspace(dn, 1.-dn, kmax)
        nloc1     *= dn
        nbuf1     *= dn
        dzdn1      = dz1/dn
        dzdn2      = dz2/dn

        dzdn       = dzdn1 + 0.5*(dzdn2-dzdn1)*(1. + np.tanh((n-nloc1)/nbuf1))
        self.dz    = dzdn*dn

        self.kmax  = kmax
        self.z     = np.zeros(self.dz.size)
        stretch    = np.zeros(self.dz.size)

        self.z[0]  = 0.5*self.dz[0]
        stretch[0] = 1.

        for k in range(1, self.kmax):
              self.z [k] = self.z[k-1] + 0.5*(self.dz[k-1]+self.dz[k])
              stretch[k] = self.dz[k]/self.dz[k-1]

        self.zh = np.insert(np.cumsum(self.dz), 0, 0)
        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]
        print('kmax={0:}, zsize={1:.4f}'.format(kmax,self.zsize))

    def plot(self):
        pl.figure()
        pl.plot(self.z, self.dz, '-x')
        pl.xlabel('z (m)')
        pl.ylabel('dz (m)')

if __name__ == "__main__":
    pl.close('all')
    pl.ion()

    # MicroHH-mode (np.float32 (single) or np.float64 (double) precision)
    tf = np.float64

    # Read the .ini file
    ini = Read_namelist()

    # Settings sine hills (all in units [m])
    amplitude = 0.00254
    wavelength_x = 0.0508
    wavelength_y = 0
    z_offset = 0.002

    # Create stretched grid
    #grid = Grid(16, 40, 5, 0.0008, 0.0015)
    #grid = Grid(96, 40, 5, 0.0004, 0.0007)
    #grid = Grid(128, 40, 5, 0.0002, 0.0005373)
    grid = Grid(256, 122, 10, 0.0001, 0.000322)
    #grid = Grid(384, 180, 20, 0.00006, 0.00021831)
    grid.plot()

    print('Effective zsize = {}'.format(grid.zsize-z_offset-amplitude))

    # Write `zsize` and `ktot` back to .ini file
    replace_namelist_value('zsize', grid.zsize)
    replace_namelist_value('ktot',  grid.kmax)

    # Create initial profiles:
    z = grid.z
    u = 0.00137 * np.ones(z.size)
    b = z

    # Write the data to a .nc file for MicroHH
    float_type = 'f8' if tf==np.float64 else 'f4'

    nc_file = nc.Dataset("sine_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
    nc_file.createDimension("z", grid.kmax)
    nc_z  = nc_file.createVariable("z" , float_type, ("z"))

    nc_group_init = nc_file.createGroup("init");
    nc_u  = nc_group_init.createVariable("u" , float_type, ("z"))
    nc_b  = nc_group_init.createVariable("b" , float_type, ("z"))

    nc_z[:] = grid.z [:]
    nc_u[:] = u[:]
    nc_b[:] = b[:]

    nc_file.close()

    # Create 2D height map
    dx = ini['grid']['xsize'] / ini['grid']['itot']
    dy = ini['grid']['ysize'] / ini['grid']['jtot']

    x  = np.arange(0.5*dx, ini['grid']['xsize'], dx)
    xh = np.arange(0, ini['grid']['xsize']+1e-9, dx)

    y  = np.arange(0.5*dy, ini['grid']['ysize'], dy)
    yh = np.arange(0, ini['grid']['ysize']+1e-9, dy)

    dem = np.zeros((ini['grid']['itot'], ini['grid']['jtot']), dtype=tf)

    for j in range(ini['grid']['jtot']):
        dem[:,j] = z_offset + amplitude + amplitude * np.sin(2*np.pi*x/wavelength_x)

    pl.figure()
    ax=pl.subplot(111, aspect='equal')
    pl.plot(x, dem[:,0])


    for k in range(z.size):
        pl.plot(xh, np.ones_like(xh)*grid.zh[k], 'k-', linewidth=0.5)
        pl.scatter(x, np.ones_like(x)*grid.z[k], s=1, color='r')
    for i in range(x.size):
        pl.plot(np.ones_like(grid.zh)*xh[i], grid.zh, 'k-', linewidth=0.5)


    dem.T.tofile('dem.0000000')

    # Create example spatially varying sbot files
    mask = dem > dem.mean()
    b    = np.zeros_like(dem)

    b[mask]  = -0.1
    b[~mask] = 0.1

    b.T.tofile('b_sbot.0000000')
