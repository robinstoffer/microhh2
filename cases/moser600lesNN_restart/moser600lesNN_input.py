import numpy
import netCDF4 as nc

float_type = 'f8'

# set the height
kmax  = 64
zsize = 2.

# define the variables
z = numpy.zeros(kmax)
u = numpy.zeros(kmax)
s = numpy.zeros(kmax)

#create equidistant grid
dz = zsize / kmax
z = numpy.arange(dz/2., zsize, dz)
s = z.copy()

# create initial parabolic shape
dpdxls = -3.0e-6
visc   =  1.0e-5
for k in range(kmax):
  u[k] = 1./(2.*visc)*dpdxls*(z[k]**2. - zsize*z[k])


# write the data to a file
nc_file = nc.Dataset("moser600lesNN_input.nc", mode="w", datamodel="NETCDF4", clobber=False)

nc_file.createDimension("z", kmax)
nc_z  = nc_file.createVariable("z" , float_type, ("z"))

nc_group_init = nc_file.createGroup("init");
nc_u = nc_group_init.createVariable("u", float_type, ("z"))
nc_s = nc_group_init.createVariable("s", float_type, ("z"))

nc_z[:] = z[:]
nc_u[:] = u[:]
nc_s[:] = s[:]

nc_file.close()
