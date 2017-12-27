import os
# -- Initialisation parameters -- ##
# Parallelisation 
parallelisation = True
# Make grid or get grid from file
make_grid = True
# Grid parameters to initialize particule positions in degrees 
# [lonleft, lon_right, lon_step, lat_bottom, lat_top, lat_step]
#parameter_grid = (289.,309., 0.04,32.0, 42.375, 0.04)
parameter_grid = (300., 310., 0.25, 37., 42., 0.25)
# If make_grid is True, specify parameters to build grid
# Define time output step (in days) for tracer
output_step = 1.
# Define box (lon_right, lon_left, lat_bottom, lat_top) to extract data on
# relevant area (box must be one degree bigger that area of interst to avoid
# boundary issues).
box = [280., 350., 10., 60.]
# First time of advection (in CNES julian days)
first_day = 20819

# -- Initialize list of tracer for collocation purposes -- ##
# Provide list of tracer to collocate in time and space along the trajectory
list_tracer = None
# Provide list of grid of tracer to collocate
list_grid = None
# Provide number corresponding to grid for each tracer
list_num = None
# Input data directory for tracer to collocate
tracer_input_dir = './'
# Filter tracer 
tracer_filter = (0, 0)

# -- Set up velocity for advection -- ##
# Path for data
vel_input_dir = '/mnt/data/project/dimup/AVISO_tmp'
# Class of the velocity file to read
vel_format = 'regular_netcdf'
# Name of velocity and coordinate variables
name_lon = 'lon'
name_lat = 'lat'
name_u = 'u'
name_v = 'v'
# List of velocity files to use
list_vel = [ 'aviso_020819.nc',
          'aviso_020820.nc',
          'aviso_020821.nc',
          'aviso_020822.nc',
          'aviso_020823.nc',
          'aviso_020824.nc',
          'aviso_020825.nc',
          'aviso_020826.nc',
          'aviso_020827.nc',
          'aviso_020828.nc',
          'aviso_020829.nc',
          'aviso_020830.nc',
          'aviso_020831.nc',
          'aviso_020832.nc',
          'aviso_020833.nc',
          'aviso_020834.nc',
          'aviso_020835.nc',
          'aviso_020836.nc',
          'aviso_020837.nc',
          'aviso_020838.nc',
          'aviso_020839.nc',
          'aviso_020840.nc',
          'aviso_020841.nc',
          'aviso_020842.nc',
          'aviso_020843.nc',
          'aviso_020844.nc',
          'aviso_020845.nc',
          'aviso_020846.nc',
          'aviso_020847.nc',
          'aviso_020848.nc',
          'aviso_020849.nc',
          'aviso_020850.nc',
          'aviso_020851.nc',
          'aviso_020852.nc',
          'aviso_020853.nc',
          'aviso_020854.nc',
          'aviso_020855.nc',
          'aviso_020856.nc',
          'aviso_020857.nc',
          'aviso_020858.nc',
          'aviso_020859.nc',
          'aviso_020860.nc',
          'aviso_020861.nc',
          'aviso_020862.nc',
          'aviso_020863.nc',
          'aviso_020864.nc',
          'aviso_020865.nc',
          'aviso_020866.nc',
          'aviso_020867.nc',
          'aviso_020868.nc',
          'aviso_020869.nc',
          'aviso_020870.nc',
          'aviso_020871.nc',
          'aviso_020872.nc',
          'aviso_020873.nc',
          'aviso_020874.nc',
          'aviso_020875.nc',
          'aviso_020876.nc',
          'aviso_020877.nc',
          'aviso_020878.nc',
          'aviso_020879.nc',
          'aviso_020880.nc',
          'aviso_020881.nc',
          'aviso_020882.nc',
          'aviso_020883.nc',
          'aviso_020884.nc',
          'aviso_020885.nc',
          'aviso_020886.nc',
          'aviso_020887.nc',
          'aviso_020888.nc',
          'aviso_020889.nc',
          'aviso_020890.nc',
          'aviso_020891.nc',
          'aviso_020892.nc',
          'aviso_020893.nc',
          'aviso_020894.nc',
          'aviso_020895.nc',
          'aviso_020896.nc',
          'aviso_020897.nc',
          'aviso_020898.nc',
          'aviso_020899.nc'
          ]
# Time step between two velocity files
vel_step = 1.

## -- ADVECTION PARAMETER -- ## 
# Time step for advection (in days)
adv_time_step = 0.11
# Time length of advection
tadvection = 79
# parameters for random walk to simulate diffusion
scale = 1.
# Diffusion parameter (sigma = 0.00000035)
sigma = 0
K = 0
# Diffusion and source and sink using low resolution tracer information
gamma = 0.
# Stationary flow: True or False
stationary = False

# -- OUTPUTS -- ##
# Set value for nan
fill_value = -1e36
# Save full resolution trajectory: True or False
save_traj = True
# Save velocity (u, v) in file
save_U = True
save_V = True
# Save diagnostic parameters
save_S = True
save_RV = True
save_OW = True
# Set output file name and path
output_dir = './'
test = 'ow'
start = int(first_day)
stop = int(first_day + tadvection)
output_file = f'{test}_{start}_{stop}.nc'
output = os.path.join(output_dir, output_file)

