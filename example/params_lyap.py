import os
import datetime

# -- Initialisation parameters -- ##
# Parallelisation 
parallelisation = True #True
# Make grid or get grid from file
make_grid = True
# Grid parameters to initialize particule positions in degrees 
# [lonleft, lon_right, lon_step, lat_bottom, lat_top, lat_step]
#parameter_grid = (289.,309., 0.10,32.0, 42.375, 0.10)
parameter_grid = (17., 35., 0.04, -43, -33, 0.04)
#parameter_grid = (337., 338., 0.04, 17, 18, 0.04)
# If make_grid is True, specify parameters to build grid
# Define box (lon_right, lon_left, lat_bottom, lat_top) to extract data on
# relevant area (box must be one degree bigger that area of interst to avoid
# boundary issues).
box = [10., 50., -50., -20.]
# First time of advection (in CNES julian days)
reference = datetime.datetime(1970, 1,1)
first_date = datetime.datetime(2020,2 , 12) # 20819
last_date = datetime.datetime(2020,1 , 14) # 20819

## -- ADVECTION PARAMETER -- ## 
# Time step for advection (in days)
adv_time_step = -0.11
# parameters for random walk to simulate diffusion
scale = 0.
# Diffusion parameter (sigma = 0.00000035)
sigma = 0
K = 0
# Diffusion and source and sink using low resolution tracer information
gamma = 0.
# Stationary flow: True or False
stationary = False

# -- Set up velocity for advection -- ##
# Path for data
vel_input_dir = '/mnt/data_b/satellite/l4/cmems_008_047/'
# Class of the velocity file to read
vel_format = 'regular_netcdf'
# Name of velocity and coordinate variables
name_lon = 'longitude'
name_lat = 'latitude'
name_u = 'ugos'
name_v = 'vgos'
# List of velocity files to use
pattern = 'dt_global_allsat_phy_l4_'
import re
MATCH = re.compile(r"dt_global_allsat_phy_l4_(\d{4})(\d{2})(\d{2})_(\d{8}).nc").search
depth = None
# Time step between two velocity files
vel_step = 1.

# -- FSLE -- ##
# Specify diagnostic, choices are 'FSLE', 'FTLE'
diagnostic = 'FSLE'
# Initial distance between particles
delta0 = 0.02
# Final distance between particles
deltaf = 1.0


# -- OUTPUTS -- ##
# Set value for nan
fill_value = -1e36
# Set output file name and path
output_dir = './'
test = f'{diagnostic}_{pattern}'
start = first_date.strftime('%Y%m%d')
output_file = f'{test}_{start}_stat{stationary}.nc'
output = os.path.join(output_dir, output_file)

