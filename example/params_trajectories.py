import os
import datetime

# -- Initialisation parameters -- ##
# Parallelisation 
parallelisation = True
# Make grid or get grid from file
make_grid = False
# Grid parameters to initialize particule positions in degrees 
# [lonleft, lon_right, lon_step, lat_bottom, lat_top, lat_step]
#parameter_grid = (289.,309., 0.04,32.0, 42.375, 0.04)
parameter_grid = (300., 310., 0.25, 37., 42., 0.25)
parameter_grid = (120., 240., 5, 0, 60, 5)
#parameter_grid = (12., 24., 1, -60, -50, 1)

# If make_grid is True, specify parameters to build grid
# Define time output step (in days) for tracer
output_step = 1.
# Define box (lon_right, lon_left, lat_bottom, lat_top) to extract data on
# relevant area (box must be one degree bigger that area of interst to avoid
# boundary issues).
box = [280., 350., 10., 60.]
box = [-180, 180, -10, 70]
#box = [11., 25., -70., -40.]

# First time of advection (in CNES julian days)
reference = datetime.datetime(1970, 1,1)
first_day = datetime.datetime(2011,1 , 8) # 20819
first_date= datetime.datetime(2014, 7, 1) # 20819
last_date= datetime.datetime(2014,7, 8) # 20819
strday = first_day.strftime('%Y%m%d')

## -- ADVECTION PARAMETER -- ## 
# Time step for advection (in days)
adv_time_step = 0.11
# Time length of advection
tadvection = 2
# parameters for random walk to simulate diffusion
scale = 1.
# Diffusion parameter (sigma = 0.00000035)
sigma = 0
K = 0
# Diffusion and source and sink using low resolution tracer information
gamma = 0.
# Stationary flow: True or False
stationary = False

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
vel_input_dir = '/mnt/data/satellite/l4/cmems_015_004'
#pattern = 'dataset-uv-rep-hourly_'
pattern = 'dataset-uv-rep-daily_'
import re
#MATCH = re.compile(r"dataset-uv-rep-hourly_(\d{4})(\d{2})(\d{2})T(\d{4})Z_P\d{8}T(\d{4}).nc").search
MATCH = re.compile(r"dataset-uv-rep-daily_(\d{4})(\d{2})(\d{2})T(\d{4})Z_P\d{8}T(\d{4}).nc").search
# Class of the velocity file to read
vel_format = 'regular_netcdf'
# Name of velocity and coordinate variables
name_lon = 'longitude'
name_lat = 'latitude'
name_u = 'uo' #'ugos'
name_v = 'vo' #'vgos'
# List of velocity files to use
# List of velocity files to use

# Time step between two velocity files

# -- OUTPUTS -- ##
# Set value for nan
fill_value = -1e36
# Save full resolution trajectory: True or False
save_traj = True
# Save velocity (u, v) in file
save_U = True
save_V = True
# Save diagnostic parameters
save_S = False
save_RV = False
save_OW = False
# Set output file name and path
output_dir = '/mnt/data/'
test = 'aviso'
if stationary is True:
    test = 'aviso_stat'
start = first_date.strftime('%Y%m%d')
stop = last_date.strftime('%Y%m%d')
output_file = f'{test}_advection_{start}_{stop}.nc'
output = os.path.join(output_dir, output_file)
_ft = f'{test}_advection_20130101_20140701.nc'
filetrajectory =  os.path.join(output_dir, _ft)
