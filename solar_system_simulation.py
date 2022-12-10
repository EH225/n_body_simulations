# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations
"""

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt
from integrators import *
from simulation_utilities import *
from barnes_hut_algo import octant_node
import matplotlib.patches as mpatches

import requests, re
import numpy as np
import pandas as pd
import datetime
### Package Imports ###


def covert_to_float(x):
    try:
        return float(x)
    except:
        return None

def request_initial_conditions(celestial_bodies=None, start_time = "2022-Dec-1 17:00:00"):
        
    celestial_body_dict ={"Sun":"010","Mercury":"199","Venus":"299","Earth":"399","Moon":"301","Mars":"499",
                          "Jupiter":"599","Saturn":"699","Uranus":"799","Neptune":"899","Pluto":"999"}
    
    # Mass x10^24 (kg)
    mass_dict ={"Sun":1988500,"Mercury":0.3302,"Venus":4.8685,"Earth":5.97219,"Moon":0.007349,"Mars":0.64171,
                          "Jupiter":1898.18722,"Saturn":568.34,"Uranus":86.813,"Neptune":1024.09,"Pluto":0.01307}
    
    # Vol. mean radius (km)
    radius_dict ={"Sun":695700,"Mercury":2440,"Venus":6051.84,"Earth":6371.01,"Moon":1737.53,"Mars":3389.92,
                      "Jupiter":69911,"Saturn":58232,"Uranus":25362,"Neptune":24624,"Pluto":1188.3}
    
    if celestial_bodies is None: # If no list of celestial bodies is specified, then assume all major solar system objects
        celestial_bodies = celestial_body_dict.keys()
    
    # Create output aggregation data structures to hold the position and velocity of each body
    pos_arr = np.zeros([len(celestial_bodies),3])
    vel_arr = np.zeros([len(celestial_bodies),3])
    mass_arr = pd.Series([mass_dict[body] for body in celestial_bodies],index=celestial_bodies)
    radius_arr = pd.Series([radius_dict[body] for body in celestial_bodies],index=celestial_bodies)
    
    start_time_dt = pd.to_datetime(start_time) # Convert to pd datetime
    stop_time_dt = start_time_dt + datetime.timedelta(seconds=1) # Set the stop time to be 1 second ahead
    
    start_time_str = "'" + start_time_dt.strftime("%Y-%b-%d %H:%M:%S") + "'"
    stop_time_str ="'" + stop_time_dt.strftime("%Y-%b-%d %H:%M:%S") + "'"

    for j,body in enumerate(celestial_bodies):
        code = celestial_body_dict[body]
    
        # Build a URL to query the JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html
        #start_time = "'2022-Dec-13 17:30:00'"
        #stop_time = "'2022-Dec-13 17:31:00'"
        req_url = "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&"
        req_url += "START_TIME="+str(start_time_str)
        req_url += "&STOP_TIME="+str(stop_time_str)
        req_url += "&TABLE_TYPE='Vector'&REF_PLANE='Ecliptic'&CENTER='@010'&COMMAND='"+str(code)+"'"
        
        response = requests.get(req_url)
        assert response.status_code == 200, f"request status code not successful, got {response.status_code}"
        response_text = response.text # Extract the text from the response
        
        # Process the relevant text section containing the position data and the velocity data
        pos_vel_data = response_text[response_text.find("$$SOE"):response_text.find("$$EOE")]
        lines = pos_vel_data.split("\n") # Split into lines around the line break character
        pos_coords = lines[2].strip().replace("X","").replace("Y","").replace("Z","").replace("=","").split() 
        pos_coords = [covert_to_float(substr) for substr in pos_coords]# Convert to floats
        assert len(pos_coords) == 3,"3 velocity values not detected for "+str(body)
    
        vel_vals = lines[3].strip().replace("VX","").replace("VY","").replace("VZ","").replace("=","").split() 
        vel_vals = [covert_to_float(substr) for substr in vel_vals]# Convert to floats
        assert len(vel_vals) == 3,"3 velocity values not detected for "+str(body)

        pos_arr[j,:] = pos_coords
        vel_arr[j,:] = vel_vals
        
    # Convert to pd dataframes and add column / row index labels
    pos_arr = pd.DataFrame(pos_arr);pos_arr.index=celestial_bodies;pos_arr.columns = ["X (km)", "Y (km)", "Z (km)"]
    vel_arr = pd.DataFrame(vel_arr);vel_arr.index=celestial_bodies;vel_arr.columns = ["VX (km/s)", "VY (km/s)", "VZ (km/s)"]
    
    return pos_arr, vel_arr, mass_arr, radius_arr

#######################################
### Create Solar System Simulations ###
#######################################

celestial_bodies = ["Sun","Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune","Pluto"]

pos_arr, vel_arr, mass_arr, radius_arr = request_initial_conditions(celestial_bodies)
vel_arr_km_day = vel_arr*60*60*24 # Convert from km/s to km/day
true_G = 6.6743 * 10**(-11) # Big G or the gravitational coupling constant
# Perform unit conversions to get it into units that are conformable to the measurements we have from JPL
# i.e. (km^3)/(kg x10^24 x days^2)
sim_G = true_G *(10**24) * (1/1000)**3 * (60*60*24)**2

###                    ###
### Inner Solar System ###
###                    ###

# Plot the inner solar system objects: ["Sun","Mercury","Venus","Earth","Mars"]
pos_agg, KE_agg, PE_agg = run_simulation(N=5, T=365*10, dt=1, softening=0.001,
                                         normalize_momentum = False, initial_conditions={"mass":mass_arr.values.reshape(-1,1).copy(),
                                                                                         "vel":vel_arr_km_day.values.copy(),
                                                                                         "pos":pos_arr.values.copy()},
                                         G=sim_G, integrator=verlet_integrator, use_BH=False, random_state=111)

time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)

# 2d 10 earth years animation - Tail lengths represent 1 years of earth time
generate_simulation_video(pos_agg[:5,:,:], 30, 2, ['gold','darkgray','orange','blue','red'],
                          show_tails=True, tail_len=365, 
                          file_type="mp4", output_filename="inner_solar_sys_2d_10yr", grid=True,
                          set_lims = [-3*10**8, 3*10**8], annotations=celestial_bodies[:5],
                          s=np.log(radius_arr[:5].values*(30/6371.01))*15)

# 3d 10 earth years animation - Tail lengths represent 1 years of earth time
generate_simulation_video(pos_agg[:5,:,:], 30, 3, ['gold','darkgray','orange','blue','red'],
                          show_tails=True, tail_len=365, 
                          file_type="mp4", output_filename="inner_solar_sys_3d_10yr",
                          set_lims = [-3*10**8, 3*10**8],annotations=celestial_bodies[:5],
                          s=np.log(radius_arr[:5].values*(30/6371.01))*15)

###                   ###
### Full Solar System ###
###                   ###

# Plot the full solar system objects: ["Sun","Mercury","Venus","Earth","Mars"]
pos_agg, KE_agg, PE_agg = run_simulation(N=len(pos_arr), T=365*30, dt=5, softening=0.001,
                                         normalize_momentum = False, initial_conditions={"mass":mass_arr.values.reshape(-1,1).copy(),
                                                                                         "vel":vel_arr_km_day.values.copy(),
                                                                                         "pos":pos_arr.values.copy()},
                                         G=sim_G, integrator=verlet_integrator, use_BH=False, random_state=111)

time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)

# 2d 30 earth years animation - Tail lengths represent 5 years of earth time
generate_simulation_video(pos_agg, 30, 2, ['gold','darkgray','orange','blue','red','darkorange','yellow','blue','navy','navy'],
                          show_tails=True, tail_len=365, 
                          file_type="mp4", output_filename="full_solar_sys_2d_30yr", grid=True,
                          set_lims = [-5*10**9, 5*10**9], annotations=["","","","Earth","","Jupiter","Saturn","Uranus","Neptune","Pluto"],
                          s=np.log(radius_arr.values*(30/6371.01))*5)

# 3d 30 earth years animation - Tail lengths represent 5 years of earth time
generate_simulation_video(pos_agg, 30, 3, ['gold','darkgray','orange','blue','red','darkorange','yellow','blue','navy','navy'],
                          show_tails=True, tail_len=365, 
                          file_type="mp4", output_filename="full_solar_sys_3d_30yr",
                          set_lims = [-5*10**9, 5*10**9], annotations=["","","","Earth","","Jupiter","Saturn","Uranus","Neptune","Pluto"],
                          s=np.log(radius_arr.values*(30/6371.01))*5)
