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
pos_agg, realism_metrics = run_simulation(N=5, T=365*10, dt=1, softening=0.001,
                                         normalize_momentum = False, initial_conditions={"mass":mass_arr.values.reshape(-1,1).copy(),
                                                                                         "vel":vel_arr_km_day.values.copy(),
                                                                                         "pos":pos_arr.values.copy()},
                                         G=sim_G, integrator=verlet_integrator, use_BH=False, random_state=111,return_realism_metrics=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])
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
pos_agg, realism_metrics = run_simulation(N=len(pos_arr), T=365*30, dt=5, softening=0.001,
                                         normalize_momentum = False, initial_conditions={"mass":mass_arr.values.reshape(-1,1).copy(),
                                                                                         "vel":vel_arr_km_day.values.copy(),
                                                                                         "pos":pos_arr.values.copy()},
                                         G=sim_G, integrator=verlet_integrator, use_BH=False, random_state=111,return_realism_metrics=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])

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


##############################################
### Hypothetical Large Solar System Models ###
##############################################

# Helper function for generating initial conditions related to solar systems
def generate_solar_system(N:int=50, G:float=1, include_sun:bool=True, sun_mass:float=10000.0,
                          mass_bounds:tuple=(2,10), radius_bounds:tuple=(40,150), sigma_z:float=3.0, rotations:tuple=(0,0,0),
                          show_pos_plot:bool=True):
    """
    Function to generate the initial conditions of a solar system or galaxy. Return the pos, vel, mass arrays of a randomly
    initialized n-body system based on specified seed criteria. Function can return a system with or without a central mass
    e.g. a sun or black hole by using the center of mass of the particles collectively instead. Particles are randomly generated
    in a circular pattern about the central mass which can be given a thickness and rotation.

    Parameters
    ----------
    N : float, optional
        The number of planets/solar systems to include around the center of mass. The default is 50.
    G : float, optional
        Newton's Gravitational constant. The default is 1.
    include_sun : bool, optional
        A toggle for whether to include a central mass in the system. The default is True.
    sun_mass : float, optional
        The mass of the central body e.g. the sun or a black hole. The default is 10000.0.
    mass_bounds : tuple, optional
        A length 2 tuple describing the bounds of random mass generation for each of the N bodies. The default is (2,10).
    radius_bounds : tuple, optional
        A length 2 tuple describing the bounds of random radius generation for each of the N bodies. The default is (40,150).
    sigma_z : float, optional
        A float parameter describing how much variance for a normal distribution used to randomly generate z-coordinates. The default is 3.0.
    rotations : tuple, optional
        A length 3 tuple describing the rotations of N bodies about the x, y, and z axes respectively in radians. The default is (0,0,0).
    show_pos_plot : bool, optional
        A toggle for if the starting positions of the n-bodies should be displayed as a quick visual check of the particle orientations

    Returns
    -------
    pos : np.array
        A [N x 3] matrix of position coordinates (x, y, z)
    vel : np.array
        A [N x 3] matrix of velocities for each particle (vx, vy, vz)
    mass : np.array
        A [N x 1] vector of particle masses
    """
    if include_sun==True:
        assert sun_mass>0 and type(sun_mass)==float, "With include_sun==True, sun_mass must be >0 and a float"
    assert type(N)==int, "N, the number of particles, must be an int"
    assert mass_bounds[0]>0 and mass_bounds[1]>mass_bounds[0], "mass_bounds must be positive and sequentially larger"
    
    mass = np.random.uniform(*mass_bounds,N) # Randomly generate planetary masses uniformly according to the mass bounds
    assert (mass>0).all(), "mass values < 0 detected"
    theta = np.random.uniform(0,2*np.pi,N) # Generate random theta values for each particle (polar coordinates)
    r = np.random.uniform(*radius_bounds,N) # Generate random radius values for each particle (polar coordinates)
    z_vals = np.random.normal(0,sigma_z,N) # Generate random z-values for the position coords, gives the system some depth
    x, y = polar_to_xy(r,theta) # Convert from (r,theta) polar coordinates into (x,y) cartesian coordinates
    pos = np.vstack([x,y,z_vals]).T # Combine to create (x,y,z) initial conditions for each planet
    # Apply rotational transformations to the starting positions and velocities to give the solar system a tilt
    assert len(rotations) == 3, "rotations must be a length 3 tuple"
    theta_x, theta_y, theta_z = rotations # Unpack into the rotations about each axis in radians
    # Create rotation matricies
    x_rotation_matrix = np.array([[1,0,0],
                          [0,np.cos(theta_x),np.sin(theta_x)],
                          [0,-np.sin(theta_x),np.cos(theta_x)]])

    y_rotation_matrix = np.array([[np.cos(theta_y),0,np.sin(theta_y)],
                                  [0,1,0],
                                  [-np.sin(theta_y),0,np.cos(theta_y)]])
    
    z_rotation_matrix = np.array([[np.cos(theta_z),-np.sin(theta_z),0],
                                  [np.sin(theta_z),np.cos(theta_z),0],
                                  [0,0,1]])
    
    # Apply the rotation matrices to the particle positions
    pos = (z_rotation_matrix @ y_rotation_matrix @ x_rotation_matrix @ pos.T).T 
    
    if include_sun==False: # If no sun is in the system, then use the system's center of mass as a replacement
        sun_mass = mass.sum()
    
    vel_t = np.sqrt(G*sun_mass/r) # Tangential velocity needed for a stable orbit: v = sqrt(GM/r), assumes M is very large in comparison
    # Swap x and y and flip the sign on y to give us the velocity decomposition perpendicular to the radius to the orgin
    vel = np.vstack([-vel_t*np.sin(theta),vel_t*np.cos(theta),np.random.normal(0,0.2,N)]).T # Convert to (x,y,z) velocity for each particle
    
    # Apply the rotation matrices to the particle velocities so that they are moving about the center of mass in the correct directions
    vel = (z_rotation_matrix @ y_rotation_matrix @ x_rotation_matrix @ vel.T).T 

    if include_sun==True: # Append to mass, pos, vel the values of the Sun in the center of the solar system / galaxy structure
        mass = np.insert(mass,0,sun_mass).reshape(-1,1) # Add the sun's mass
        pos = np.insert(pos,np.array([0]),0,axis=0) # Add the sun's starting position in the center at the origin
        vel = np.insert(vel,np.array([0]),0,axis=0) # Add the sun's starting velocity of zero
    
    if show_pos_plot == True: # Plot the initial positions generated for the n-body system for visual inspection
        fig, ax = plt.subplots(figsize=(8,6)) # Create a plotting space for the inital positions
        ax = fig.add_subplot(projection='3d') # Create 3d projection sub-axis
        ax.scatter(pos[:,0],pos[:,1],pos[:,2]) # Plot the positions of all the starting particles in 3d
        ax_lims = abs(pos).max() # Find the max value of the particle positions so that all particles can be viewed in frame
        ax.set_ylim(-ax_lims,ax_lims);ax.set_zlim(-ax_lims,ax_lims);ax.set_xlim(-ax_lims,ax_lims) # Set axis limits
    
    # Returns initial conditions of the generated a solar system
    return pos, vel, mass.reshape(-1,1)

# Demo the generate_solar_system() functionality
pos, vel, mass = generate_solar_system(N=500)

# Run the simulation, compute the position evolution over time of each particle in the system
pos_agg = run_simulation(N=pos.shape[0], T=10, dt=0.05, softening=3, G=1, integrator=leap_frog_integrator,
                                          initial_conditions={"mass":mass,"pos":pos.copy(),"vel":vel.copy()}, 
                                          normalize_momentum=False, use_BH=False, random_state=111, return_realism_metrics=False)

# Generate a video of the simulation using the masses to create different sized particles
generate_simulation_video(pos_agg, 10, 3, ["red"]+['navy']*(pos.shape[0]-1), show_tails=False, file_type="mp4",
                          s=[500]+list(mass[1:,].reshape(-1)*(8/mass[1:,].mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="solar_system_random_seed_with_sun")


# Demo the generate_solar_system() functionality with no sun included in the middle
pos, vel, mass = generate_solar_system(N=50,include_sun=False,mass_bounds=(50,100),radius_bounds=(100,150))

# Run the simulation, compute the position evolution over time of each particle in the system
pos_agg = run_simulation(N=pos.shape[0], T=30, dt=0.05, softening=3, G=1, integrator=leap_frog_integrator,
                                          initial_conditions={"mass":mass,"pos":pos.copy(),"vel":vel.copy()}, 
                                          normalize_momentum=False, use_BH=False, random_state=111, return_realism_metrics=False)

generate_simulation_video(pos_agg, 10, 3, ['navy']*(pos.shape[0]), show_tails=False, file_type="mp4",
                          s=list(mass.reshape(-1)*(8/mass.mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="solar_system_random_seed_without_sun")


###                               ###
### Large Solar System Simulation ###
###                               ###
N = 700
pos, vel, mass = generate_solar_system(N=700, G=1, include_sun=True, sun_mass=80000.0,
                          mass_bounds=(2,10), radius_bounds=(40,150), sigma_z=3.0, rotations=(-np.pi/6,-np.pi/6,0))

pos_agg, realism_metrics = run_simulation(N=(N+1), T=30, dt=0.01, softening=3, G=1, integrator=leap_frog_integrator,
                                          initial_conditions={"mass":mass,"pos":pos.copy(),"vel":vel.copy()}, 
                                          normalize_momentum=True,
                                          use_BH=False, random_state=111, return_realism_metrics=True)
    
time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])

# Generate 2d and 3d animations
generate_simulation_video(pos_agg, 30, 3, ["red"]+['navy']*N, show_tails=False, file_type="mp4",
                          s=[500]+list(mass[1:,].reshape(-1)*(8/mass[1:,].mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="large_solar_sys_sim_3d_no_tails")

generate_simulation_video(pos_agg, 30, 3, ["red"]+['navy']*N, show_tails=True, file_type="mp4",
                          s=[500]+list(mass[1:,].reshape(-1)*(8/mass[1:,].mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="large_solar_sys_sim_3d")

generate_simulation_video(pos_agg, 30, 2, ["red"]+['navy']*N, show_tails=True, file_type="mp4",
                          s=[500]+list(mass[1:,].reshape(-1)*(8/mass[1:,].mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="large_solar_sys_sim_2d")


###                                        ###
### Large Solar System Run for a Long TIme ###
###                                        ###

N = 700
pos, vel, mass = generate_solar_system(N=700, G=1, include_sun=True, sun_mass=80000.0,
                          mass_bounds=(2,5), radius_bounds=(20,150), sigma_z=3.0, rotations=(-np.pi/6,-np.pi/6,0))

pos_agg = run_simulation(N=(300+1), T=80, dt=0.01, softening=3, G=1, integrator=leap_frog_integrator,
                                          initial_conditions={"mass":mass,"pos":pos.copy(),"vel":vel.copy()}, 
                                          normalize_momentum=False,
                                          use_BH=False, random_state=111, return_realism_metrics=False)

generate_simulation_video(pos_agg, 10, 3, ["red"]+['navy']*N, show_tails=False, file_type="mp4",
                          s=[500]+list(mass[1:,].reshape(-1)*(8/mass[1:,].mean())),
                          tail_len=25, set_lims=(-150,150),output_filename="large_solar_sys_sim_fast")



###                                   ###
### Galaxy Collision Simulation Model ###
###                                   ###

N = 500

### Galaxy 1 ###
pos_1, vel_1, mass_1 = generate_solar_system(N=500, G=1, include_sun=True, sun_mass=10000.0,
                          mass_bounds=(2,10), radius_bounds=(40,150), sigma_z=3.0, rotations=(0,0,0))

# Move the position of everything in Galaxy 1 over to the left and add a net velocity to the right
pos_1 = pos_1 - np.array([100,100,0])
vel_1 = vel_1 + np.array([5,10,0])

### Galaxy 2 ###
pos_2, vel_2, mass_2 = generate_solar_system(N=500, G=1, include_sun=True, sun_mass=10000.0,
                          mass_bounds=(2,10), radius_bounds=(40,150), sigma_z=3.0, rotations=(0,0,0))

# Move the position of everything in Galaxy 2 over to the right and add a net velocity to the left
pos_2 = pos_2 + np.array([100,100,0])
vel_2 = vel_2 - np.array([5,10,0])

# Concatenate together the starting positions, starting velocities and masses of each particle
pos = np.concatenate([pos_1,pos_2]);vel = np.concatenate([vel_1,vel_2]);mass = np.concatenate([mass_1,mass_2])


# Run the simulation, compute the position evolution of both galaxy clusters over time
pos_agg = run_simulation(N=(N+1), T=20, dt=0.01, softening=3, G=1, integrator=leap_frog_integrator,
                                          initial_conditions={"mass":mass,"pos":pos.copy(),"vel":vel.copy()}, 
                                          normalize_momentum=False, use_BH=False, random_state=111, return_realism_metrics=False)
    
generate_simulation_video(pos_agg, 20, 3, ["red"]+['navy']*N+["darkorange"]+['green']*N, show_tails=False, file_type="mp4",
                          s=[500]+list(mass_1[1:,].reshape(-1)*(8/mass_1[1:,].mean()))+[500]+list(mass_2[1:,].reshape(-1)*(8/mass_2[1:,].mean())),
                          tail_len=25, set_lims=(-200,200),output_filename="galaxy_collision_test_3d_large_dims")

