# -*- coding: utf-8 -*-
"""
Authors: Eric Helmold, Alex Ho, Ivonne Martinez
AM 111 Final Project: N-body Simulations

DOC: 11/05/2022
Last Updated: 12/07/2022
Purpose: Replicate the work found here: 
    https://medium.com/swlh/create-your-own-n-body-simulation-with-python-f417234885e9
    https://github.com/pmocz/nbody-python
    https://github.com/techwithtim/Python-Planet-Simulation/blob/main/tutorial.py
    
    Potential Energy: https://www.school-for-champions.com/science/gravitation_potential_energy.htm
    Creating a simulation video: https://www.youtube.com/watch?v=WTLPmUHTPqo
    Simulation videos: https://www.geeksforgeeks.org/moviepy-creating-animation-using-matplotlib/
    Simulation Videos: https://stackoverflow.com/questions/34975972/how-can-i-make-a-video-from-array-of-images-in-matplotlib
"""

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt
from integrators import *
from simulation_utilities import *
from barnes_hut_algo import octant_node
### Package Imports ###

##########################
### N-Body Simulations ###
##########################

###                           ###
### Single cluster simulation ###
###                           ###
pos_agg, KE_agg, PE_agg = run_simulation(N=50, T=20, dt=0.01, softening=0.1, G=1, integrator=leap_frog_integrator, use_BH=False, random_state=111)
time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)
generate_simulation_video(pos_agg, 20, 2, ['blue'], show_tails=True, file_type="mp4", output_filename="particle_sim_n50")

###                      ###
### 2-cluster simulation ###
###                      ###
N = 50 
pos_agg, KE_agg, PE_agg = run_simulation(N=N, T=10, dt=0.01, softening=0.1, G=1,
                                         initial_conditions={"mass":20.0*np.ones((N,1))/N,
                                                             "pos":np.concatenate([np.random.normal(2,0.5,N//2*3).reshape(N//2,3),np.random.normal(-2,0.5,N//2*3).reshape(N//2,3)]),
                                                             "vel":np.random.randn(N,3)},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111)

time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)

generate_simulation_video(pos_agg, 10, 3, ['red']*25+['blue']*25,show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3), s=[150],
                          file_type="mp4", output_filename="two_cluster_particle_simulation_3d")


###                                         ###
### 2 planet, 1 sun stable orbit Simulation ###
###                                         ###
N = 3;v0=5.15
pos_agg, KE_agg, PE_agg = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111)

time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)

generate_simulation_video(pos_agg, 10, 2, ['green','red','blue'],show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3),
                          s=np.array([150*5,150,150]), file_type="mp4", output_filename="solar_system_2_planets_2d")

generate_simulation_video(pos_agg, 10, 3, ['green','red','blue'],show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3),
                          s=np.array([150*5,150,150]), file_type="mp4", output_filename="solar_system_2_planets_3d")


###                                         ###
### 4 planet, 1 sun stable orbit Simulation ###
###                                         ###
N = 5;v0=7;v1=5.5
pos_agg, KE_agg, PE_agg = run_simulation(N=5, T=20, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([100,20,20,20,20]).astype(np.float16).reshape(5,1),
                                                             "pos":np.array([[0,0,0],
                                                                             [-2,0,0],[2,0,0],
                                                                             [0,5,0],[0,-5,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],
                                                                             [0,v0,0],[0,-v0,0],
                                                                             [v1,0,0],[-v1,0,0]]).astype(np.float16)},
                                         integrator=leap_frog_integrator, random_state=111)

time_axis = np.arange(len(KE_agg))*0.01;plot_simulation_energy(time_axis,KE_agg,PE_agg)

generate_simulation_video(pos_agg, 20, 2, ['red','green','green','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-6,6),
                          ylim=(-6,6), zlim=(-6,6), s=np.array([150*5,150,150,150,150]), file_type="mp4", output_filename="solar_system_4_planets_2d")

generate_simulation_video(pos_agg, 20, 3, ['red','green','green','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150,150]), file_type="mp4", output_filename="solar_system_4_planets_3d")



###############################################
### Comparing Various Time-Step Integrators ###
###############################################
N = 3;v0=5.15 # Euler's Method
pos_agg_e, KE_agg_e, PE_agg_e = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=euler_integrator, use_BH=False, random_state=111)
# Euler-Cromer
pos_agg_ec, KE_agg_ec, PE_agg_ec = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=euler_cromer_integrator, use_BH=False, random_state=111)
# Leap-Frog
pos_agg_lf, KE_agg_lf, PE_agg_lf = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111)

# Euler Richardson
pos_agg_er, KE_agg_er, PE_agg_er = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=euler_richardson_integrator, use_BH=False, random_state=111)

# Runge-Kutta 4
pos_agg_rk4, KE_agg_rk4, PE_agg_rk4 = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=RK4_integrator, use_BH=False, random_state=111)


# Generate simulation comparison videos
pos_agg = np.concatenate([pos_agg_e,pos_agg_ec],axis=0)
green_patch = mpatches.Patch(color='green', label='Euler');blue_patch = mpatches.Patch(color='blue', label='Euler-Cromer')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="Euler_vs_Euler_Cromer_Method",legend=[green_patch,blue_patch])

pos_agg = np.concatenate([pos_agg_ec,pos_agg_lf],axis=0)
green_patch = mpatches.Patch(color='green', label='Euler-Cromer');blue_patch = mpatches.Patch(color='blue', label='Leap Frog')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="Euler_Cromer_vs_Leap_Frog_Method",legend=[green_patch,blue_patch])

pos_agg = np.concatenate([pos_agg_er,pos_agg_lf],axis=0)
green_patch = mpatches.Patch(color='green', label='Euler-Richardson');blue_patch = mpatches.Patch(color='blue', label='Leap Frog')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="Euler_Richardson_vs_Leap_Frog_Method",legend=[green_patch,blue_patch])

pos_agg = np.concatenate([pos_agg_er,pos_agg_lf],axis=0)
green_patch = mpatches.Patch(color='green', label='Euler-Richardson');blue_patch = mpatches.Patch(color='blue', label='Leap Frog')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="Euler_Richardson_vs_Leap_Frog_Method",legend=[green_patch,blue_patch])

pos_agg = np.concatenate([pos_agg_rk4,pos_agg_lf],axis=0)
green_patch = mpatches.Patch(color='green', label='RK4');blue_patch = mpatches.Patch(color='blue', label='Leap Frog')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="RK4_vs_Leap_Frog_Method",legend=[green_patch,blue_patch])


###################################
### Testing the Barnes-Hut Algo ###
###################################

from time import time

#### Using the standard approach computing all particle interactions

start_time = time() # Not using Barnes-Hut, for a relatively small number of particles (n=50)
pos_agg, KE_agg, PE_agg = run_simulation(N=50, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 0.16213035583496094 (leap frog), 0.045989990234375 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a moderate number of particles (n=500)
pos_agg, KE_agg, PE_agg = run_simulation(N=500, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 25.875255584716797 (leap frog), 7.08690071105957 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a large number of particles (n=5000)
pos_agg, KE_agg, PE_agg = run_simulation(N=5000, T=0.125, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 44.27939558029175 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a very large number of particles (n=10000)
pos_agg, KE_agg, PE_agg = run_simulation(N=10000, T=0.05, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 77.6750750541687 (Euler-Cromer)


#### Using the Barnes-Hut algorithm all particle interactions

start_time = time() # Using Barnes-Hut, for a relatively small number of particles (n=50)
pos_agg, KE_agg, PE_agg = run_simulation(N=50, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 18.157663583755493 (leap frog), 2.2221713066101074 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a moderate number of particles (n=500)
pos_agg, KE_agg, PE_agg = run_simulation(N=500, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 104.12792992591858 (leap frog), 62.24120903015137 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a large number of particles (n=5000)
pos_agg, KE_agg, PE_agg = run_simulation(N=5000, T=0.125, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 68.89757561683655 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a very large number of particles (n=10000)
pos_agg, KE_agg, PE_agg = run_simulation(N=10000, T=0.05, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 75.06097078323364 (Euler-Cromer)

### Testing the Barnes-Hut algo vs the standard approach calculating all forces ###
N=10;np.random.seed(111) # Set the number of simulation particles to 10 and a random seed for replicability of results
mass = 20.0*np.ones((N,1))/N  # Total mass of all particles together is 20, evenly split
pos  = np.random.randn(N,3)   # Randomly generate initial positions using N(0,1)
vel  = np.random.randn(N,3)   # Randomly generate initial velocities using N(0,1)
G = 1 # Set the gravitational constant

acc_calc1 = compute_acceleration(pos, mass, G, softening=0.1) # Compute acceleration using the full set of all pairwise interactions
acc_calc2 = compute_acceleration_BH(pos, mass, G, softening=0.1, theta=0.0) # Calculate acceleration instead using the Barnes-Hut algorithm
# Specifically, use a value of theta = 0 so that there is no reduction occuring, BH should in that case return the exact same results as the original
assert np.allclose(acc_calc1,acc_calc2) # Check that computing acceleration due to gravity is the same using either method

# Now increase the value of theta and see how much the acceleration values differ relative to the original approaching using MAP
values_of_theta = np.linspace(0,5) # The values of theta to iterate over and create a comparison with
acc_MAP = [abs(acc_calc1-compute_acceleration_BH(pos, mass, G, softening=0.1, theta=theta)).mean() for theta in values_of_theta]

plt.figure(figsize=(8,6));AttributeErrorplt.plot(values_of_theta,acc_MAP,zorder=3,color='darkorange')
plt.xlabel("Value of Theta");plt.ylabel("MAP");plt.title("Mean Absolute Error (MAP) of Acc. Calc. - BH Tree Calc")
plt.grid(color='lightgray',zorder=-3) # As expected, when theta is very close to zero, there is no difference, as theta increases, the error
# also increases up to a point at which it plateaus since all particles summarized as 1 cernter of mass is an upper limit of aggregation
