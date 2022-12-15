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
import matplotlib.patches as mpatches
### Package Imports ###

##########################
### N-Body Simulations ###
##########################

###                           ###
### Single cluster simulation ###
###                           ###
pos_agg, realism_metrics = run_simulation(N=50, T=20, dt=0.01, softening=0.1, G=1, integrator=leap_frog_integrator, use_BH=False, random_state=111, return_realism_metrics=True)
time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])
generate_simulation_video(pos_agg, 20, 2, ['blue'], show_tails=True, file_type="mp4", output_filename="particle_sim_n50")

# Using leap frog integration
pos_agg, realism_metrics = run_simulation(N=50, T=10, dt=0.01, softening=0.1, G=1, integrator=leap_frog_integrator, use_BH=False, random_state=111, return_realism_metrics=True)
time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])
generate_simulation_video(pos_agg, 10, 2, ['red'], show_tails=True, file_type="mp4", output_filename="leapfrog_integration")

# Using Verlet integration
pos_agg, realism_metrics = run_simulation(N=50, T=10, dt=0.01, softening=0.1, G=1, integrator=verlet_integrator, use_BH=False, random_state=111, return_realism_metrics=True)
time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])
generate_simulation_video(pos_agg, 10, 2, ['red'], show_tails=True, file_type="mp4", output_filename="verlet_integration")


###                      ###
### 2-cluster simulation ###
###                      ###
N = 50 
pos_agg, realism_metrics = run_simulation(N=N, T=10, dt=0.01, softening=0.1, G=1,
                                         initial_conditions={"mass":20.0*np.ones((N,1))/N,
                                                             "pos":np.concatenate([np.random.normal(2,0.5,N//2*3).reshape(N//2,3),np.random.normal(-2,0.5,N//2*3).reshape(N//2,3)]),
                                                             "vel":np.random.randn(N,3)},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111, return_realism_metrics=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])
generate_simulation_video(pos_agg, 10, 3, ['red']*25+['blue']*25,show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3), s=[150],
                          file_type="mp4", output_filename="two_cluster_particle_simulation_3d")


###                                         ###
### 2 planet, 1 sun stable orbit Simulation ###
###                                         ###
N = 3;v0=5
pos_agg, realism_metrics = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111, return_realism_metrics=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])

generate_simulation_video(pos_agg, 10, 2, ['green','red','blue'],show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3),
                          s=np.array([150*5,150,150]), file_type="mp4", output_filename="solar_system_2_planets_2d")

generate_simulation_video(pos_agg, 10, 3, ['green','red','blue'],show_tails=True, figsize=(10,10), xlim = (-3,3), ylim=(-3,3),
                          s=np.array([150*5,150,150]), file_type="mp4", output_filename="solar_system_2_planets_3d")


###                                         ###
### 4 planet, 1 sun stable orbit Simulation ###
###                                         ###
N = 5;v0=7;v1=5.5
pos_agg, realism_metrics = run_simulation(N=5, T=20, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([100,20,20,20,20]).astype(np.float16).reshape(5,1),
                                                             "pos":np.array([[0,0,0],
                                                                             [-2,0,0],[2,0,0],
                                                                             [0,5,0],[0,-5,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],
                                                                             [0,v0,0],[0,-v0,0],
                                                                             [v1,0,0],[-v1,0,0]]).astype(np.float16)},
                                         integrator=leap_frog_integrator, random_state=111, return_realism_metrics=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])

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

# Verlet
pos_agg_v, KE_agg_v, PE_agg_v = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":np.array([50,20,20]).astype(np.float16).reshape(3,1),
                                                             "pos":np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16),
                                                             "vel":np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)},
                                         integrator=verlet_integrator, use_BH=False, random_state=111)

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

pos_agg = np.concatenate([pos_agg_v,pos_agg_lf],axis=0)
green_patch = mpatches.Patch(color='green', label='Verlet');blue_patch = mpatches.Patch(color='blue', label='Leap Frog')
generate_simulation_video(pos_agg, 10, 3, ['red','green','green','red','blue','blue'],show_tails=True, figsize=(10,10), xlim = (-5,5),
                          ylim=(-5,5), zlim=(-5,5), s=np.array([150*5,150,150,150*5,150,150]), file_type="mp4",
                          output_filename="Verlet_vs_Leap_Frog_Method",legend=[green_patch,blue_patch])

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
pos_agg = run_simulation(N=50, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 0.16213035583496094 (leap frog), 0.045989990234375 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a moderate number of particles (n=500)
pos_agg = run_simulation(N=500, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 25.875255584716797 (leap frog), 7.08690071105957 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a large number of particles (n=5000)
pos_agg = run_simulation(N=5000, T=0.125, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 44.27939558029175 (Euler-Cromer)

start_time = time() # Not using Barnes-Hut, for a very large number of particles (n=10000)
pos_agg = run_simulation(N=10000, T=0.05, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=False, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 77.6750750541687 (Euler-Cromer)


#### Using the Barnes-Hut algorithm all particle interactions

start_time = time() # Using Barnes-Hut, for a relatively small number of particles (n=50)
pos_agg = run_simulation(N=50, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 18.157663583755493 (leap frog), 2.2221713066101074 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a moderate number of particles (n=500)
pos_agg = run_simulation(N=500, T=2, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 104.12792992591858 (leap frog), 62.24120903015137 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a large number of particles (n=5000)
pos_agg = run_simulation(N=5000, T=0.125, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
print("\nDuration:",time()-start_time) # Duration: 68.89757561683655 (Euler-Cromer)

start_time = time() # Using Barnes-Hut, for a very large number of particles (n=10000)
pos_agg = run_simulation(N=10000, T=0.05, dt=0.01, softening=0.1, G=1, integrator=euler_cromer_integrator, use_BH=True, theta=1.5, random_state=111)
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


###
### Generating run-time plots for various values of theta compared to the standard approach
###
vals_of_n = np.array([5,50,100,250,500,1000,5000,10000,12000]) # The number of particles to include in the simulation
T_arr = np.array([0.20 if n <= 200 else 0.05 if n <= 2000 else 0.03 for n in vals_of_n]) # The duration of the simulation
theta_values = [0.5,1.5,2.5] # The values of theta to use in the Barnes Hut algorithm
integrator_method = leap_frog_integrator # Set an integrator to be used for the simulations

BH_results_summary = np.zeros([len(theta_values), len(vals_of_n)]) # Create a summary table to hold all the results

# Compute the runtime per iteration for a series of experiments using BH with differing values of theta
for j,theta in enumerate(theta_values):
    print("("+str(j+1)+"/"+str(len(theta_values))+") Evaluating theta="+str(theta))
    table_row = [] # Set up a list to hold the runtime values of the summary table for this particular value of theta
    for n,T in zip(vals_of_n,T_arr):
        start_time = time()
        pos_agg = run_simulation(N=n, T=T, dt=0.01, softening=0.1, G=1, integrator=integrator_method, use_BH=True, theta=theta, random_state=111)
        table_row.append((time()-start_time) / (pos_agg.shape[2]-1)) # Compute the time to compute per iteration
    BH_results_summary[j,:] = table_row # Record the runtimes for this value of theta into the overall summary table

# Compute the run-time per iteration using the standard non-BH approach
run_times = [] # Set up a list to hold the runtime values of the summary table
for n,T in zip(vals_of_n,T_arr):
    start_time = time()
    pos_agg = run_simulation(N=n, T=T, dt=0.01, softening=0.1, G=1, integrator=integrator_method, use_BH=False, random_state=111)
    run_times.append((time()-start_time) / (pos_agg.shape[2]-1)) # Compute the time to compute per iteration
run_times = np.array(run_times)

# Generate a comparison plot of run time as a function of input size N
plt.figure(figsize=(8,6))
for j,row in enumerate(BH_results_summary):
    plt.plot(vals_of_n,row,label="Theta="+str(theta_values[j]),zorder=3)
plt.plot(vals_of_n,run_times,label="Non BH",zorder=3)
plt.yscale("log");plt.xscale("log");plt.grid(color='lightgray',zorder=-3)
plt.ylabel("S / Iter (log)");plt.xlabel("Value of N (log)")
plt.title("Runtime Comparison Plot: Standard Approach vs Barnes-Hut")
plt.legend();plt.show()

###
### Generate plots of runtime vs scaling of N for each integrator
###

# The set of integrators to explore
integrators_list = [leap_frog_integrator, euler_integrator, euler_cromer_integrator,
                    euler_richardson_integrator, verlet_integrator, RK4_integrator]

vals_of_n = [10,150,200,300,500,1000] # The values of n to run for each integrator
integrator_results_summary = np.zeros([len(integrators_list), len(vals_of_n)]) # Create a summary table to hold all the results

for j, integrator in enumerate(integrators_list):
    run_times = [] # Set up a list to hold the runtime values of the summary table
    for N in vals_of_n:
        start_time = time()
        pos_agg = run_simulation(N=N, T=0.5, dt=0.01, softening=0.1, G=1, integrator=intergrator, use_BH=False, random_state=111)
        run_times.append((pos_agg.shape[2]-1) / (time()-start_time))  # Compute the time to compute per iteration
    integrator_results_summary[j,:] = run_times 

# Generate a comparison plot of run time as a function of input size N
plt.figure(figsize=(8,6))
for j,row in enumerate(integrator_results_summary):
    plt.plot(vals_of_n, row, zorder=3)
plt.grid(color='lightgray',zorder=-3);plt.ylabel("S / Iter");plt.xlabel("Value of N")
plt.yscale("log");plt.title("Runtime Comparison Plots for Integrators");plt.show()

# Summary of iterations / second for each method on various sized inputs
print(integrator_results_summary.round(0))
"""
[[5001  204  118   38   12    3]
 [4416  187  116   38   13    3]
 [5014  209  113   38   12    3]
 [5542  204  117   38   13    3]
 [4995  192  121   37   12    3]
 [4540  200  109   38   13    3]]
"""


def center_of_mass(pos, mass):
    return (mass.T @ pos)/mass.sum()

def calc_momentum(vel,mass):
    return (vel*mass).sum()

#def calc_angular_momentum(pos,vel,mass):
#    cm = center_of_mass(pos, mass) # Compute the center of mass for the system
#    n = pos - cm # Compute the normal vector for each particle
#    n_dot_n = (n**2).sum(axis=1) # Compute |n|^2 for each particle
#    vel_proj = pos - ((n * pos).sum(axis=1) /n_dot_n).reshape(-1,1)*n
#    #L = mass * vel_proj * np.linalg.norm(n,axis=1).reshape(-1,1) # Compute angular momentum
#    L = mass * np.linalg.norm(vel_proj,axis=1).reshape(-1,1) * np.linalg.norm(n,axis=1).reshape(-1,1) # Compute angular momentum
#    return L.sum()
    
def compute_angular_mom(pos,vel,mass):
    L = 0
    for i in range(len(mass)):
        L += mass[i]*np.cross(pos[i,:], vel[i,:])
    return L

def calc_angular_mom(pos,vel,mass):
    return mass.reshape(-1,1). T @ np.cross(pos, vel)

# https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Variational_Principles_in_Classical_Mechanics_(Cline)/02%3A_Review_of_Newtonian_Mechanics/2.09%3A_Angular_Momentum_of_a_Many-Body_System

N = 50
mass = 20.0*np.ones((N,1))/N  # Total mass of all particles together is 20, evenly split
pos  = np.random.randn(N,3)   # Randomly generate initial positions using N(0,1)
vel  = np.random.randn(N,3) 
vel -= np.mean(mass * vel, axis=0) / np.mean(mass)

pos_agg, vel_agg, realism_metrics = run_simulation(N=N, T=20, dt=0.01, softening=0.1, G=1, 
                                                   initial_conditions={"mass":mass,"vel":vel,"pos":pos},
                                          integrator=verlet_integrator, use_BH=False, 
                                          random_state=111, return_realism_metrics=True, return_velocity=True)

# Make an energy stability plot
time_axis = np.arange(len(realism_metrics["KE"]))*0.01;plot_simulation_energy(time_axis,realism_metrics["KE"],realism_metrics["PE"])

sys_momentum = [calc_momentum(vel_agg[:,:,i], mass) for i in range(vel_agg.shape[2])]

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_axis,sys_momentum, label="System Momentum",zorder=3)
ax.set_ylim(-1,1)
ax.legend();ax.grid(color='lightgray',zorder=-3)
ax.set_xlabel("Time");ax.set_ylabel("Momentum")
ax.set_title("System Total Momentum Over Time")

center_of_mass_arr = np.concatenate([center_of_mass(pos_agg[:,:,i], mass) for i in range(pos_agg.shape[2])]).T

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_axis,center_of_mass_arr[0,:], label="X",zorder=3)
ax.plot(time_axis,center_of_mass_arr[1,:], label="Y",zorder=3)
ax.plot(time_axis,center_of_mass_arr[2,:], label="Z",zorder=3)
ax.legend();ax.grid(color='lightgray',zorder=-3)
ax.set_xlabel("Time");ax.set_ylabel("Momentum")
ax.set_title("Center of Mass Coordinates Time Series")

sys_angular_momentum = np.stack([compute_angular_mom(pos_agg[:,:,i],vel_agg[:,:,i], mass) for i in range(vel_agg.shape[2])])

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_axis,sys_angular_momentum, label="X",zorder=3)
ax.legend();ax.grid(color='lightgray',zorder=-3)
ax.set_xlabel("Time");ax.set_ylabel("Angular Momentum")
ax.set_title("Total System Angular Momentum")

N = 3;v0=5
mass = np.array([50,20,20]).astype(np.float16).reshape(3,1)
pos = np.array([[0,0,0],[-2,0,0],[2,0,0]]).astype(np.float16)
vel = np.array([[0,0,0],[0,v0,0],[0,-v0,0]]).astype(np.float16)
pos_agg, vel_agg, realism_metrics = run_simulation(N=3, T=10, dt=0.01, softening=0.1, G=1, normalize_momentum=False,
                                         initial_conditions={"mass":mass,"pos":pos,"vel":vel},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111, return_realism_metrics=True,return_velocity=True)

time_axis = np.arange(len(realism_metrics["KE"]))*0.01

#sys_angular_momentum = [calc_angular_momentum(pos_agg[:,:,i],vel_agg[:,:,i], mass) for i in range(vel_agg.shape[2])]
sys_angular_momentum = np.concatenate([calc_angular_mom(pos_agg[:,:,i],vel_agg[:,:,i], mass) for i in range(vel_agg.shape[2])]).T

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(time_axis,sys_angular_momentum[0,:], label="X",zorder=3)
ax.plot(time_axis,sys_angular_momentum[1,:], label="Y",zorder=3)
ax.plot(time_axis,sys_angular_momentum[2,:], label="Z",zorder=3)
ax.legend();ax.grid(color='lightgray',zorder=-3)
ax.set_xlabel("Time");ax.set_ylabel("Angular Momentum")
ax.set_title("Total System Angular Momentum")

# https://math.stackexchange.com/questions/2545652/projection-of-vector-onto-the-plane
# Vector projection formula 
# Conserved quantities on n-body problems: https://physics.stackexchange.com/questions/4459/conserved-quantities-in-generalized-n-body-problem
# https://farside.ph.utexas.edu/teaching/301/lectures/node120.html


def polar_to_xy(r,theta):
    """
    Converts (r,theta) polar coordinate pairs into (x,y) Cartesian coordinates

    Parameters
    ----------
    r : float or np.array
        A collection of radius values in polar coordinates i.e. distance from origin.
    theta : float or np.array
        A collection of theta values in polar coordinates i.e. angle on the unit circle.

    Returns
    -------
    x : float or np.array
        A collection of x-coordinates for each particle in the Cartesian plane.
    y : float or np.array
        A collection of y-coordinates for each particle in the Cartesian plane.
    """
    return (r * np.cos(theta), r * np.sin(theta))