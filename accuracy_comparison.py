# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:09:55 2022

@author: alexh
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from integrators import *
from simulation_utilities import *
from kepler_v2 import *
from barnes_hut_algo import octant_node

N = 2 # number of particles 
T = 10
dt = 0.01
G = 1
r = 1
m1 = 50 # Mass of particle 1
m2 = 1 # Mass of particle 2
pos1 = np.array([0.0,0.0,0.0]) # Starting position of particle 1
pos2 = np.array([1.0,0.0,0.0]) # Starting position of particle 2
vel1 = np.array([0,0,0])
vel2 = np.array([0,np.sqrt(m2*G/r) ,0])
total_m = np.array([50, 1]).reshape(N,1)
total_pos = np.array([[0,0,0], [1,0,0]])

# kepler
pos_agg_kepler = kepler_pos(total_pos, total_m, T, dt, G)

# leap frog
pos_agg_leap_frog = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=leap_frog_integrator, use_BH=False, random_state=111)

# euler_integrator
pos_agg_euler = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=euler_integrator, use_BH=False, random_state=111)

# euler_cromer
pos_agg_euler_cromer = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=euler_cromer_integrator, use_BH=False, random_state=111)

# euler_richardson_integrator
pos_agg_euler_rich = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=euler_richardson_integrator, use_BH=False, random_state=111)

# verlet_integrator
pos_agg_verlet = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=verlet_integrator, use_BH=False, random_state=111)

# RK4_integrator
pos_agg_RK4 = run_simulation(N=N, T=T, dt=dt, softening=0.1, G=G,
                                         initial_conditions={"mass":np.array([m1, m2]).reshape(N,1),
                                                             "pos":np.array([pos1, pos2]),
                                                             "vel":np.array([vel1, vel2])},
                                         integrator=RK4_integrator, use_BH=False, random_state=111)

kepler_leap_diff = []
kepler_euler_diff = []
kepler_euler_cromer_diff = []
kepler_euler_rich_diff = []
kepler_verlet_diff = []
kepler_RK4_diff = []

for i in range(len(pos_agg_kepler[1][1])):
    ### leap frog
    kepler_leap_diff.append(np.linalg.norm(pos_agg_leap_frog[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_leap_diff_avg = np.average(np.array(kepler_leap_diff))
    
    ### euler
    kepler_euler_diff.append(np.linalg.norm(pos_agg_euler[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_euler_diff_avg = np.average(np.array(kepler_euler_diff))
    
    ### euler_cromer
    kepler_euler_cromer_diff.append(np.linalg.norm(pos_agg_euler_cromer[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_euler_cromer_diff_avg = np.average(np.array(kepler_euler_cromer_diff))
    
    ### euler_rich
    kepler_euler_rich_diff.append(np.linalg.norm(pos_agg_euler_rich[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_euler_rich_diff_avg = np.average(np.array(kepler_euler_rich_diff))
    
    ### verlet
    kepler_verlet_diff.append(np.linalg.norm(pos_agg_verlet[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_verlet_diff_avg = np.average(np.array(kepler_verlet_diff))
    
    ### RK4
    kepler_RK4_diff.append(np.linalg.norm(pos_agg_RK4[1][1][i] - pos_agg_kepler[1][1][i]))
    kepler_RK4_diff_avg = np.average(np.array(kepler_RK4_diff))
    
kepler_leap_diff_avg    
kepler_euler_diff_avg


