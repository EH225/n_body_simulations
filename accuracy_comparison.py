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
import pandas as pd
from tabulate import tabulate

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

# =============================================================================
# Get MSE and RMSE for each numerical solution v. analytical solution
# =============================================================================

### kepler_leap_mse & rmse & L2 norm
kepler_leap_mse = round(((pos_agg_kepler.astype(float)-pos_agg_leap_frog.astype(float))**2).sum(), 3)
kepler_leap_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_leap_frog.astype(float))**2)).sum(), 3)
kepler_leap_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_leap_frog)[0], 3)

### kepler_euler_mse & rmse
kepler_euler_mse = round(((pos_agg_kepler.astype(float)-pos_agg_euler.astype(float))**2).sum(), 3)
kepler_euler_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_euler.astype(float))**2)).sum(), 3)
kepler_euler_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_euler)[0], 3)

### kepler_euler_cromer_mse & rmse
kepler_euler_cromer_mse = round(((pos_agg_kepler.astype(float)-pos_agg_euler_cromer.astype(float))**2).sum(), 3)
kepler_euler_cromer_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_euler_cromer.astype(float))**2)).sum(), 3)
kepler_euler_cromer_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_euler_cromer)[0], 3)

### kepler_euler_rich_mse & rmse
kepler_euler_rich_mse = round(((pos_agg_kepler.astype(float)-pos_agg_euler_rich.astype(float))**2).sum(), 3)
kepler_euler_rich_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_euler_rich.astype(float))**2)).sum(), 3)
kepler_euler_rich_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_euler_rich)[0], 3)

### kepler_verlet_mse & rmse
kepler_verlet_mse = round(((pos_agg_kepler.astype(float)-pos_agg_verlet.astype(float))**2).sum(), 3)
kepler_verlet_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_verlet.astype(float))**2)).sum(), 3)
kepler_verlet_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_verlet)[0], 3)

### kepler_RK4_mse & rmse
kepler_RK4_mse = round(((pos_agg_kepler.astype(float)-pos_agg_RK4.astype(float))**2).sum(), 3)
kepler_RK4_rmse = round(np.sqrt(((pos_agg_kepler.astype(float)-pos_agg_RK4.astype(float))**2)).sum(), 3)
kepler_RK4_norm = round(np.linalg.norm(pos_agg_kepler - pos_agg_RK4)[0], 3)


# =============================================================================
# Put values in a table
# =============================================================================
table = [[kepler_leap_mse, kepler_leap_rmse, kepler_leap_norm], 
         [kepler_euler_mse, kepler_euler_rmse, kepler_euler_norm],
         [kepler_euler_cromer_mse, kepler_euler_cromer_rmse, kepler_euler_cromer_norm],
         [kepler_euler_rich_mse, kepler_euler_rich_rmse, kepler_euler_rich_norm],
         [kepler_verlet_mse, kepler_verlet_rmse, kepler_verlet_norm],
         [kepler_RK4_mse, kepler_RK4_rmse, kepler_RK4_norm]]
df = pd.DataFrame(table, columns = ['MSE', 'RMSE', 'L2-norm'], 
                  index=['Kepler v. leap-Frog', 
                         'Kepler v. Euler',
                         'Kepler v. Euler-Cromer',
                         'Kepler v. Euler-Richardson',
                         'Kepler v. Verlet',
                         'Kepler v. RK4'])
columns = ['Model Comparison', 'MSE', 'RMSE', 'L2-norm']
# print(df)
print(tabulate(df, headers=columns, tablefmt='fancy_grid', numalign="right", floatfmt=".2f"))

# plot the dataframe
for col in df.columns:
    df[col].plot(kind='bar', figsize=(12,8))
    plt.title(col)
    plt.show()